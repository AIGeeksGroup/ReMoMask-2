import argparse
import codecs as cs
import os
from os.path import join as pjoin
import clip
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from models.vq.model import RVQVAE
from utils.get_opt import get_opt
MIN_MOTION_LENGTH = 40
N_JOINTS = 22
N_FEAT = 12
V1_ID_MAX_MOTION_LENGTH = 224

def motion_to_x2d(motion: np.ndarray, n_j: int=N_JOINTS) -> np.ndarray:
    T = motion.shape[0]
    x2 = motion[:, 4:4 + (n_j - 1) * 3]
    x3 = motion[:, 4 + (n_j - 1) * 3:4 + (n_j - 1) * 9]
    x4 = motion[:, 4 + (n_j - 1) * 9:4 + (n_j - 1) * 9 + n_j * 3]
    x_pos = x2.reshape(T, n_j - 1, 3)
    x_rot = x3.reshape(T, n_j - 1, 6)
    x_speed = x4.reshape(T, n_j, 3)
    x_joints = np.zeros([T, n_j, N_FEAT], dtype=np.float32)
    x_joints[:, 1:, :3] = x_pos
    x_joints[:, 1:, 3:9] = x_rot
    x_joints[:, :, 9:12] = x_speed
    return x_joints

def load_vq_model(vq_opt, device):
    vq_model = RVQVAE(vq_opt, vq_opt.dim_pose, vq_opt.down_t, vq_opt.stride_t, vq_opt.width, vq_opt.depth, vq_opt.dilation_growth_rate, vq_opt.vq_act, vq_opt.vq_norm)
    ckpt_path = pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    vq_model.load_state_dict(ckpt['vq_model'])
    vq_model.to(device)
    vq_model.eval()
    print(f'Loaded VQ-VAE: {ckpt_path}')
    print(f"  code_dim2d = {vq_opt.code_dim2d}, ep = {ckpt.get('ep', '?')}")
    return vq_model

def encode_ze_batch(vq_model, x2d_batch, device, n_joints=N_JOINTS):
    x2d = torch.from_numpy(x2d_batch).float().to(device)
    with torch.inference_mode():
        if n_joints == 21:
            x2d_padded = F.pad(x2d, (0, 0, 1, 2))
        else:
            x2d_padded = F.pad(x2d, (0, 0, 1, 1))
        z_e = vq_model.encoder2d(rearrange(x2d_padded, 'b t j d -> b d t j'))
        z_e_pooled = z_e.mean(dim=[2, 3])
        z_e_pooled = F.normalize(z_e_pooled, dim=1)
    return z_e_pooled.cpu().numpy()

def prepare_motion(raw_motion, mean, std, f_tag, to_tag, max_motion_length, n_joints=N_JOINTS, fps=20):
    if f_tag == 0.0 and to_tag == 0.0:
        motion = raw_motion
    else:
        motion = raw_motion[int(f_tag * fps):int(to_tag * fps)]
    motion_norm = ((motion - mean) / std).astype(np.float32)
    m_length = motion_norm.shape[0]
    if m_length >= max_motion_length:
        motion_norm = motion_norm[:max_motion_length]
    else:
        pad_len = max_motion_length - m_length
        motion_norm = np.concatenate([motion_norm, np.zeros((pad_len, motion_norm.shape[1]), dtype=np.float32)], axis=0)
    return motion_to_x2d(motion_norm, n_j=n_joints)

def main():
    parser = argparse.ArgumentParser(description='Build z_e retrieval database')
    parser.add_argument('--dataset_name', type=str, default='humanml3d', choices=['humanml3d', 'kit'])
    parser.add_argument('--vq_name', type=str, default='pretrain_vq')
    parser.add_argument('--output_dir', type=str, default='database_ze')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--max_motion_length', type=int, default=196)
    parser.add_argument('--vq_batch_size', type=int, default=64)
    args = parser.parse_args()
    device = args.device
    data_root = args.dataset_root or ('dataset/KIT-ML' if args.dataset_name == 'kit' else 'dataset/HumanML3D')
    fps = 12.5 if args.dataset_name == 'kit' else 20
    split_file = pjoin(data_root, 'train.txt')
    motion_dir = pjoin(data_root, 'new_joint_vecs')
    text_dir = pjoin(data_root, 'texts')
    opt_path = pjoin('logs', args.dataset_name, args.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, device=device)
    vq_model = load_vq_model(vq_opt, device)
    code_dim2d = vq_opt.code_dim2d
    print(f'code_dim2d = {code_dim2d}')
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))
    motion_embeddings = []
    caption_list = []
    motion_ids = []
    tag_lists = []
    x2d_batch = []
    names = open(split_file, 'r').readlines()
    print(f'Processing {len(names)} motion files ...')

    def flush_batch():
        if not x2d_batch:
            return
        x2d_arr = np.stack(x2d_batch, axis=0)
        z_e = encode_ze_batch(vq_model, x2d_arr, device, n_joints=vq_opt.joints_num)
        for i in range(z_e.shape[0]):
            motion_embeddings.append(z_e[i])
        x2d_batch.clear()
    for name in tqdm(names, desc='Encoding'):
        name = name.strip()
        if not name:
            continue
        motion_path = pjoin(motion_dir, name + '.npy')
        text_path = pjoin(text_dir, name + '.txt')
        if not os.path.exists(motion_path) or not os.path.exists(text_path):
            continue
        raw_motion = np.load(motion_path)
        if len(raw_motion) < MIN_MOTION_LENGTH or np.isnan(raw_motion).any():
            continue
        with cs.open(text_path, encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            for (idx, line) in enumerate(lines):
                line_split = line.strip().split('#')
                if len(line_split) < 4 or len(line_split[0]) == 0:
                    continue
                caption = line_split[0]
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag
                x2d_np = prepare_motion(raw_motion, mean, std, f_tag, to_tag, args.max_motion_length, n_joints=vq_opt.joints_num, fps=fps)
                if f_tag == 0.0 and to_tag == 0.0:
                    crop_len = len(raw_motion)
                else:
                    crop_len = len(raw_motion[int(f_tag * fps):int(to_tag * fps)])
                id_idx = 0 if crop_len >= V1_ID_MAX_MOTION_LENGTH else idx
                x2d_batch.append(x2d_np)
                caption_list.append(caption)
                motion_ids.append(f'{name}_{id_idx}')
                tag_lists.append([f_tag, to_tag])
                if len(x2d_batch) >= args.vq_batch_size:
                    flush_batch()
    flush_batch()
    total = len(caption_list)
    print(f'Total samples: {total}')
    motion_embeddings = np.stack(motion_embeddings, axis=0)[:, np.newaxis, :]
    print(f'motion_embeddings: {motion_embeddings.shape}')
    print('Encoding captions with CLIP ViT-B/32 ...')
    (clip_model, _) = clip.load('ViT-B/32', device=device, jit=False)
    clip_model.eval()
    text_feats_list = []
    clip_bs = 256
    for i in tqdm(range(0, total, clip_bs), desc='CLIP encoding'):
        batch = caption_list[i:i + clip_bs]
        with torch.inference_mode():
            text_ids = clip.tokenize(batch, truncate=True).to(device)
            text_feat = clip_model.encode_text(text_ids).float()
            text_feat = F.normalize(text_feat, dim=1)
        text_feats_list.append(text_feat.cpu().numpy())
    text_embeddings = np.concatenate(text_feats_list, axis=0)[:, np.newaxis, :]
    print(f'text_embeddings: {text_embeddings.shape}')
    del clip_model
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(pjoin(args.output_dir, 'encoded_motions.npy'), motion_embeddings)
    np.save(pjoin(args.output_dir, 'encoded_texts_clip.npy'), text_embeddings)
    np.save(pjoin(args.output_dir, 'all_captions.npy'), caption_list)
    np.save(pjoin(args.output_dir, 'motion_ids.npy'), motion_ids)
    np.save(pjoin(args.output_dir, 'tag_lists.npy'), tag_lists)
    print(f'Database saved to: {args.output_dir}')
if __name__ == '__main__':
    main()
