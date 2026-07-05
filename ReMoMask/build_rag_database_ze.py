"""
Build retrieval database using RVQVAE encoder z_e (pre-quantization latent).

Replaces Part_TMR-based database with z_e vectors from the VQ-VAE 2D branch.
Motion encoding: encoder2d -> mean pool (time + joints) -> L2 normalize -> (N, 1, 1024).
Text encoding: CLIP ViT-B/32 text features for cross-modal placeholder (until query projector is trained).

With --use_quantized, the encoder2d latent is additionally routed through the frozen
residual quantizer stack (quantizer2d) and the SUMMED multi-level quantized
reconstruction z_q is used instead of the raw pre-quantization z_e (same shape,
same pooling + L2 norm). See E04 (z_e vs z_q retrieval key) in EXPERIMENTS-SPEC.md.

Usage:
    cd ReMoMask
    python build_rag_database_ze.py \
        --vq_name pretrain_vq \
        --output_dir database_ze \
        --device cuda:0

    # quantized (z_q) variant for E04:
    python build_rag_database_ze.py \
        --vq_name pretrain_vq \
        --use_quantized \
        --device cuda:0
"""

import argparse
import codecs as cs
import json
import logging
import os
import shutil
import sys
import time
from os.path import join as pjoin

import clip
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.vq.model import RVQVAE
from utils.get_opt import get_opt

logger = logging.getLogger(__name__)

# HumanML3D constants
MIN_MOTION_LENGTH = 40
FPS = 20
UNIT_LENGTH = 4
N_JOINTS = 22
N_FEAT = 12  # per-joint feature dim for 2D branch


def motion_to_x2d(motion: np.ndarray, n_j: int = N_JOINTS) -> np.ndarray:
    """Convert 263-dim HumanML3D motion to per-joint (T, J, 12) tensor.
    Same logic as data/t2m_dataset.py and scripts/analyze_ze.py."""
    T = motion.shape[0]
    x2 = motion[:, 4: 4 + (n_j - 1) * 3]
    x3 = motion[:, 4 + (n_j - 1) * 3: 4 + (n_j - 1) * 9]
    x4 = motion[:, 4 + (n_j - 1) * 9: 4 + (n_j - 1) * 9 + n_j * 3]

    x_pos = x2.reshape(T, n_j - 1, 3)
    x_rot = x3.reshape(T, n_j - 1, 6)
    x_speed = x4.reshape(T, n_j, 3)

    x_joints = np.zeros([T, n_j, N_FEAT], dtype=np.float32)
    x_joints[:, 1:, :3] = x_pos
    x_joints[:, 1:, 3:9] = x_rot
    x_joints[:, :, 9:12] = x_speed
    return x_joints


def load_vq_model(vq_opt, device):
    """Load RVQVAE from checkpoint (same pattern as demo.py / eval_vq.py)."""
    vq_model = RVQVAE(
        vq_opt,
        vq_opt.dim_pose,
        vq_opt.down_t,
        vq_opt.stride_t,
        vq_opt.width,
        vq_opt.depth,
        vq_opt.dilation_growth_rate,
        vq_opt.vq_act,
        vq_opt.vq_norm,
    )
    ckpt_path = pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name,
                      'model', 'net_best_fid.tar')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    vq_model.to(device)
    vq_model.eval()
    print(f"Loaded VQ-VAE: {ckpt_path}")
    print(f"  code_dim2d = {vq_opt.code_dim2d}, ep = {ckpt.get('ep', '?')}")
    return vq_model


def encode_ze_batch(vq_model, x2d_batch, device, use_quantized=False):
    """Encode a batch of x2d tensors to z_e (or z_q) vectors.

    Args:
        vq_model: frozen RVQVAE
        x2d_batch: (B, T, 22, 12) numpy array
        device: torch device
        use_quantized: if True, route the encoder2d latent through the residual
            quantizer stack (quantizer2d) and use the summed multi-level
            quantized reconstruction (z_q) instead of the raw z_e.

    Returns:
        z_e (or z_q): (B, code_dim2d) numpy, L2-normalised
    """
    x2d = torch.from_numpy(x2d_batch).to(device)
    with torch.inference_mode():
        x2d_padded = F.pad(x2d, (0, 0, 1, 1))  # (B, T, 24, 12)
        z_e = vq_model.encoder2d(
            rearrange(x2d_padded, 'b t j d -> b d t j')
        )  # (B, code_dim2d, T/4, 6)

        if use_quantized:
            # Route z_e through quantizer2d and take the SUM of the per-level
            # quantized outputs -- this is the standard RVQ reconstruction that
            # RVQVAE.forward()/decoder2d would consume, before the cross-joint
            # `linear_merge` mixing (a decode-only step that would change the
            # latent's shape/semantics away from a per-joint code comparable to
            # z_e, and is not part of "the quantized latent" itself).
            #
            # We call `quantizer2d.quantize(x, return_latent=True)` -- the same
            # API RVQVAE.encode() uses for token-id extraction -- rather than
            # `quantizer2d.forward()` (used inside RVQVAE.forward() during
            # training), because `.quantize()` is deterministic at eval time
            # (no quantize_dropout / sample_codebook_temp, both of which only
            # exist for training-time regularisation) and returns the
            # per-residual-level codes explicitly, so the "sum over all
            # residual levels" required here is done explicitly rather than
            # hidden inside the module.
            B_, D_, T_, J_ = z_e.shape
            z_e_flat = rearrange(z_e, 'b d t j -> (b j) d t')
            _, all_codes = vq_model.quantizer2d.quantize(z_e_flat, return_latent=True)
            # all_codes: (num_quantizers, B*J, D, T) -> sum over residual levels
            z_q_flat = all_codes.sum(dim=0)  # (B*J, D, T)
            z_e = rearrange(z_q_flat, '(b j) d t -> b d t j', j=J_)

        z_e_pooled = z_e.mean(dim=[2, 3])  # (B, code_dim2d)
        z_e_pooled = F.normalize(z_e_pooled, dim=1)
    return z_e_pooled.cpu().numpy()


def prepare_motion(raw_motion, mean, std, f_tag, to_tag, max_motion_length):
    """Crop, normalise, pad a single motion. Returns (max_T, 22, 12) or None."""
    if f_tag == 0.0 and to_tag == 0.0:
        motion = raw_motion
    else:
        motion = raw_motion[int(f_tag * FPS): int(to_tag * FPS)]

    if len(motion) < MIN_MOTION_LENGTH:
        return None

    motion_norm = ((motion - mean) / std).astype(np.float32)
    m_length = motion_norm.shape[0]
    if m_length >= max_motion_length:
        motion_norm = motion_norm[:max_motion_length]
    else:
        pad_len = max_motion_length - m_length
        motion_norm = np.concatenate(
            [motion_norm, np.zeros((pad_len, 263), dtype=np.float32)], axis=0
        )
    return motion_to_x2d(motion_norm, n_j=N_JOINTS)


def main():
    parser = argparse.ArgumentParser(description="Build z_e retrieval database")
    parser.add_argument('--vq_name', type=str, default='pretrain_vq',
                        help='VQ checkpoint name under logs/humanml3d/')
    parser.add_argument('--output_dir', type=str, default='database_ze',
                        help='Output directory for database files')
    parser.add_argument('--use_quantized', action='store_true',
                        help='Use the quantized RVQ reconstruction z_q (sum over all '
                             'residual levels of quantizer2d) instead of the raw '
                             'pre-quantization z_e (E04 ablation).')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset_root', type=str, default='dataset/HumanML3D',
                        help='HumanML3D data root')
    parser.add_argument('--max_motion_length', type=int, default=196)
    parser.add_argument('--vq_batch_size', type=int, default=64,
                        help='Batch size for VQ-VAE encoding')
    args = parser.parse_args()

    if args.use_quantized and args.output_dir == parser.get_default('output_dir'):
        args.output_dir = 'database_zq'
        print(f"Notice: --use_quantized set with default --output_dir; "
              f"switching output_dir to '{args.output_dir}' (pass --output_dir "
              f"explicitly to override).")

    t_start = time.time()

    device = args.device
    data_root = args.dataset_root
    split_file = pjoin(data_root, 'train.txt')
    motion_dir = pjoin(data_root, 'new_joint_vecs')
    text_dir = pjoin(data_root, 'texts')

    # ---- 1. Load VQ-VAE ----
    opt_path = pjoin('logs', 'humanml3d', args.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, device=device)
    vq_model = load_vq_model(vq_opt, device)
    code_dim2d = vq_opt.code_dim2d
    print(f"code_dim2d = {code_dim2d}")

    # ---- 2. Load Mean / Std ----
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))

    # ---- 3. Streaming encode: iterate files, batch encode z_e ----
    # We accumulate x2d into batches, encode, then discard x2d to save memory.
    motion_embeddings = []   # small: each (1, code_dim2d)
    caption_list = []
    motion_ids = []
    tag_lists = []

    x2d_batch = []  # temporary batch buffer

    names = open(split_file, 'r').readlines()
    print(f"Processing {len(names)} motion files ...")

    def flush_batch():
        """Encode accumulated x2d batch and store results."""
        if not x2d_batch:
            return
        x2d_arr = np.stack(x2d_batch, axis=0)  # (B, T, 22, 12)
        z_e = encode_ze_batch(vq_model, x2d_arr, device,
                              use_quantized=args.use_quantized)  # (B, code_dim2d)
        for i in range(z_e.shape[0]):
            motion_embeddings.append(z_e[i])  # (code_dim2d,)
        x2d_batch.clear()

    for name in tqdm(names, desc="Encoding"):
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
            for idx, line in enumerate(lines):
                line_split = line.strip().split('#')
                if len(line_split) < 4:
                    continue
                caption = line_split[0]
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                x2d_np = prepare_motion(raw_motion, mean, std, f_tag, to_tag,
                                        args.max_motion_length)
                if x2d_np is None:
                    continue

                x2d_batch.append(x2d_np)
                caption_list.append(caption)
                motion_ids.append(f"{name}_{idx}")
                tag_lists.append([f_tag, to_tag])

                if len(x2d_batch) >= args.vq_batch_size:
                    flush_batch()

    # Flush remaining
    flush_batch()

    total = len(caption_list)
    print(f"Total samples: {total}")

    # Stack motion embeddings: (N, code_dim2d) -> (N, 1, code_dim2d)
    motion_embeddings = np.stack(motion_embeddings, axis=0)[:, np.newaxis, :]
    print(f"motion_embeddings: {motion_embeddings.shape}")

    # ---- 4. Batch encode captions with CLIP ----
    print("Encoding captions with CLIP ViT-B/32 ...")
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    text_feats_list = []
    clip_bs = 256
    for i in tqdm(range(0, total, clip_bs), desc="CLIP encoding"):
        batch = caption_list[i:i + clip_bs]
        with torch.inference_mode():
            text_ids = clip.tokenize(batch, truncate=True).to(device)
            text_feat = clip_model.encode_text(text_ids).float()
            text_feat = F.normalize(text_feat, dim=1)
        text_feats_list.append(text_feat.cpu().numpy())

    text_embeddings = np.concatenate(text_feats_list, axis=0)[:, np.newaxis, :]  # (N, 1, 512)
    print(f"text_embeddings: {text_embeddings.shape}")

    del clip_model

    # ---- 5. Save database ----
    os.makedirs(args.output_dir, exist_ok=True)

    np.save(pjoin(args.output_dir, 'encoded_motions.npy'), motion_embeddings)
    np.save(pjoin(args.output_dir, 'encoded_texts_clip.npy'), text_embeddings)
    np.save(pjoin(args.output_dir, 'all_captions.npy'), caption_list)
    np.save(pjoin(args.output_dir, 'motion_ids.npy'), motion_ids)
    np.save(pjoin(args.output_dir, 'tag_lists.npy'), tag_lists)

    # ---- 6. Copy motion_tokens from existing database if available ----
    existing_tokens = 'database/motion_tokens.npy'
    if os.path.exists(existing_tokens):
        dst = pjoin(args.output_dir, 'motion_tokens.npy')
        shutil.copy2(existing_tokens, dst)
        print(f"Copied motion_tokens.npy from {existing_tokens}")
    else:
        tokens_dir = pjoin(data_root, 'TOKENS')
        if os.path.isdir(tokens_dir):
            motion_token_dict = {}
            for i in tqdm(range(total), desc="Loading tokens"):
                motion_name = motion_ids[i]
                f_tag, to_tag = tag_lists[i]
                token_path = pjoin(tokens_dir, motion_name.split('_')[0] + '.npy')
                if not os.path.exists(token_path):
                    continue
                motion_token = np.load(token_path)[0]
                if f_tag != 0.0 or to_tag != 0.0:
                    if int(f_tag * FPS / UNIT_LENGTH) < int(to_tag * FPS / UNIT_LENGTH):
                        motion_token = motion_token[
                            int(f_tag * FPS / UNIT_LENGTH): int(to_tag * FPS / UNIT_LENGTH)
                        ]
                motion_token_dict[motion_name] = motion_token
            np.save(pjoin(args.output_dir, 'motion_tokens.npy'), motion_token_dict)
            print(f"Saved motion_tokens.npy ({len(motion_token_dict)} entries)")
        else:
            print(f"Warning: No TOKENS directory at {tokens_dir}, skipping motion_tokens.")

    # ---- 7. Save metadata ----
    metadata = {
        'code_dim2d': int(code_dim2d),
        'encoder_type': 'rvqvae_2d_quantized' if args.use_quantized else 'rvqvae_2d',
        'use_quantized': bool(args.use_quantized),
        'vq_name': args.vq_name,
        'dataset_root': args.dataset_root,
        'num_samples': total,
        'motion_embedding_shape': list(motion_embeddings.shape),
        'text_embedding_shape': list(text_embeddings.shape),
        'text_encoder': 'clip_vit_b32',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'note': 'text features are raw CLIP embeddings; replace after query projector training (LA-03)',
    }
    with open(pjoin(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    t_end = time.time()
    duration = t_end - t_start
    print(f"\nDone. Duration: {duration:.2f}s ({duration/60:.2f}min)")
    print(f"Database saved to: {args.output_dir}/")
    print(f"  encoded_motions.npy: {motion_embeddings.shape}")
    print(f"  encoded_texts_clip.npy: {text_embeddings.shape}")
    print(f"  metadata.json: code_dim2d={code_dim2d}, encoder_type={metadata['encoder_type']}")


if __name__ == '__main__':
    main()
