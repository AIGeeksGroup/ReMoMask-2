import codecs as cs
import logging
import os
from os.path import join as pjoin
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from omegaconf import OmegaConf
from Part_TMR.models.builder_bimoco import MoCoTMR
from Part_TMR.datasets.utils import whole2parts
from Part_TMR.scripts.utils import set_seed
logger = logging.getLogger(__name__)
min_motion_length = 40

@hydra.main(version_base=None, config_name='config', config_path='Part_TMR/conf')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.train.seed)
    encode_motion(cfg)
    print(f'----------- model.device: {cfg.device} ------------')

def prepare_test_model(cfg):
    tmr_model_path = cfg.checkpoints_dir
    text_encoder_alias = cfg.model.text_encoder
    text_encoder_trainable: bool = cfg.train.train_text_encoder
    motion_embedding_dims: int = cfg.model.motion_embedding_dims
    text_embedding_dims: int = cfg.model.text_embedding_dims
    projection_dims: int = cfg.model.projection_dims
    model = MoCoTMR(text_encoder_alias, text_encoder_trainable, motion_embedding_dims, text_embedding_dims, projection_dims, dropout=0.5 if cfg.dataset.dataset_name == 'HumanML3D' else 0.0, mode='t2m' if cfg.dataset.dataset_name == 'HumanML3D' else 'kit', temp=0.07, alpha=0.9996, config=cfg)
    if cfg.eval.use_best_model:
        model_path = pjoin(tmr_model_path, 'best_model.pt')
    else:
        model_path = pjoin(tmr_model_path, 'last_model.pt')
    print(model_path)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model

def encode_motion(cfg):
    fps = cfg.dataset.fps
    device = cfg.device
    data_root = cfg.dataset.data_root
    split_file = pjoin(data_root, cfg.dataset.train_split_filename)
    motion_dir = pjoin(data_root, 'new_joint_vecs')
    text_dir = pjoin(data_root, 'texts')
    print('Loading the model')
    model = prepare_test_model(cfg).to(cfg.device)
    print(f'----------- model.device: {model.device} ------------')
    model.eval()
    mean = np.load(pjoin(cfg.dataset.data_root, 'Mean.npy')).astype(np.float32)
    std = np.load(pjoin(cfg.dataset.data_root, 'Std.npy')).astype(np.float32)
    motion_embeddings = []
    text_embeddings = []
    caption_list = []
    motion_ids = []
    tag_lists = []
    for name in tqdm(open(split_file, 'r').readlines()):
        name = name.strip()
        motion_path = pjoin(motion_dir, name + '.npy')
        text_path = pjoin(text_dir, name + '.txt')
        if not os.path.exists(motion_path):
            logger.warning(f'Motion file missing: {motion_path}, skipping.')
            continue
        if not os.path.exists(text_path):
            logger.warning(f'Text file missing: {text_path}, skipping.')
            continue
        raw_motion = np.load(motion_path)
        if len(raw_motion) < min_motion_length:
            continue
        if np.isnan(raw_motion).any():
            continue
        with cs.open(text_path) as f:
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
                if f_tag == 0.0 and to_tag == 0.0:
                    motion = raw_motion
                else:
                    motion = raw_motion[int(f_tag * fps):int(to_tag * fps)]
                motion = torch.from_numpy(motion).to(torch.float)
                motion = (motion - mean) / std
                m_length = motion.shape[0]
                max_motion_length = cfg.dataset.max_motion_length
                if m_length >= max_motion_length:
                    idx = 0
                    motion = motion[idx:idx + max_motion_length]
                    m_length = max_motion_length
                else:
                    padding_len = max_motion_length - m_length
                    D = motion.shape[1]
                    padding_zeros = np.zeros((padding_len, D), dtype=np.float32)
                    motion = np.concatenate((motion, padding_zeros), axis=0)
                (Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm) = whole2parts(motion, mode='t2m' if cfg.dataset.dataset_name == 'HumanML3D' else 'kit')
                (Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm) = (Root.to(device).unsqueeze(0), R_Leg.to(device).unsqueeze(0), L_Leg.to(device).unsqueeze(0), Backbone.to(device).unsqueeze(0), R_Arm.to(device).unsqueeze(0), L_Arm.to(device).unsqueeze(0))
                motions = [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]
                with torch.inference_mode():
                    motion_features = model.encode_motion(motions)['global']
                    motion_features = motion_features / motion_features.norm(dim=1, keepdim=True)
                    motion_latent = motion_features.cpu().numpy()
                with torch.inference_mode():
                    texts_token = model.tokenize(caption).to(device)
                    text_features = model.encode_text(texts_token)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)
                    text_latent = text_features.cpu().numpy()
                motion_embeddings.append(motion_latent)
                text_embeddings.append(text_latent)
                caption_list.append(caption)
                motion_ids.append(f'{name}_{idx}')
                tag_lists.append([f_tag, to_tag])
    motion_embeddings = np.array(motion_embeddings)
    text_embeddings = np.array(text_embeddings)
    output_folder = cfg.rag.database_path
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, 'all_captions.npy')
    np.save(path, caption_list)
    path = os.path.join(output_folder, 'motion_ids.npy')
    np.save(path, motion_ids)
    path = os.path.join(output_folder, 'tag_lists.npy')
    np.save(path, tag_lists)
    path = os.path.join(output_folder, 'encoded_motions.npy')
    np.save(path, motion_embeddings)
    print(f'Encoding done, motion latent saved in:\n{path}')
    path = os.path.join(output_folder, 'encoded_texts.npy')
    np.save(path, text_embeddings)
    print(f'Encoding done, text latent saved in:\n{path}')
if __name__ == '__main__':
    main()
