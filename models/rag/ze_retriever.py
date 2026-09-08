import os
import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ZeRetriever(nn.Module):

    def __init__(self, database_path: str, query_projector_path: str, top_k: int=1, num_retrieval: int=10):
        super().__init__()
        self.top_k = top_k
        self.num_retrieval = num_retrieval
        print(f'Loading z_e retrieval database from {database_path}')
        self.motion_names = np.load(os.path.join(database_path, 'motion_ids.npy'))
        self.captions = np.load(os.path.join(database_path, 'all_captions.npy'))
        motion_features = np.load(os.path.join(database_path, 'encoded_motions.npy'))[:, 0, :]
        text_features_path = os.path.join(database_path, 'encoded_texts_clip.npy')
        text_features = np.load(text_features_path)[:, 0, :]
        self.register_buffer('motion_features', torch.tensor(motion_features, dtype=torch.float32))
        self.register_buffer('text_features', torch.tensor(text_features, dtype=torch.float32))
        self.code_dim = self.motion_features.shape[1]
        assert self.motion_features.size(0) == self.text_features.size(0) == len(self.motion_names) == len(self.captions), 'Database size mismatch between motion features, text features, names, and captions'
        print(f'  Database loaded: {self.motion_features.shape[0]} samples, motion dim={self.code_dim}, text dim={self.text_features.shape[1]}')
        from models.rag.query_projector import load_projector
        self.query_projector = load_projector(query_projector_path, map_location='cpu')
        self.query_projector.eval()
        assert self.query_projector.ze_dim == self.code_dim, 'Projector and motion feature dimensions differ'
        assert self.query_projector.clip_dim == self.text_features.shape[1], 'Projector and text feature dimensions differ'
        self._clip_model = None
        self._clip_device = None

    def _ensure_clip(self, device):
        if self._clip_model is None or self._clip_device != device:
            (self._clip_model, _) = clip.load('ViT-B/32', device=device, jit=False)
            self._clip_model.eval()
            self._clip_device = device

    def tokenize(self, text):
        return clip.tokenize(text, truncate=True)

    def encode_text(self, text_ids):
        device = self.motion_features.device
        self._ensure_clip(device)
        text_ids = text_ids.to(device)
        with torch.inference_mode():
            text_embedding = self._clip_model.encode_text(text_ids).float()
        return text_embedding

    def encode_query_text(self, caption: str) -> torch.Tensor:
        device = self.motion_features.device
        self._ensure_clip(device)
        with torch.inference_mode():
            text_ids = clip.tokenize([caption], truncate=True).to(device)
            text_feat = self._clip_model.encode_text(text_ids).float()
            text_feat = F.normalize(text_feat, dim=1)
        with torch.inference_mode():
            query_vec = self.query_projector(text_feat)
            query_vec = F.normalize(query_vec, dim=1)
        return query_vec

    def cal_similarity(self, query: torch.Tensor, database: torch.Tensor) -> torch.Tensor:
        assert query.shape[-1] == database.shape[-1], 'Query and database dimensions differ'
        query = F.normalize(query, dim=-1)
        database = F.normalize(database, dim=-1)
        return torch.mm(query, database.t()).squeeze(0)

    def get_knn_samples(self, caption: str, k: int):
        query_vec = self.encode_query_text(caption)
        score = self.cal_similarity(query_vec, self.motion_features)
        indexes = torch.argsort(score, descending=True).cpu().numpy()
        nn_indexes = []
        nn_names = []
        for index in indexes:
            motion_name_ = self.motion_names[index].split('_')[0]
            if motion_name_ in nn_names:
                continue
            nn_indexes.append(index)
            nn_names.append(motion_name_)
            if len(nn_indexes) >= self.num_retrieval:
                break
        return nn_indexes[:k]

    def retrieve(self, caption: str) -> List[int]:
        return self.get_knn_samples(caption, k=self.top_k)

    def forward(self, captions: List[str]) -> dict:
        b = len(captions)
        all_indexes = []
        for b_ix in range(b):
            batch_indexes = self.retrieve(captions[b_ix])
            all_indexes.append(batch_indexes)
        all_indexes = torch.tensor(all_indexes, dtype=torch.long)
        (b, k) = all_indexes.shape
        flat_indexes = all_indexes.view(-1)
        selected_motions = self.motion_features[flat_indexes]
        selected_texts = self.text_features[flat_indexes]
        with torch.no_grad():
            selected_texts = self.query_projector(selected_texts)
        all_motions_feature = selected_motions.view(b, k, 1, -1)
        all_texts_feature = selected_texts.view(b, k, 1, -1)
        re_dict = {'re_motion': all_motions_feature, 're_text': all_texts_feature}
        return re_dict
