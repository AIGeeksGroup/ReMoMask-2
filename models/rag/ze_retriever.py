"""
ZeRetriever: retriever using RVQVAE encoder z_e space.

Loads pre-built database_ze/ (from build_rag_database_ze.py) and performs
cosine similarity retrieval in the z_e latent space.

Before query projector is trained (LA-03), text queries use raw CLIP
embeddings matched against z_e motion features (cross-space, suboptimal
but functional). After LA-03, the projector maps CLIP text -> z_e space.

Interface matches MocoTmrRetriever.forward() -> re_dict format.
"""

import os
import sys

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class ZeRetriever(nn.Module):
    """Retriever in RVQVAE z_e latent space.

    Args:
        database_path: path to database_ze/ directory
        query_projector_path: path to trained projector checkpoint (None = use raw CLIP)
        top_k: number of retrieval results per caption
        num_retrieval: candidate pool size before top_k selection
        use_shuffle: whether to shuffle candidate pool before taking top_k
    """

    def __init__(
        self,
        database_path: str = 'database_ze',
        query_projector_path: Optional[str] = None,
        top_k: int = 2,
        num_retrieval: int = 10,
        use_shuffle: bool = False,
    ):
        super().__init__()

        self.top_k = top_k
        self.num_retrieval = num_retrieval
        self.use_shuffle = use_shuffle

        # ---- Load database ----
        print(f"Loading z_e retrieval database from {database_path}")
        self.motion_names = np.load(os.path.join(database_path, 'motion_ids.npy'))
        self.captions = np.load(os.path.join(database_path, 'all_captions.npy'))

        motion_features = np.load(
            os.path.join(database_path, 'encoded_motions.npy')
        )[:, 0, :]  # (N, code_dim2d)

        # Text features: CLIP 512-d (placeholder until projector is ready)
        text_features_path = os.path.join(database_path, 'encoded_texts_clip.npy')
        text_features = np.load(text_features_path)[:, 0, :]  # (N, 512)

        self.register_buffer('motion_features', torch.tensor(motion_features, dtype=torch.float32))
        self.register_buffer('text_features', torch.tensor(text_features, dtype=torch.float32))

        self.code_dim = self.motion_features.shape[1]  # 1024

        assert (
            self.motion_features.size(0)
            == self.text_features.size(0)
            == len(self.motion_names)
            == len(self.captions)
        ), "Database size mismatch between motion features, text features, names, and captions"

        print(f"  Database loaded: {self.motion_features.shape[0]} samples, "
              f"motion dim={self.code_dim}, text dim={self.text_features.shape[1]}")

        # ---- Load query projector ----
        self.query_projector = None
        if query_projector_path is not None and os.path.exists(query_projector_path):
            print(f"Loading query projector from {query_projector_path}")
            self.query_projector = torch.load(query_projector_path, map_location='cpu')
            self.query_projector.eval()
            print("  Query projector loaded.")
        else:
            print("  No query projector loaded; using raw CLIP text for retrieval "
                  "(cross-space matching, suboptimal).")

        # ---- Load CLIP for text encoding ----
        self._clip_model = None
        self._clip_device = None

    def _ensure_clip(self, device):
        """Lazy-load CLIP model on first use."""
        if self._clip_model is None or self._clip_device != device:
            self._clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
            self._clip_model.eval()
            self._clip_device = device

    def encode_query_text(self, caption: str) -> torch.Tensor:
        """Encode a text caption to the retrieval space.

        If query_projector is available: CLIP text -> projector -> z_e space (1024-d)
        Otherwise: CLIP text embedding (512-d), matched against z_e via cosine sim
        (cross-space, functional but noisy).

        Returns:
            query_vec: (1, D) where D = code_dim if projector, else 512
        """
        device = self.motion_features.device
        self._ensure_clip(device)

        with torch.inference_mode():
            text_ids = clip.tokenize([caption], truncate=True).to(device)
            text_feat = self._clip_model.encode_text(text_ids).float()  # (1, 512)
            text_feat = F.normalize(text_feat, dim=1)

        if self.query_projector is not None:
            with torch.inference_mode():
                query_vec = self.query_projector(text_feat)  # (1, code_dim)
                query_vec = F.normalize(query_vec, dim=1)
            return query_vec
        else:
            return text_feat  # (1, 512) — cross-space fallback

    def cal_similarity(self, query: torch.Tensor, database: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between query and database vectors.

        When projector is not available, query is 512-d and database is 1024-d.
        In this case we use the first 512 dims of database features for matching
        (a crude heuristic; proper alignment comes from the projector in LA-03).
        """
        if query.shape[-1] != database.shape[-1]:
            # Cross-space fallback: truncate database to match query dim
            # This is intentionally suboptimal; projector will fix this
            min_dim = min(query.shape[-1], database.shape[-1])
            query = query[..., :min_dim]
            database = database[..., :min_dim]

        query = F.normalize(query, dim=-1)
        database = F.normalize(database, dim=-1)
        return torch.mm(query, database.t()).squeeze(0)  # (N,)

    def shuffle_samples(self, nn_indexes, nn_names):
        """Shuffle candidate pool (same logic as MocoTmrRetriever)."""
        paired = list(zip(nn_indexes, nn_names))
        np.random.shuffle(paired)
        shuffled_indexes, shuffled_names = zip(*paired) if paired else ([], [])
        return list(shuffled_indexes), list(shuffled_names)

    def get_knn_samples(self, caption: str, k: int = 2, nn_names=None):
        """Get top-k nearest neighbours for a caption.

        De-duplicates by motion ID (same as MocoTmrRetriever).
        """
        query_vec = self.encode_query_text(caption)  # (1, D)
        score = self.cal_similarity(query_vec, self.motion_features)  # (N,)
        indexes = torch.argsort(score, descending=True).cpu().numpy()

        nn_indexes = []
        nn_names = [] if nn_names is None else nn_names
        for index in indexes:
            motion_name_ = self.motion_names[index].split('_')[0]
            if motion_name_ in nn_names:
                continue
            nn_indexes.append(index)
            nn_names.append(motion_name_)
            if len(nn_indexes) >= self.num_retrieval:
                break

        if self.use_shuffle:
            nn_indexes, nn_names = self.shuffle_samples(nn_indexes, nn_names)

        return nn_indexes[:k]

    def retrieve(self, caption: str, *args, **kwargs) -> List[int]:
        """Retrieve motion indexes for a caption."""
        k = kwargs.get('k', self.top_k)
        return self.get_knn_samples(caption, k=k)

    def forward(self, captions: List[str], *args, **kwargs) -> dict:
        """Batch retrieval, returning re_dict compatible with MocoTmrRetriever.

        Args:
            captions: list of B caption strings
            k: optional override for top_k

        Returns:
            re_dict with:
                - re_motion: (B, K, 1, code_dim2d) z_e features of retrieved motions
                - re_text: (B, K, 1, 512) CLIP text features of retrieved captions
        """
        k = kwargs.get('k', self.top_k)
        b = len(captions)
        all_indexes = []

        for b_ix in range(b):
            batch_indexes = self.retrieve(captions[b_ix], k=k)
            all_indexes.append(batch_indexes)

        all_indexes = torch.tensor(all_indexes, dtype=torch.long)  # (B, K)
        b, k = all_indexes.shape

        flat_indexes = all_indexes.view(-1)  # (B*K)

        selected_motions = self.motion_features[flat_indexes]  # (B*K, code_dim2d)
        selected_texts = self.text_features[flat_indexes]      # (B*K, 512)

        all_motions_feature = selected_motions.view(b, k, 1, -1)  # (B, K, 1, code_dim2d)
        all_texts_feature = selected_texts.view(b, k, 1, -1)      # (B, K, 1, 512)

        re_dict = {
            're_motion': all_motions_feature,
            're_text': all_texts_feature,
        }
        return re_dict


if __name__ == '__main__':
    print("Testing ZeRetriever ...")
    retriever = ZeRetriever(
        database_path='database_ze',
        query_projector_path=None,
        top_k=2,
        num_retrieval=10,
        use_shuffle=False,
    )

    captions = [
        'a person walks forward and then turns around',
        'a person jumps up and down',
    ]
    re_dict = retriever(captions, k=2)
    for k_name, v in re_dict.items():
        print(f"  {k_name}: {v.shape}")

    print("Done.")
