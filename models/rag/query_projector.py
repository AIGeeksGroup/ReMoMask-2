"""
QueryProjector: maps CLIP 512-d text embeddings into the VQ-VAE z_e space.

Architecture (per D-04):
    Linear(clip_dim -> ze_dim) + GELU + Linear(ze_dim -> ze_dim)
    Output is L2-normalised.

Usage:
    from models.rag.query_projector import QueryProjector, save_projector, load_projector

    proj = QueryProjector(clip_dim=512, ze_dim=1024)
    out = proj(clip_text_emb)  # (B, 1024), L2-normalised
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryProjector(nn.Module):
    """2-layer MLP projecting CLIP text embeddings into the z_e latent space."""

    def __init__(self, clip_dim: int = 512, ze_dim: int = 1024):
        super().__init__()
        self.clip_dim = clip_dim
        self.ze_dim = ze_dim
        self.mlp = nn.Sequential(
            nn.Linear(clip_dim, ze_dim),
            nn.GELU(),
            nn.Linear(ze_dim, ze_dim),
        )

    def forward(self, clip_text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_text_embedding: (B, clip_dim) CLIP text features.
        Returns:
            (B, ze_dim) projected vector, L2-normalised.
        """
        projected = self.mlp(clip_text_embedding)
        return F.normalize(projected, dim=-1)


# ------------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------------

def save_projector(model: QueryProjector, path: str, extra: dict | None = None):
    """Save projector checkpoint with metadata."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "clip_dim": model.clip_dim,
        "ze_dim": model.ze_dim,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_projector(
    path: str,
    clip_dim: int | None = None,
    ze_dim: int | None = None,
    map_location: str = "cpu",
) -> QueryProjector:
    """Load projector from checkpoint.

    If *clip_dim* / *ze_dim* are ``None`` they are read from the checkpoint.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    clip_dim = clip_dim or ckpt["clip_dim"]
    ze_dim = ze_dim or ckpt["ze_dim"]
    model = QueryProjector(clip_dim=clip_dim, ze_dim=ze_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    return model
