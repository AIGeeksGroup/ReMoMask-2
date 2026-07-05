"""
QueryProjector: maps CLIP 512-d text embeddings into the VQ-VAE z_e space.

Architecture (per D-04), capacity configurable via `hidden` (G7):
    hidden == 0:    Linear(clip_dim -> ze_dim)                       [Linear variant]
    hidden != 0:    Linear(clip_dim -> hidden) + GELU + Linear(hidden -> ze_dim)
    Default hidden=1024 reproduces the original 2-layer MLP exactly.
    Output is always L2-normalised.

Usage:
    from models.rag.query_projector import QueryProjector, save_projector, load_projector

    proj = QueryProjector(clip_dim=512, ze_dim=1024, hidden=1024)
    out = proj(clip_text_emb)  # (B, 1024), L2-normalised
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryProjector(nn.Module):
    """MLP (or single Linear) projecting CLIP text embeddings into the z_e latent space.

    `hidden` controls capacity (G7):
        0    -> single Linear(clip_dim -> ze_dim), no nonlinearity.
        1024 -> current 2-layer MLP (default, backward compatible).
        2048 -> wider 2-layer MLP.
    """

    def __init__(self, clip_dim: int = 512, ze_dim: int = 1024, hidden: int = 1024):
        super().__init__()
        self.clip_dim = clip_dim
        self.ze_dim = ze_dim
        self.hidden = hidden
        if hidden == 0:
            self.mlp = nn.Sequential(
                nn.Linear(clip_dim, ze_dim),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(clip_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, ze_dim),
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
        "hidden": model.hidden,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_projector(
    path: str,
    clip_dim: int | None = None,
    ze_dim: int | None = None,
    hidden: int | None = None,
    map_location: str = "cpu",
) -> QueryProjector:
    """Load projector from checkpoint.

    If *clip_dim* / *ze_dim* / *hidden* are ``None`` they are read from the
    checkpoint. Checkpoints saved before G7 do not contain a "hidden" field;
    those were all trained with the original 2-layer MLP, equivalent to
    hidden=1024, so that is the default when the field is absent.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    clip_dim = clip_dim or ckpt["clip_dim"]
    ze_dim = ze_dim or ckpt["ze_dim"]
    hidden = hidden if hidden is not None else ckpt.get("hidden", 1024)
    model = QueryProjector(clip_dim=clip_dim, ze_dim=ze_dim, hidden=hidden)
    model.load_state_dict(ckpt["model_state_dict"])
    return model
