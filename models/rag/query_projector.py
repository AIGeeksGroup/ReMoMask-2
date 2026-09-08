import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryProjector(nn.Module):

    def __init__(self, clip_dim: int=512, ze_dim: int=1024, hidden: int=1024):
        super().__init__()
        self.clip_dim = clip_dim
        self.ze_dim = ze_dim
        self.hidden = hidden
        self.mlp = nn.Sequential(nn.Linear(clip_dim, hidden), nn.GELU(), nn.Linear(hidden, ze_dim))

    def forward(self, clip_text_embedding: torch.Tensor) -> torch.Tensor:
        projected = self.mlp(clip_text_embedding)
        return F.normalize(projected, dim=-1)

def save_projector(model: QueryProjector, path: str, extra: dict | None=None):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload = {'model_state_dict': model.state_dict(), 'clip_dim': model.clip_dim, 'ze_dim': model.ze_dim, 'hidden': model.hidden}
    if extra:
        payload.update(extra)
    torch.save(payload, path)

def load_projector(path: str, map_location: str='cpu') -> QueryProjector:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    clip_dim = ckpt['clip_dim']
    ze_dim = ckpt['ze_dim']
    hidden = ckpt['hidden']
    model = QueryProjector(clip_dim=clip_dim, ze_dim=ze_dim, hidden=hidden)
    model.load_state_dict(ckpt['model_state_dict'])
    return model
