import torch
from torch import nn
from .positional_encoding import PositionalEncoding
import clip

class ProjectionHead(nn.Module):

    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        return self.layer_norm(x)

class TextEncoder(nn.Module):

    def __init__(self, model_name: str='ViT-B-32.pt', trainable: bool=False) -> None:
        super().__init__()
        (self.clip_model, _) = clip.load(model_name, device='cpu', jit=False)
        for p in self.clip_model.parameters():
            p.requires_grad = trainable
        if not trainable:
            self.clip_model.eval()

    def forward(self, text_ids):
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_ids).float()
        return text_features

    def tokenize(self, text):
        text_ids = clip.tokenize(text, truncate=True)
        return text_ids

    @property
    def device(self):
        return next(self.clip_model.parameters()).device

class MotionEncoder(nn.Module):

    def __init__(self, image_embedding_dim, mode: str='t2m', ff_size: int=1024, num_layers: int=4, num_heads: int=4, dropout: float=0.1, activation: str='gelu') -> None:
        super().__init__()
        self.parts_name = ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']
        if mode == 't2m':
            parts_input_dim = {'Root': 7, 'R_Leg': 50, 'L_Leg': 50, 'Backbone': 60, 'R_Arm': 60, 'L_Arm': 60}
        else:
            parts_input_dim = {'Root': 7, 'R_Leg': 62, 'L_Leg': 62, 'Backbone': 48, 'R_Arm': 48, 'L_Arm': 48}
        for name in self.parts_name:
            raw_dim = parts_input_dim[name]
            latent_dim = image_embedding_dim
            skel_embedding = nn.Linear(raw_dim, image_embedding_dim)
            emb_token = nn.Parameter(torch.randn(latent_dim))
            sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)
            seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size, dropout=dropout, activation=activation)
            seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer, num_layers=num_layers)
            setattr(self, f'skel_embedding_{name}', skel_embedding)
            setattr(self, f'emb_token_{name}', emb_token)
            setattr(self, f'sequence_pos_encoding_{name}', sequence_pos_encoding)
            setattr(self, f'seqTransEncoder_{name}', seqTransEncoder)
        self.target_token_idx = 0

    def extarct_feature(self, x, name):
        (bs, nframes, nfeats) = x.shape
        skel_embedding = getattr(self, f'skel_embedding_{name}')
        x = skel_embedding(x)
        x = x.permute(1, 0, 2)
        emb_token = getattr(self, f'emb_token_{name}')
        emb_token = torch.tile(emb_token, (bs,)).reshape(bs, -1)
        xseq = torch.cat((emb_token[None], x), 0)
        sequence_pos_encoding = getattr(self, f'sequence_pos_encoding_{name}')
        seqTransEncoder = getattr(self, f'seqTransEncoder_{name}')
        xseq = sequence_pos_encoding(xseq)
        final = seqTransEncoder(xseq)
        return final[self.target_token_idx]

    def forward(self, parts):
        assert isinstance(parts, list)
        assert len(parts) == len(self.parts_name)
        embedding_parts = []
        for (i, name) in enumerate(self.parts_name):
            embedding_parts.append(self.extarct_feature(parts[i], name))
        parts_features = torch.stack(embedding_parts, dim=1)
        concat_features = torch.cat(embedding_parts, dim=1)
        return {'parts': parts_features, 'concat': concat_features}
