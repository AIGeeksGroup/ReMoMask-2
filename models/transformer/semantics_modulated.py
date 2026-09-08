import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticsModulatedAttention(nn.Module):

    def __init__(self, latent_dim, text_latent_dim, num_heads, dropout, rt_in_value=False, retrieval_dim=None):
        super().__init__()
        self.rt_in_value = rt_in_value
        self.vgate = False
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.retrieval_dim = retrieval_dim if retrieval_dim is not None else latent_dim
        if self.retrieval_dim != latent_dim:
            self.re_motion_proj = nn.Linear(self.retrieval_dim, latent_dim)
            self.re_text_proj = nn.Linear(self.retrieval_dim, latent_dim)
        else:
            self.re_motion_proj = nn.Identity()
            self.re_text_proj = nn.Identity()
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.info_mlp = nn.Sequential(nn.Linear(3 * latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, xf, src_mask, cond_type, re_dict=None):
        (B, N, D) = x.shape
        H = self.num_heads
        re_motion_raw = re_dict['re_motion'].squeeze(2)
        re_text_raw = re_dict['re_text'].squeeze(2)
        re_motion = self.re_motion_proj(re_motion_raw)
        re_text = self.re_text_proj(re_text_raw)
        text_cond = (cond_type % 10 > 0).float()
        retr_cond = (cond_type // 10 > 0).float()
        z_norm = self.norm(x)
        Q = self.query(z_norm) * src_mask
        R_m_pooled = re_motion.mean(dim=1, keepdim=True)
        R_t_pooled = re_text.mean(dim=1, keepdim=True)
        fused_input = torch.cat([xf * text_cond, R_m_pooled * retr_cond, R_t_pooled * retr_cond], dim=-1)
        info = self.info_mlp(fused_input)
        kv_input = torch.cat([z_norm, info], dim=1)
        K = self.key(kv_input)
        R_v = R_m_pooled + R_t_pooled if self.rt_in_value else R_m_pooled
        if self.vgate:
            R_v = R_v * retr_cond
        V_input = torch.cat([z_norm, R_v], dim=1)
        V = self.value(V_input)
        Q = Q.view(B, N, H, self.head_dim).transpose(1, 2)
        K = K.view(B, N + 1, H, self.head_dim).transpose(1, 2)
        V = V.view(B, N + 1, H, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, N, D)
        output = self.out_proj(attn_output)
        return output
