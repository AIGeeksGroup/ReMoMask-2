import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from models.transformer.tools import cosine_schedule, uniform, get_mask_subset_prob, cal_performance, eval_decorator, top_k, gumbel_sample, lengths_to_mask

class InputProcess(nn.Module):

    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = x.permute((1, 0, 2))
        x = self.poseEmbedding(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = nn.Parameter(pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class OutputProcess_Bert(nn.Module):

    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)
        output = output.permute(1, 2, 0)
        return output

class MaskTransformer(nn.Module):

    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1, clip_version=None, opt=None):
        super(MaskTransformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')
        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt
        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=num_heads, dim_feedforward=ff_size, dropout=dropout, activation='gelu')
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=num_layers)
        self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        _num_tokens = opt.num_tokens1d + 2
        self.mask_id = opt.num_tokens1d
        self.pad_id = opt.num_tokens1d + 1
        self.output_process = OutputProcess_Bert(out_feats=opt.num_tokens1d, latent_dim=latent_dim)
        self.token_emb = nn.Embedding(_num_tokens, self.code_dim)
        self.apply(self.__init_weights)
        print('Loading CLIP...')
        self.clip_version = clip_version
        self.clip_model = None
        self.noise_schedule = cosine_schedule

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for (name, p) in self.named_parameters() if not name.startswith('clip_model.')]

    def mask_cond(self, cond, force_mask=False):
        (bs, d) = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.0:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1.0 - mask)
        else:
            return cond

    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=False):
        cond = self.mask_cond(cond, force_mask=force_mask)
        x = self.token_emb(motion_ids)
        x = self.input_process(x)
        cond = cond.unsqueeze(0)
        x = self.position_enc(x)
        xseq = torch.cat([cond, x], dim=0)
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[1:]
        logits = self.output_process(output)
        return logits

    def forward(self, ids, y, m_lens):
        (bs, ntokens) = ids.shape
        device = ids.device
        non_pad_mask = lengths_to_mask(m_lens, ntokens)
        ids = torch.where(non_pad_mask, ids, self.pad_id)
        force_mask = False
        cond_vector = y.to(device).float()
        rand_time = uniform((bs,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        mask &= non_pad_mask
        labels = torch.where(mask, ids, self.mask_id)
        x_ids = ids.clone()
        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_id = torch.randint_like(x_ids, high=self.mask_id)
        x_ids = torch.where(mask_rid, rand_id, x_ids)
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        x_ids = torch.where(mask_mid, self.mask_id, x_ids)
        logits = self.trans_forward(x_ids, cond_vector, ~non_pad_mask, force_mask)
        (ce_loss, pred_id, acc) = cal_performance(logits, labels, ignore_index=self.mask_id)
        return (ce_loss, pred_id, acc)

    def forward_with_cond_scale(self, motion_ids, cond_vector, padding_mask, cond_scale=3, force_mask=False):
        if force_mask:
            return self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)
        logits = self.trans_forward(motion_ids, cond_vector, padding_mask)
        if cond_scale == 1:
            return logits
        aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)
        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self, conds, m_lens, timesteps: int, cond_scale: int, temperature=1, topk_filter_thres=0.9, gsample=False, force_mask=False):
        device = next(self.parameters()).device
        seq_len = max(m_lens)
        batch_size = len(m_lens)
        cond_vector = conds.to(device).float()
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        scores = torch.where(padding_mask, 100000.0, 0.0)
        for timestep in torch.linspace(0, 1, timesteps, device=device):
            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)
            sorted_indices = scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = ranks < num_token_masked.unsqueeze(-1)
            ids = torch.where(is_mask, self.mask_id, ids)
            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector, padding_mask=padding_mask, cond_scale=cond_scale, force_mask=force_mask)
            logits = logits.permute(0, 2, 1)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
            if gsample:
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            else:
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                pred_ids = Categorical(probs).sample()
            ids = torch.where(is_mask, pred_ids, ids)
            probs_without_temperature = logits.softmax(dim=-1)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))
            scores = scores.squeeze(-1)
            scores = scores.masked_fill(~is_mask, 100000.0)
        ids = torch.where(padding_mask, -1, ids)
        return ids
