import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import clip
import math
from einops import rearrange
from models.transformer.tools import cosine_schedule, uniform, get_mask_subset_prob, cal_performance, eval_decorator, top_k, gumbel_sample
from models.transformer.semantics_modulated import SemanticsModulatedAttention

class InputProcess(nn.Module):

    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = self.poseEmbedding(x)
        return x

class PositionalEncoding2D(nn.Module):

    def __init__(self, d_model, dropout=0.1, height=200, width=50):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        if d_model % 4 != 0:
            raise ValueError('Cannot use sin/cos positional encoding with odd dimension (got dim={:d})'.format(d_model))
        pe = torch.zeros(d_model, height, width)
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0.0, width).unsqueeze(1)
        pos_h = torch.arange(0.0, height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe = pe.permute(1, 2, 0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :x.shape[1], None, :]
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

class MaskTransformer2D(nn.Module):

    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1, clip_version=None, opt=None, retrieval_dim=None):
        super(MaskTransformer2D, self).__init__()
        self.retr_cfgmix_w = None
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')
        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt
        self.retrieval_dim = retrieval_dim
        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position2d_enc = PositionalEncoding2D(self.latent_dim, self.dropout)
        if opt.attnj:
            self.attnj = True
            seqTransEncoderLayer2 = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=num_heads, dim_feedforward=ff_size, dropout=dropout, activation='gelu')
            self.seqTransEncoder2 = nn.TransformerEncoder(seqTransEncoderLayer2, num_layers=num_layers)
        else:
            self.attnj = False
        if 'attnt' in opt and opt.attnt:
            self.attnt = True
            seqTransEncoderLayer3 = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=num_heads, dim_feedforward=ff_size, dropout=dropout, activation='gelu')
            self.seqTransEncoder3 = nn.TransformerEncoder(seqTransEncoderLayer3, num_layers=num_layers)
        else:
            self.attnt = False
        self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        _num_tokens = opt.num_tokens2d + 2
        self.mask_id = opt.num_tokens2d
        self.pad_id = opt.num_tokens2d + 1
        self.output_process = OutputProcess_Bert(out_feats=opt.num_tokens2d, latent_dim=self.latent_dim)
        self.token_emb = nn.Embedding(_num_tokens, self.code_dim)
        self.apply(self.__init_weights)
        print('Loading CLIP...')
        self.clip_version = clip_version
        self.clip_model = self.load_and_freeze_clip(clip_version)
        self.noise_schedule = cosine_schedule
        cfg = {'latent_dim': latent_dim, 'text_latent_dim': clip_dim, 'num_heads': num_heads, 'dropout': dropout, 'retrieval_dim': retrieval_dim}
        self.semanticTransEncoder = nn.ModuleList()
        for i in range(num_layers):
            self.semanticTransEncoder.append(SemanticsModulatedAttention(**cfg))

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

    def load_and_freeze_clip(self, clip_version):
        (clip_model, clip_preprocess) = clip.load(clip_version, device='cpu', jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def mask_text_and_retr_cond(self, motion_ids, force_mask=False):
        (bs, t, j) = motion_ids.shape
        x = motion_ids
        cond_type = torch.zeros(bs, 1, 1).to(x.device) + 99
        if force_mask:
            cond_type = torch.zeros(bs, 1, 1).to(x.device) + 0
        elif self.training:
            cond_type = torch.randint(0, 100, size=(bs, 1, 1)).to(x.device)
        return cond_type

    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=False, re_dict=None):
        cond_type = self.mask_text_and_retr_cond(motion_ids, force_mask)
        x = self.token_emb(motion_ids)
        (bs, T, J, dim) = x.shape
        x = self.input_process(x.reshape(bs, T * J, dim).permute(1, 0, 2))
        cond = cond.unsqueeze(0)
        x = x.reshape(T, J, bs, x.shape[-1])
        x = self.position2d_enc(x)
        xseq = x.reshape(T * J, bs, x.shape[-1])
        padding_mask = rearrange(padding_mask, 'b t j -> b (t j)')
        hidden_state = xseq.permute(1, 0, 2)
        xf = cond.permute(1, 0, 2)
        src_mask = (~padding_mask).unsqueeze(-1)
        for module in self.semanticTransEncoder:
            res = hidden_state
            hidden_state = module(x=hidden_state, xf=xf, src_mask=src_mask, cond_type=cond_type, re_dict=re_dict)
            hidden_state = res + hidden_state
        output = hidden_state.permute(1, 0, 2)
        logits = self.output_process(output)
        return logits

    def forward(self, ids_j, y, m_lens, re_dict=None):
        device = ids_j.device
        (bs, xtokens, ytokens) = ids_j.shape
        non_pad_mask = torch.arange(xtokens, device=device).expand(bs, xtokens) < m_lens.unsqueeze(1)
        non_pad_mask = non_pad_mask[..., None].repeat(1, 1, ytokens)
        ids_j = torch.where(non_pad_mask, ids_j, self.pad_id)
        force_mask = False
        cond_vector = y.to(device).float()
        ntokens = xtokens * ytokens
        rand_time = uniform((bs,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (xtokens * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((bs, xtokens), device=device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        mask = mask & non_pad_mask[..., 0]
        labels = torch.where(mask[..., None].repeat(1, 1, ytokens), ids_j, self.mask_id)
        x_ids_j = ids_j.clone()
        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_id = torch.randint_like(x_ids_j, high=self.opt.num_tokens2d)
        x_ids_j = torch.where(mask_rid[..., None].repeat(1, 1, ytokens), rand_id, x_ids_j)
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        x_ids_j = torch.where(mask_mid[..., None].repeat(1, 1, ytokens), self.mask_id, x_ids_j)
        mask_time = mask
        mask_time = mask_time[..., None].repeat(1, 1, ytokens)
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        mask = mask & non_pad_mask.reshape(bs, -1)
        mask = mask & ~mask_time.reshape(bs, -1)
        labels = torch.where(mask, x_ids_j.reshape(bs, -1), labels.reshape(bs, -1))
        x_ids_j = x_ids_j.reshape(bs, -1)
        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_id = torch.randint_like(x_ids_j, high=self.opt.num_tokens2d)
        x_ids_j = torch.where(mask_rid, rand_id, x_ids_j)
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        x_ids_j = torch.where(mask_mid, self.mask_id, x_ids_j)
        logits = self.trans_forward(x_ids_j.reshape(bs, xtokens, ytokens), cond_vector, ~non_pad_mask, force_mask, re_dict=re_dict)
        (ce_loss, pred_id, acc) = cal_performance(logits, labels, ignore_index=self.mask_id)
        return (ce_loss, pred_id, acc)

    def forward_with_cond_scale(self, motion_ids, cond_vector, padding_mask, cond_scale=3, force_mask=False, re_dict=None):
        if force_mask:
            return self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True, re_dict=re_dict)
        logits = self.trans_forward(motion_ids, cond_vector, padding_mask, re_dict=re_dict)
        if cond_scale == 1:
            return logits
        aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True, re_dict=re_dict)
        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self, conds, m_lens, timesteps: int, cond_scale: int, n_j=1, temperature=1, topk_filter_thres=0.9, gsample=False, force_mask=False, re_dict=None):
        device = next(self.parameters()).device
        seq_len = max(m_lens)
        batch_size = len(m_lens)
        cond_vector = conds.to(device).float()
        non_pad_mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) < m_lens.unsqueeze(1)
        padding_mask = ~non_pad_mask[..., None].repeat(1, 1, n_j)
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        scores = torch.where(padding_mask, 100000.0, 0.0).reshape(batch_size, seq_len * n_j)
        cfgmix_w = self.retr_cfgmix_w
        re_dict_zero = None
        if cfgmix_w is not None and re_dict is not None:
            re_dict_zero = {k: torch.zeros_like(v) for (k, v) in re_dict.items()}
        for timestep in torch.linspace(0, 1, timesteps, device=device):
            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = torch.round(rand_mask_prob * m_lens * n_j).clamp(min=1)
            sorted_indices = scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = ranks < num_token_masked.unsqueeze(-1)
            ids = torch.where(is_mask, self.mask_id, ids.reshape(batch_size, -1)).reshape(batch_size, seq_len, n_j)
            if re_dict_zero is None:
                logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector, padding_mask=padding_mask, cond_scale=cond_scale, force_mask=force_mask, re_dict=re_dict)
            else:
                logits_real = self.forward_with_cond_scale(ids, cond_vector=cond_vector, padding_mask=padding_mask, cond_scale=cond_scale, force_mask=force_mask, re_dict=re_dict)
                logits_zero = self.forward_with_cond_scale(ids, cond_vector=cond_vector, padding_mask=padding_mask, cond_scale=cond_scale, force_mask=force_mask, re_dict=re_dict_zero)
                logits = logits_zero + (logits_real - logits_zero) * cfgmix_w
            logits = logits.permute(0, 2, 1)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
            if gsample:
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            else:
                if temperature == 1:
                    probs = F.softmax(filtered_logits, dim=-1)
                    pred_ids = Categorical(probs / temperature).sample()
                else:
                    probs = F.softmax(filtered_logits / temperature, dim=-1)
                    pred_ids = Categorical(probs).sample()
            ids = torch.where(is_mask, pred_ids, ids.reshape(batch_size, -1)).reshape(batch_size, seq_len, n_j)
            probs_without_temperature = logits.softmax(dim=-1)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))
            scores = scores.squeeze(-1)
            scores = scores.masked_fill(~is_mask, 100000.0)
            scores = scores.reshape(batch_size, seq_len * n_j)
        ids = torch.where(padding_mask, -1, ids)
        return ids
