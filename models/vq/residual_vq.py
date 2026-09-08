import random
from random import randrange
import torch
from torch import nn
import torch.nn.functional as F
from models.vq.quantizer import QuantizeEMAReset
from einops import repeat

class ResidualVQ(nn.Module):

    def __init__(self, num_quantizers, shared_codebook=False, quantize_dropout_prob=0.5, quantize_dropout_cutoff_index=0, **kwargs):
        super().__init__()
        self.num_quantizers = num_quantizers
        if shared_codebook:
            layer = QuantizeEMAReset(**kwargs)
            self.layers = nn.ModuleList([layer for _ in range(num_quantizers)])
        else:
            self.layers = nn.ModuleList([QuantizeEMAReset(**kwargs) for _ in range(num_quantizers)])
        assert quantize_dropout_cutoff_index >= 0 and quantize_dropout_prob >= 0
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_prob = quantize_dropout_prob

    @property
    def codebooks(self):
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        return codebooks

    def get_codes_from_indices(self, indices):
        (batch, quantize_dim) = (indices.shape[0], indices.shape[-1])
        if quantize_dim < self.num_quantizers:
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)
        codebooks = repeat(self.codebooks, 'q c d -> q b c d', b=batch)
        gather_indices = repeat(indices, 'b n q -> q b n d', d=codebooks.shape[-1])
        mask = gather_indices == -1.0
        gather_indices = gather_indices.masked_fill(mask, 0)
        all_codes = codebooks.gather(2, gather_indices)
        all_codes = all_codes.masked_fill(mask, 0.0)
        return all_codes

    def forward(self, x, sample_codebook_temp=None):
        (num_quant, quant_dropout_prob, device) = (self.num_quantizers, self.quantize_dropout_prob, x.device)
        quantized_out = 0.0
        residual = x
        all_losses = []
        all_indices = []
        all_perplexity = []
        should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob
        start_drop_quantize_index = num_quant
        if should_quantize_dropout:
            start_drop_quantize_index = randrange(self.quantize_dropout_cutoff_index, num_quant)
            null_indices_shape = [x.shape[0], x.shape[-1]]
            null_indices = torch.full(null_indices_shape, -1.0, device=device, dtype=torch.long)
        for (quantizer_index, layer) in enumerate(self.layers):
            if should_quantize_dropout and quantizer_index > start_drop_quantize_index:
                all_indices.append(null_indices)
                continue
            (quantized, *rest) = layer(residual, return_idx=True, temperature=sample_codebook_temp)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            (embed_indices, loss, perplexity) = rest
            all_indices.append(embed_indices)
            all_losses.append(loss)
            all_perplexity.append(perplexity)
        all_indices = torch.stack(all_indices, dim=-1)
        all_losses = sum(all_losses) / len(all_losses)
        all_perplexity = sum(all_perplexity) / len(all_perplexity)
        ret = (quantized_out, all_indices, all_losses, all_perplexity)
        return ret

    def quantize(self, x):
        all_indices = []
        residual = x
        all_codes = []
        for (quantizer_index, layer) in enumerate(self.layers):
            (quantized, *rest) = layer(residual, return_idx=True)
            (embed_indices, loss, perplexity) = rest
            residual = residual - quantized.detach()
            all_indices.append(embed_indices)
            all_codes.append(quantized)
        code_idx = torch.stack(all_indices, dim=-1)
        all_codes = torch.stack(all_codes, dim=0)
        return (code_idx, all_codes)
