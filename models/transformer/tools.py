import torch
import torch.nn.functional as F
import math

def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

def eval_decorator(fn):

    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask

def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature=1.0, dim=1):
    return (t / max(temperature, 1e-10) + gumbel_noise(t)).argmax(dim=dim)

def top_k(logits, thres=0.9, dim=1):
    k = math.ceil((1 - thres) * logits.shape[dim])
    (val, ind) = logits.topk(k, dim=dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    return probs

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def cal_performance(pred, labels, ignore_index=None):
    loss = F.cross_entropy(pred, labels, ignore_index=ignore_index)
    pred_id_k = torch.topk(pred, k=1, dim=1).indices
    pred_id = pred_id_k[:, 0]
    mask = labels.ne(ignore_index)
    n_correct = (pred_id_k == labels.unsqueeze(1)).any(dim=1).masked_select(mask)
    acc = torch.mean(n_correct.float()).item()
    return (loss, pred_id, acc)
