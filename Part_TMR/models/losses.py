import torch
import torch.nn.functional as F

def compute_infonce_loss(query, key_positive, key_negatives, temperature=0.07):
    (B, D) = query.shape
    K = key_negatives.shape[1]
    query = F.normalize(query, dim=-1)
    key_positive = F.normalize(key_positive, dim=-1)
    key_negatives = F.normalize(key_negatives, dim=0)
    positive_sim = torch.sum(query * key_positive, dim=1) / temperature
    negative_sims = query @ key_negatives / temperature
    logits = torch.cat([positive_sim.unsqueeze(1), negative_sims], dim=1)
    labels = torch.zeros(B, dtype=torch.long, device=query.device)
    loss = F.cross_entropy(logits, labels)
    return loss
