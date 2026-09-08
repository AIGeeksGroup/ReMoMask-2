import torch
import torch.nn as nn
from .losses import compute_infonce_loss

class HBMLoss(nn.Module):

    def __init__(self, temperature=0.07, lambda_part=1.0, num_parts=6):
        super().__init__()
        self.temperature = temperature
        self.lambda_part = lambda_part
        self.num_parts = num_parts

    def forward(self, text_feat, motion_feat_global, motion_feat_parts, text_feat_m, motion_feat_global_m, motion_feat_parts_m, text_queue, motion_queue, part_queues):
        B = text_feat.shape[0]
        loss_inst_t2m = compute_infonce_loss(query=text_feat, key_positive=motion_feat_global_m, key_negatives=motion_queue, temperature=self.temperature)
        loss_inst_m2t = compute_infonce_loss(query=motion_feat_global, key_positive=text_feat_m, key_negatives=text_queue, temperature=self.temperature)
        loss_instance = loss_inst_t2m + loss_inst_m2t
        losses_t2p = []
        for k in range(self.num_parts):
            part_feat_m_k = motion_feat_parts_m[:, k, :]
            part_queue_k = part_queues[k]
            loss_k = compute_infonce_loss(query=text_feat, key_positive=part_feat_m_k, key_negatives=part_queue_k, temperature=self.temperature)
            losses_t2p.append(loss_k)
        loss_part_t2p = torch.stack(losses_t2p).mean()
        losses_p2t = []
        for k in range(self.num_parts):
            part_feat_k = motion_feat_parts[:, k, :]
            loss_k = compute_infonce_loss(query=part_feat_k, key_positive=text_feat_m, key_negatives=text_queue, temperature=self.temperature)
            losses_p2t.append(loss_k)
        loss_part_p2t = torch.stack(losses_p2t).mean()
        loss_part = loss_part_t2p + loss_part_p2t
        loss_total = loss_instance + self.lambda_part * loss_part
        return {'total': loss_total, 'instance': loss_instance, 'part': loss_part, 'inst_t2m': loss_inst_t2m, 'inst_m2t': loss_inst_m2t, 'part_t2p': loss_part_t2p, 'part_p2t': loss_part_p2t}
