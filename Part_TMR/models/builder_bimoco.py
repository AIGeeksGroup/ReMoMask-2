import torch
import torch.nn.functional as F
import torch.nn as nn
from Part_TMR.models.encdoc import ProjectionHead, TextEncoder, MotionEncoder

class MoCoTMR(nn.Module):

    def __init__(self, text_encoder_alias='ViT-B-32.pt', text_encoder_trainable: bool=False, motion_embedding_dims: int=512, text_embedding_dims: int=512, projection_dims: int=512, dropout: float=0.5, mode: str='t2m', temp=0.07, alpha=0.9996, config=None) -> None:
        super().__init__()
        embed_dim = config.model.embed_dim
        self.queue_size = config.model.queue_size
        self.momentum = config.model.momentum
        self.temp = temp
        self.use_hbm_loss = config.model.use_hbm_loss
        self.lambda_part = config.model.lambda_part
        motion_encoder = MotionEncoder(image_embedding_dim=motion_embedding_dims, num_layers=4, num_heads=4, mode=mode)
        text_encoder = TextEncoder(model_name=text_encoder_alias, trainable=text_encoder_trainable)
        self.motion_encoder = motion_encoder
        self.text_encoder = text_encoder
        self.motion_projection = ProjectionHead(embedding_dim=motion_embedding_dims * len(motion_encoder.parts_name), projection_dim=projection_dims, dropout=dropout)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_dims, projection_dim=projection_dims, dropout=dropout)
        self.motion_encoder_m = MotionEncoder(image_embedding_dim=motion_embedding_dims, num_layers=4, num_heads=4, mode=mode)
        self.motion_projection_m = ProjectionHead(embedding_dim=motion_embedding_dims * len(motion_encoder.parts_name), projection_dim=projection_dims, dropout=dropout)
        self.text_encoder_m = TextEncoder(model_name=text_encoder_alias, trainable=text_encoder_trainable)
        self.text_projection_m = ProjectionHead(embedding_dim=text_embedding_dims, projection_dim=projection_dims, dropout=dropout)
        self.model_pairs = [[self.motion_encoder, self.motion_encoder_m], [self.motion_projection, self.motion_projection_m], [self.text_encoder, self.text_encoder_m], [self.text_projection, self.text_projection_m]]
        self.copy_params()
        self.register_buffer('motion_queue', torch.randn(embed_dim, self.queue_size))
        self.register_buffer('text_queue', torch.randn(embed_dim, self.queue_size))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.motion_queue = nn.functional.normalize(self.motion_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        num_parts = 6
        part_queue_size = config.model.part_queue_size
        self.register_buffer('part_queues', torch.randn(num_parts, embed_dim, part_queue_size))
        self.part_queues = nn.functional.normalize(self.part_queues, dim=1)
        if self.use_hbm_loss:
            from .hbm_loss import HBMLoss
            self.hbm_loss = HBMLoss(temperature=temp, lambda_part=self.lambda_part, num_parts=num_parts)

    @property
    def device(self):
        return self.text_encoder.device

    def encode_motion(self, motion):
        motion_output = self.motion_encoder(motion)
        motion_global = self.motion_projection(motion_output['concat'])
        motion_output['global'] = motion_global
        if self.use_hbm_loss:
            return motion_output
        else:
            motion_embeddings = self.motion_projection(motion_output['concat'])
        return motion_embeddings

    def tokenize(self, text):
        text_ids = self.text_encoder.tokenize(text)
        return text_ids

    def encode_text(self, text):
        text_features = self.text_encoder(text)
        text_embeddings = self.text_projection(text_features)
        return text_embeddings

    def forward(self, motions, texts: dict, captions: list[str], return_loss=True):
        B = texts.shape[0]
        text_features = self.text_encoder(texts)
        text_feat = self.text_projection(text_features)
        motion_output = self.motion_encoder(motions)
        if self.use_hbm_loss:
            motion_feat_global = self.motion_projection(motion_output['concat'])
            (B_check, K, D) = motion_output['parts'].shape
            parts_proj = motion_output['parts'].reshape(B_check * K, D)
            motion_feat_parts = parts_proj.reshape(B_check, K, D)
        else:
            motion_feat = self.motion_projection(motion_output['concat'])
        if not return_loss:
            if self.use_hbm_loss:
                return (motion_feat_global, text_feat, motion_feat_parts)
            else:
                return (motion_feat, text_feat)
        with torch.no_grad():
            self._momentum_update()
            text_features = self.text_encoder_m(texts)
            text_feat_m = self.text_projection_m(text_features)
            motion_output_m = self.motion_encoder_m(motions)
        if self.use_hbm_loss:
            with torch.no_grad():
                motion_feat_global_m = self.motion_projection_m(motion_output_m['concat'])
                parts_proj_m = motion_output_m['parts'].reshape(B * K, D)
                motion_feat_parts_m = parts_proj_m.reshape(B, K, D)
            loss_dict = self.hbm_loss(text_feat=text_feat, motion_feat_global=motion_feat_global, motion_feat_parts=motion_feat_parts, text_feat_m=text_feat_m, motion_feat_global_m=motion_feat_global_m, motion_feat_parts_m=motion_feat_parts_m, text_queue=self.text_queue.clone().detach(), motion_queue=self.motion_queue.clone().detach(), part_queues=self.part_queues.clone().detach())
            loss = loss_dict['total']
            if self.training:
                self._dequeue_and_enqueue_hbm(motion_feat_global_m, text_feat_m, motion_feat_parts_m)
            return (loss, loss_dict)
        else:
            with torch.no_grad():
                motion_feat_m = self.motion_projection_m(motion_output_m['concat'])
                motion_feat_all = torch.cat([motion_feat_m.T, self.motion_queue.clone().detach()], dim=1)
                text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)
                K_queue = self.motion_queue.shape[1]
                sim_hard_targets = torch.zeros(B, B + K_queue).to(motion_output_m['concat'].device)
                sim_hard_targets.fill_diagonal_(1)
                sim_targets = sim_hard_targets
            sim_m2t = motion_feat @ text_feat_all / self.temp
            sim_t2m = text_feat @ motion_feat_all / self.temp
            loss_m2t = -torch.sum(F.log_softmax(sim_m2t, dim=1) * sim_targets, dim=1).mean()
            loss_t2m = -torch.sum(F.log_softmax(sim_t2m, dim=1) * sim_targets, dim=1).mean()
            loss = (loss_m2t + loss_t2m) / 2
            if self.training:
                self._dequeue_and_enqueue(motion_feat_m, text_feat_m)
            return loss

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for (param, param_m) in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for (param, param_m) in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, motion_feat, text_feat):
        image_feats = concat_all_gather(motion_feat)
        text_feats = concat_all_gather(text_feat)
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0
        self.motion_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_hbm(self, motion_feat_global, text_feat, motion_feat_parts):
        motion_feats = concat_all_gather(motion_feat_global)
        text_feats = concat_all_gather(text_feat)
        (B, K, D) = motion_feat_parts.shape
        parts_feats = concat_all_gather(motion_feat_parts.reshape(B, K * D))
        parts_feats = parts_feats.reshape(-1, K, D)
        batch_size = motion_feats.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0
        self.motion_queue[:, ptr:ptr + batch_size] = motion_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        part_queue_size = self.part_queues.shape[2]
        assert part_queue_size % batch_size == 0
        ptr_part = ptr % part_queue_size
        for k in range(K):
            self.part_queues[k, :, ptr_part:ptr_part + batch_size] = parts_feats[:, k, :].T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

@torch.no_grad()
def concat_all_gather(tensor):
    if not torch.distributed.is_initialized():
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
