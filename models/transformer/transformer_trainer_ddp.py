import torch
import torch.optim as optim
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict, defaultdict
from os.path import join as pjoin
import numpy as np
import time
from utils.utils import print_current_loss
from utils.eval_t2m_ddp import evaluation_mask_transformer
from data.t2m_dataset import collate_fn
import tqdm
import clip

class MaskTransformerTrainer:

    def __init__(self, args, mask_transformer_aux, mask_transformer_ts, vq_model, retriever):
        self.opt = args
        self.mask_transformer_aux = mask_transformer_aux
        self.mask_transformer_ts = mask_transformer_ts
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()
        (self.rank, self.world_size) = (distributed.get_rank(), distributed.get_world_size())
        self.retriever = retriever
        self.retriever.eval()
        if args.is_train:
            self.logger = SummaryWriter(args.log_dir) if self.rank == 0 else None

    def tokenize(self, text: str | list[str]):
        text_ids = clip.tokenize(text, truncate=True)
        return text_ids

    def encode_text(self, text_ids):
        text_embedding = self.retriever.encode_text(text_ids)
        return text_embedding

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_mask_transformer_aux.param_groups:
            param_group['lr'] = current_lr
        for param_group in self.opt_mask_transformer_ts.param_groups:
            param_group['lr'] = current_lr
        return current_lr

    def forward(self, batch_data, re_dict=None, use_ddp=True):
        (conds, motion, joints, m_lens) = batch_data
        motion = motion.detach().float().to(self.device)
        joints = joints.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)
        (code_idx1d, _, code_idx2d, _) = self.vq_model.encode(motion, joints)
        m_lens_4 = m_lens // 4
        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds
        aux = self.mask_transformer_aux
        spatial = self.mask_transformer_ts
        if not use_ddp:
            if isinstance(aux, torch.nn.parallel.DistributedDataParallel):
                aux = aux.module
            if isinstance(spatial, torch.nn.parallel.DistributedDataParallel):
                spatial = spatial.module
        (_loss1, _pred_ids1, _acc1) = aux(code_idx1d[..., 0], conds, m_lens_4)
        (_loss2, _pred_ids2, _acc2) = spatial(code_idx2d[..., 0], conds, m_lens_4, re_dict)
        return (_loss1, _acc1, _loss2, _acc2)

    def update(self, batch_data, re_dict=None):
        (loss1, acc1, loss2, acc2) = self.forward(batch_data, re_dict)
        total_loss = loss1 + loss2
        self.opt_mask_transformer_aux.zero_grad()
        self.opt_mask_transformer_ts.zero_grad()
        total_loss.backward()
        self.opt_mask_transformer_aux.step()
        self.opt_mask_transformer_ts.step()
        self.scheduler_aux.step()
        self.scheduler_ts.step()
        return (loss1.item(), acc1, loss2.item(), acc2)

    def save(self, file_name, ep, total_it):
        mask_trans_aux_state_dict = self.mask_transformer_aux.state_dict() if not isinstance(self.mask_transformer_aux, torch.nn.parallel.DistributedDataParallel) else self.mask_transformer_aux.module.state_dict()
        clip_weights1 = [e for e in mask_trans_aux_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights1:
            del mask_trans_aux_state_dict[e]
        mask_trans_ts_state_dict = self.mask_transformer_ts.state_dict() if not isinstance(self.mask_transformer_ts, torch.nn.parallel.DistributedDataParallel) else self.mask_transformer_ts.module.state_dict()
        clip_weights2 = [e for e in mask_trans_ts_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights2:
            del mask_trans_ts_state_dict[e]
        state = {'mask_transformer_aux': mask_trans_aux_state_dict, 'mask_transformer_ts': mask_trans_ts_state_dict, 'opt_mask_transformer_aux': self.opt_mask_transformer_aux.state_dict(), 'scheduler_aux': self.scheduler_aux.state_dict(), 'opt_mask_transformer_ts': self.opt_mask_transformer_ts.state_dict(), 'scheduler_ts': self.scheduler_ts.state_dict(), 'ep': ep, 'total_it': total_it}
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        for (key, model) in [('mask_transformer_aux', self.mask_transformer_aux), ('mask_transformer_ts', self.mask_transformer_ts)]:
            module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            (missing, unexpected) = module.load_state_dict(checkpoint[key], strict=False)
            assert not unexpected, unexpected
            assert all((k.startswith('clip_model.') for k in missing)), missing
        self.opt_mask_transformer_aux.load_state_dict(checkpoint['opt_mask_transformer_aux'])
        self.scheduler_aux.load_state_dict(checkpoint['scheduler_aux'])
        self.opt_mask_transformer_ts.load_state_dict(checkpoint['opt_mask_transformer_ts'])
        self.scheduler_ts.load_state_dict(checkpoint['scheduler_ts'])
        return (checkpoint['ep'], checkpoint['total_it'])

    def train(self, train_dataset, val_dataset, eval_dataset, batch_size, eval_wrapper, plot_eval):
        device = next(self.mask_transformer_ts.parameters()).device
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, drop_last=True, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
        eval_sampler = DistributedSampler(eval_dataset, shuffle=True)
        eval_val_loader = DataLoader(eval_dataset, batch_size=32, num_workers=4, drop_last=True, sampler=eval_sampler, collate_fn=collate_fn)
        self.vq_model.to(self.device)
        self.opt_mask_transformer_aux = optim.AdamW([{'params': self.mask_transformer_aux.parameters()}], betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-05)
        self.scheduler_aux = optim.lr_scheduler.MultiStepLR(self.opt_mask_transformer_aux, milestones=self.opt.milestones, gamma=self.opt.gamma)
        self.opt_mask_transformer_ts = optim.AdamW([{'params': self.mask_transformer_ts.parameters()}], betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-05)
        self.scheduler_ts = optim.lr_scheduler.MultiStepLR(self.opt_mask_transformer_ts, milestones=self.opt.milestones, gamma=self.opt.gamma)
        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            (epoch, it) = self.resume(model_dir)
            print('Load model epoch:%d iterations:%d' % (epoch, it))
        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(float, OrderedDict())
        (best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer) = evaluation_mask_transformer(self.opt.save_root, eval_val_loader, self.mask_transformer_aux, self.mask_transformer_ts, self.vq_model, self.logger, epoch, best_fid=100, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper, plot_func=plot_eval, save_ckpt=False, save_anim=True, retriever=self.retriever)
        best_acc = 0.0
        while epoch < self.opt.max_epoch:
            self.mask_transformer_aux.train()
            self.mask_transformer_ts.train()
            self.vq_model.eval()
            train_sampler.set_epoch(epoch)
            pbar = tqdm.tqdm(train_loader, leave=False)
            for (i, batch) in enumerate(pbar):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                (captions, motion, joints, m_lens) = batch
                re_dict = self.retriever(captions)
                caption_ids = self.tokenize(captions).to(device)
                caption_embedding = self.encode_text(caption_ids)
                batch = (caption_embedding, motion, joints, m_lens)
                (_, _, loss, acc) = self.update(batch_data=batch, re_dict=re_dict)
                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr'] += self.opt_mask_transformer_ts.param_groups[0]['lr']
                pbar.set_description(f'batchLoss: {loss:.4f} | Epoch: {epoch} | Step: {i}')
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for (tag, value) in logs.items():
                        if self.rank == 0:
                            self.logger.add_scalar('Train/%s' % tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(float, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)
            if self.rank == 0:
                self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1
            (best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer) = evaluation_mask_transformer(self.opt.save_root, eval_val_loader, self.mask_transformer_aux, self.mask_transformer_ts, self.vq_model, self.logger, epoch, best_fid=best_fid, best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3, best_matching=best_matching, eval_wrapper=eval_wrapper, plot_func=plot_eval, save_ckpt=True, save_anim=epoch % self.opt.eval_every_e == 0, retriever=self.retriever)
            if self.rank == 0:
                print('==> Validation:')
                self.vq_model.eval()
                self.mask_transformer_aux.eval()
                self.mask_transformer_ts.eval()
                val_loss = []
                val_acc = []
                with torch.no_grad():
                    for (i, batch_data) in enumerate(tqdm.tqdm(val_loader)):
                        (captions, motion, joints, m_lens) = batch_data
                        re_dict = self.retriever(captions)
                        caption_ids = self.tokenize(captions).to(device)
                        caption_embedding = self.encode_text(caption_ids)
                        batch = (caption_embedding, motion, joints, m_lens)
                        (_, _, loss, acc) = self.forward(batch, re_dict, use_ddp=False)
                        val_loss.append(loss.item())
                        val_acc.append(acc)
                print(f'Validation loss:{np.mean(val_loss):.3f}, accuracy:{np.mean(val_acc):.3f}')
                self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
                self.logger.add_scalar('Val/acc', np.mean(val_acc), epoch)
                if np.mean(val_acc) > best_acc:
                    print(f'Improved accuracy from {best_acc:.02f} to {np.mean(val_acc)}!!!')
                    self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
                    best_acc = np.mean(val_acc)
