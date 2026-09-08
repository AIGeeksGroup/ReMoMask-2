import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from os.path import join as pjoin
import time
import numpy as np
from collections import OrderedDict, defaultdict
from utils.eval_t2m_ddp import evaluation_vqvae
from utils.utils import print_current_loss
from models.vq.quantizer import QuantizeEMAReset

class RVQTokenizerTrainer:

    def __init__(self, args, vq_model):
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device
        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            if args.recons_loss == 'l1':
                self.l1_criterion = torch.nn.L1Loss()
            elif args.recons_loss == 'l1_smooth':
                self.l1_criterion = torch.nn.SmoothL1Loss()

    def forward(self, batch_data):
        motions = batch_data[0].detach().to(self.device).float()
        motions2d = batch_data[1].detach().to(self.device).float()
        (pred_motion_aux, loss_commit_aux, perplexity_aux, pred_motion_ts, loss_commit, perplexity, pred_motion) = self.vq_model(motions, motions2d)
        self.motions = motions
        self.pred_motion_aux = pred_motion_aux
        self.pred_motion_ts = pred_motion_ts
        self.pred_motion = pred_motion
        loss_rec_aux = self.l1_criterion(pred_motion_aux, motions)
        pred_local_pos1d = pred_motion_aux[..., 4:(self.opt.joints_num - 1) * 3 + 4]
        local_pos = motions[..., 4:(self.opt.joints_num - 1) * 3 + 4]
        loss_explicit1d = self.l1_criterion(pred_local_pos1d, local_pos)
        loss1 = loss_rec_aux + self.opt.loss_vel * loss_explicit1d + self.opt.commit * loss_commit_aux
        (B, T0, J0, _) = motions2d.shape
        if J0 == 22:
            jmotion = torch.zeros([B, T0, 263], device=motions2d.device)
        elif J0 == 21:
            jmotion = torch.zeros([B, T0, 251], device=motions2d.device)
        jmotion[:, :, :4] = motions2d[:, :, 0, :4].reshape(B, T0, -1)
        jmotion[:, :, -4:] = motions2d[:, :, 0, 4:8].reshape(B, T0, -1)
        jmotion[:, :, 4:4 + (J0 - 1) * 3] = motions2d[:, :, 1:, :3].reshape(B, T0, -1)
        jmotion[:, :, 4 + (J0 - 1) * 3:4 + (J0 - 1) * 9] = motions2d[:, :, 1:, 3:9].reshape(B, T0, -1)
        jmotion[:, :, 4 + (J0 - 1) * 9:4 + (J0 - 1) * 9 + J0 * 3] = motions2d[:, :, :, 9:12].reshape(B, T0, -1)
        loss_rec_ts = self.l1_criterion(pred_motion_ts, jmotion)
        pred_local_pos_ts = pred_motion_ts[..., 4:(self.opt.joints_num - 1) * 3 + 4]
        loss_explicit_ts = self.l1_criterion(pred_local_pos_ts, local_pos)
        loss2 = loss_rec_ts + self.opt.loss_vel * loss_explicit_ts + self.opt.commit * loss_commit
        loss_rec = self.l1_criterion(pred_motion, motions)
        pred_local_pos = pred_motion[..., 4:(self.opt.joints_num - 1) * 3 + 4]
        loss_explicit = self.l1_criterion(pred_local_pos, local_pos)
        loss = loss_rec + self.opt.loss_vel * loss_explicit + self.opt.commit * (loss_commit_aux + loss_commit) / 2
        return (loss, loss1, loss2, loss_rec, loss_explicit, loss_commit, perplexity)

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group['lr'] = current_lr
        return current_lr

    def save(self, file_name, ep, total_it):
        quantizer_ema = {}
        for name, module in self.vq_model.named_modules():
            if isinstance(module, QuantizeEMAReset):
                quantizer_ema[name] = {
                    'init': module.init,
                    'code_sum': module.code_sum.detach().cpu() if module.code_sum is not None else None,
                    'code_count': module.code_count.detach().cpu() if module.code_count is not None else None,
                }
        state = {'vq_model': self.vq_model.state_dict(), 'opt_vq_model': self.opt_vq_model.state_dict(), 'scheduler': self.scheduler.state_dict(), 'quantizer_ema': quantizer_ema, 'ep': ep, 'total_it': total_it}
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        quantizer_ema = checkpoint['quantizer_ema']
        quantizers = {name: module for name, module in self.vq_model.named_modules() if isinstance(module, QuantizeEMAReset)}
        assert quantizers.keys() == quantizer_ema.keys(), 'Quantizer training state differs from model'
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        for name, module in quantizers.items():
            state = quantizer_ema[name]
            module.init = state['init']
            module.code_sum = state['code_sum'].to(module.codebook.device) if state['code_sum'] is not None else None
            module.code_count = state['code_count'].to(module.codebook.device) if state['code_count'] is not None else None
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return (checkpoint['ep'], checkpoint['total_it'])

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None):
        self.vq_model.to(self.device)
        self.opt_vq_model = optim.AdamW([{'params': self.vq_model.encoder1d.parameters()}, {'params': self.vq_model.decoder1d.parameters()}, {'params': self.vq_model.quantizer1d.parameters()}, {'params': self.vq_model.encoder2d.parameters()}, {'params': self.vq_model.decoder2d.parameters()}, {'params': self.vq_model.quantizer2d.parameters()}, {'params': self.vq_model.linear_merge.parameters()}, {'params': self.vq_model.linear_out.parameters()}], lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model, milestones=self.opt.milestones, gamma=self.opt.gamma)
        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            (epoch, it) = self.resume(model_dir)
            print('Load model epoch:%d iterations:%d' % (epoch, it))
        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(eval_val_loader)))
        current_lr = self.opt.lr
        logs = defaultdict(float, OrderedDict())
        (best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer) = evaluation_vqvae(self.opt.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=1000, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper, plot_func=plot_eval, save=False)
        while epoch < self.opt.max_epoch:
            self.vq_model.train()
            for (i, batch_data) in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                (loss, loss1, loss2, loss_rec, loss_vel, loss_commit, perplexity) = self.forward(batch_data)
                loss_all = 1 * (loss1 + loss2) + loss
                self.opt_vq_model.zero_grad()
                loss_all.backward()
                self.opt_vq_model.step()
                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()
                logs['loss'] += loss.item()
                logs['loss_rec'] += loss_rec.item()
                logs['loss_vel'] += loss_vel.item()
                logs['loss_commit'] += loss_commit.item()
                logs['perplexity'] += perplexity.item()
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for (tag, value) in logs.items():
                        self.logger.add_scalar('Train/%s' % tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(float, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)
                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1
            print('==> Validation time:')
            self.vq_model.eval()
            val_loss_rec = []
            val_loss_vel = []
            val_loss_commit = []
            val_loss = []
            val_perpexity = []
            with torch.no_grad():
                for (i, batch_data) in enumerate(val_loader):
                    (loss, _, _, loss_rec, loss_vel, loss_commit, perplexity) = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_loss_rec.append(loss_rec.item())
                    val_loss_vel.append(loss_vel.item())
                    val_loss_commit.append(loss_commit.item())
                    val_perpexity.append(perplexity.item())
            self.logger.add_scalar('Val/loss', sum(val_loss) / len(val_loss), epoch)
            self.logger.add_scalar('Val/loss_rec', sum(val_loss_rec) / len(val_loss_rec), epoch)
            self.logger.add_scalar('Val/loss_vel', sum(val_loss_vel) / len(val_loss_vel), epoch)
            self.logger.add_scalar('Val/loss_commit', sum(val_loss_commit) / len(val_loss), epoch)
            self.logger.add_scalar('Val/loss_perplexity', sum(val_perpexity) / len(val_loss_rec), epoch)
            print('Validation Loss: %.5f Reconstruction: %.5f, Velocity: %.5f, Commit: %.5f' % (sum(val_loss) / len(val_loss), sum(val_loss_rec) / len(val_loss), sum(val_loss_vel) / len(val_loss), sum(val_loss_commit) / len(val_loss)))
            (best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer) = evaluation_vqvae(self.opt.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=best_fid, best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3, best_matching=best_matching, eval_wrapper=eval_wrapper, plot_func=plot_eval)
            if epoch % self.opt.eval_every_e == 0:
                data = torch.cat([self.motions[:4], self.pred_motion[:4]], dim=0).detach().cpu().numpy()
                save_dir = pjoin(self.opt.eval_dir, 'E%04d' % epoch)
                os.makedirs(save_dir, exist_ok=True)
                plot_eval(data, save_dir)
