import itertools
import logging
import os
import sys
from os.path import join as pjoin
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.insert(0, os.getcwd())
from Part_TMR.datasets import TextMotionDataset
from Part_TMR.scripts.test import eval, prepare_test_dataset
from Part_TMR.scripts.utils import set_seed
import time
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
log = logging.getLogger(__name__)
from Part_TMR.models.builder_bimoco import MoCoTMR

@hydra.main(version_base=None, config_name='config', config_path='../conf')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print('----------- details ------------')
    print(f'Batch size: {cfg.train.batch_size}')
    print(f"Queue size: {(cfg.model.queue_size if hasattr(cfg.model, 'queue_size') else 'NOT SET')}")
    print(f'motion_lr: {cfg.train.optimizer.motion_lr * cfg.dataset.motion_lr_factor}')
    print(f'text_lr: {cfg.train.optimizer.text_lr * cfg.dataset.text_lr_factor}')
    print(f'head_lr: {cfg.train.optimizer.head_lr * cfg.dataset.head_lr_factor}')
    set_seed(cfg.train.seed)
    (train_dataloader, test_dataloader) = prepare_dataset(cfg)
    eval_dataloader = prepare_test_dataset(cfg)
    (model, optimizer, scheduler) = prepare_model(cfg, train_dataloader)
    print(f'----------- model.device: {model.device} ------------')
    train(cfg, train_dataloader, test_dataloader, eval_dataloader, model, optimizer, scheduler)
    print(f'----------- model.device: {model.device} ------------')

def prepare_dataset(cfg):
    mean = np.load(pjoin(cfg.dataset.data_root, 'Mean.npy'))
    std = np.load(pjoin(cfg.dataset.data_root, 'Std.npy'))
    train_split_file = pjoin(cfg.dataset.data_root, cfg.dataset.train_split_filename)
    train_dataset = TextMotionDataset(cfg, mean, std, train_split_file, fps=True)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True, num_workers=16)
    val_split_file = pjoin(cfg.dataset.data_root, 'val.txt')
    val_dataset = TextMotionDataset(cfg, mean, std, val_split_file, fps=True)
    test_dataloader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=16)
    return (train_dataloader, test_dataloader)

def prepare_model(cfg, train_dataloader):
    device = cfg.device
    text_encoder_alias = cfg.model.text_encoder
    text_encoder_trainable: bool = cfg.train.train_text_encoder
    motion_embedding_dims: int = cfg.model.motion_embedding_dims
    text_embedding_dims: int = cfg.model.text_embedding_dims
    projection_dims: int = cfg.model.projection_dims
    model = MoCoTMR(text_encoder_alias, text_encoder_trainable, motion_embedding_dims, text_embedding_dims, projection_dims, dropout=0.5 if cfg.dataset.dataset_name == 'HumanML3D' else 0.0, mode='t2m' if cfg.dataset.dataset_name == 'HumanML3D' else 'kit', temp=0.07, alpha=0.9996, config=cfg)
    model.to(device)
    projection_params = [model.motion_projection.parameters(), model.text_projection.parameters()]
    parameters = [{'params': model.motion_encoder.parameters(), 'lr': cfg.train.optimizer.motion_lr * cfg.dataset.motion_lr_factor}, {'params': model.text_encoder.parameters(), 'lr': cfg.train.optimizer.text_lr * cfg.dataset.text_lr_factor}, {'params': itertools.chain(*projection_params), 'lr': cfg.train.optimizer.head_lr * cfg.dataset.head_lr_factor}]
    optimizer = optim.Adam(parameters)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader) * cfg.train.epoch * 2)
    return (model, optimizer, scheduler)

def estimate_eta(start_time, current_step, total_steps):
    elapsed = time.time() - start_time
    avg_time = elapsed / (current_step + 1e-08)
    remaining_time = avg_time * (total_steps - current_step)
    return time.strftime('%H:%M:%S', time.gmtime(remaining_time))

def train(cfg, train_dataloader, test_dataloader, eval_dataloader, model, optimizer, scheduler):
    device = cfg.device
    best_te_loss = 100000.0
    best_t2m_r1 = 0
    best_m2t_r1 = 0
    best_ep = -1
    total_train_steps = len(train_dataloader) * cfg.train.epoch
    train_start = time.time()
    for epoch in range(cfg.train.epoch):
        print(f'running epoch {epoch}, best test loss {best_te_loss} best_t2m_r1 {best_t2m_r1} best_m2t_r1 {best_m2t_r1} after epoch {best_ep}')
        step = 0
        tr_loss = 0
        tr_loss_inst = 0
        tr_loss_part = 0
        model.train()
        pbar = tqdm(train_dataloader, leave=False)
        for batch in pbar:
            step += 1
            train_step_start = time.time()
            optimizer.zero_grad()
            (Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm, texts, _, _) = batch
            captions = list(texts)
            (Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm) = (Root.to(device), R_Leg.to(device), L_Leg.to(device), Backbone.to(device), R_Arm.to(device), L_Arm.to(device))
            motions = [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]
            texts = model.tokenize(texts)
            texts = texts.to(device)
            model_output = model(motions, texts, captions, return_loss=True)
            if isinstance(model_output, tuple):
                (total_loss, loss_dict) = model_output
                loss_inst = loss_dict['instance'].item()
                loss_part = loss_dict['part'].item()
                tr_loss_inst += loss_inst
                tr_loss_part += loss_part
            else:
                total_loss = model_output
                loss_inst = 0
                loss_part = 0
            total_loss.backward()
            tr_loss += total_loss.item()
            optimizer.step()
            scheduler.step()
            train_step_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - train_step_start))
            eta = estimate_eta(train_start, epoch * step + step, total_train_steps)
            elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start))
            if isinstance(model_output, tuple):
                pbar.set_description(f'Total: {total_loss.item():.4f} | Inst: {loss_inst:.4f} | Part: {loss_part:.4f} | Step: {train_step_time} | ETA: {eta}')
            else:
                pbar.set_description(f'bacthCE: {total_loss.item():.4f} | Step: {train_step_time} | Elapsed: {elapsed_time} | ETA: {eta}')
        tr_loss /= step
        if tr_loss_inst > 0:
            tr_loss_inst /= step
            tr_loss_part /= step
        step = 0
        te_loss = 0
        te_loss_inst = 0
        te_loss_part = 0
        with torch.no_grad():
            model.eval()
            test_pbar = tqdm(test_dataloader, leave=False)
            for batch in test_pbar:
                step += 1
                (Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm, texts, _, _) = batch
                captions = list(texts)
                (Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm) = (Root.to(device), R_Leg.to(device), L_Leg.to(device), Backbone.to(device), R_Arm.to(device), L_Arm.to(device))
                motions = [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]
                texts = model.tokenize(texts)
                texts = texts.to(device)
                model_output = model(motions, texts, captions, return_loss=True)
                if isinstance(model_output, tuple):
                    (total_loss, loss_dict) = model_output
                    te_loss_inst += loss_dict['instance'].item()
                    te_loss_part += loss_dict['part'].item()
                else:
                    total_loss = model_output
                te_loss += total_loss.item()
                test_pbar.set_description(f'test batchCE: {total_loss.item()}', refresh=True)
            te_loss /= step
            if te_loss_inst > 0:
                te_loss_inst /= step
                te_loss_part /= step
        if te_loss < best_te_loss:
            best_te_loss = te_loss
        Path(cfg.checkpoints_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), pjoin(cfg.checkpoints_dir, 'last_model.pt'))
        (t2m_r1, m2t_r1) = eval(cfg, eval_dataloader, model, verbose=False)
        if tr_loss_inst > 0:
            log.info(f'epoch {epoch}, tr_loss {tr_loss:.4f} (inst: {tr_loss_inst:.4f}, part: {tr_loss_part:.4f}), te_loss {te_loss:.4f} (inst: {te_loss_inst:.4f}, part: {te_loss_part:.4f}), t2m_r1 {t2m_r1:.2f}, m2t_r1 {m2t_r1:.2f}')
        else:
            log.info(f'epoch {epoch}, tr_loss {tr_loss}, te_loss {te_loss}, t2m_r1 {t2m_r1}, m2t_r1 {m2t_r1} ')
        best_t2m_r1 = max(best_t2m_r1, t2m_r1)
        best_m2t_r1 = max(best_m2t_r1, m2t_r1)
        if best_t2m_r1 == t2m_r1 and abs(t2m_r1 - m2t_r1) < 10:
            best_ep = epoch
            torch.save(model.state_dict(), pjoin(cfg.checkpoints_dir, 'best_model.pt'))
if __name__ == '__main__':
    main()
