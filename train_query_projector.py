import re

def split_motion_id(full_id):
    m = re.match('^(M?\\d+)_(\\d+)$', full_id)
    if m is None:
        return (full_id, 0)
    return (m.group(1), int(m.group(2)))
import argparse
import json
import os
from os.path import join as pjoin
import numpy as np
import torch
import torch.nn.functional as F
from models.rag.query_projector import QueryProjector, save_projector
from utils.fixseed import fixseed

@torch.no_grad()
def recall_at_k(projector: QueryProjector, clip_texts: torch.Tensor, ze_motions: torch.Tensor, eval_idx: torch.Tensor=None, ks=(1, 5, 10), batch_size: int=256) -> dict:
    projector.eval()
    N = clip_texts.shape[0]
    if eval_idx is None:
        eval_idx = torch.arange(N, device=clip_texts.device)
    n_eval = eval_idx.shape[0]
    hits = {k: 0 for k in ks}
    for start in range(0, n_eval, batch_size):
        end = min(start + batch_size, n_eval)
        batch_rows = eval_idx[start:end]
        batch_text = clip_texts[batch_rows]
        projected = projector(batch_text)
        sim = projected @ ze_motions.T
        for i in range(sim.shape[0]):
            gt_idx = int(batch_rows[i].item())
            (_, topk_indices) = sim[i].topk(max(ks))
            for k in ks:
                if gt_idx in topk_indices[:k]:
                    hits[k] += 1
    return {f'R@{k}': hits[k] / n_eval for k in ks}

@torch.no_grad()
def teacher_consistency_at_k(projector: QueryProjector, clip_texts: torch.Tensor, ze_motions: torch.Tensor, teacher_topk_indices: torch.Tensor, eval_idx: torch.Tensor, ks=(1,), batch_size: int=256) -> dict:
    projector.eval()
    n_eval = eval_idx.shape[0]
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    for start in range(0, n_eval, batch_size):
        end = min(start + batch_size, n_eval)
        batch_rows = eval_idx[start:end]
        batch_text = clip_texts[batch_rows]
        projected = projector(batch_text)
        sim = projected @ ze_motions.T
        (_, topk_indices) = sim.topk(max_k, dim=1)
        for i in range(batch_rows.shape[0]):
            gt_idx = int(batch_rows[i].item())
            teacher_set = set(teacher_topk_indices[gt_idx].tolist())
            student_topk = topk_indices[i]
            for k in ks:
                student_set = set(student_topk[:k].tolist())
                if student_set & teacher_set:
                    hits[k] += 1
    return {f'TeacherConsistency@{k}': hits[k] / n_eval for k in ks}

def _held_out_split(motion_ids, held_out_frac: float, split_seed: int):
    base_ids = np.array([split_motion_id(str(mid))[0] for mid in motion_ids])
    unique_bases = np.unique(base_ids)
    rng = np.random.default_rng(split_seed)
    shuffled = rng.permutation(unique_bases)
    n_held_out_bases = max(1, int(round(len(shuffled) * held_out_frac)))
    held_out_bases = set(shuffled[:n_held_out_bases].tolist())
    is_held_out = np.array([b in held_out_bases for b in base_ids])
    held_out_idx = np.nonzero(is_held_out)[0].astype(np.int64)
    train_idx = np.nonzero(~is_held_out)[0].astype(np.int64)
    return (train_idx, held_out_idx)

def train(args):
    fixseed(args.split_seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    print('Loading z_e motion features ...')
    ze_motions_raw = np.load(pjoin(args.database_ze_path, 'encoded_motions.npy'))
    ze_motions_raw = ze_motions_raw[:, 0, :]
    ze_dim = ze_motions_raw.shape[1]
    print(f'  z_e motions: {ze_motions_raw.shape}  (ze_dim = {ze_dim})')
    print('Loading CLIP text features ...')
    clip_texts_raw = np.load(pjoin(args.database_ze_path, 'encoded_texts_clip.npy'))
    clip_texts_raw = clip_texts_raw[:, 0, :]
    clip_dim = clip_texts_raw.shape[1]
    print(f'  CLIP texts:  {clip_texts_raw.shape}  (clip_dim = {clip_dim})')
    print('Loading BMM (Part_TMR) features ...')
    bmm_motions_raw = np.load(pjoin(args.database_bmm_path, 'encoded_motions.npy'))
    bmm_motions_raw = bmm_motions_raw[:, 0, :]
    bmm_texts_raw = np.load(pjoin(args.database_bmm_path, 'encoded_texts.npy'))
    bmm_texts_raw = bmm_texts_raw[:, 0, :]
    print(f'  BMM motions: {bmm_motions_raw.shape}')
    print(f'  BMM texts:   {bmm_texts_raw.shape}')
    N = ze_motions_raw.shape[0]
    assert clip_texts_raw.shape[0] == N, f'Sample count mismatch: z_e has {N}, CLIP has {clip_texts_raw.shape[0]}'
    print('Loading motion_ids for held-out split ...')
    motion_ids = np.load(pjoin(args.database_ze_path, 'motion_ids.npy'), allow_pickle=True)
    assert motion_ids.shape[0] == N, f'motion_ids has {motion_ids.shape[0]} rows, expected {N}'
    assert bmm_motions_raw.shape[0] == bmm_texts_raw.shape[0] == N
    bmm_ids = np.load(pjoin(args.database_bmm_path, 'motion_ids.npy'), allow_pickle=True)
    ze_captions = np.load(pjoin(args.database_ze_path, 'all_captions.npy'), allow_pickle=True)
    bmm_captions = np.load(pjoin(args.database_bmm_path, 'all_captions.npy'), allow_pickle=True)
    assert np.array_equal(motion_ids, bmm_ids), 'Retrieval database row order mismatch'
    assert np.array_equal(ze_captions, bmm_captions), 'Retrieval database captions mismatch'
    (train_idx, held_out_idx) = _held_out_split(motion_ids, args.held_out_frac, args.split_seed)
    n_train = train_idx.shape[0]
    n_held_out = held_out_idx.shape[0]
    print(f'Held-out split (held_out_frac={args.held_out_frac}, split_seed={args.split_seed}): {n_train} train rows, {n_held_out} held-out rows (by base motion_id)')
    ze_motions = torch.from_numpy(ze_motions_raw).float().to(device)
    clip_texts = torch.from_numpy(clip_texts_raw).float().to(device)
    bmm_motions = torch.from_numpy(bmm_motions_raw).float().to(device)
    bmm_texts = torch.from_numpy(bmm_texts_raw).float().to(device)
    ze_motions = F.normalize(ze_motions, dim=1)
    clip_texts = F.normalize(clip_texts, dim=1)
    bmm_motions = F.normalize(bmm_motions, dim=1)
    bmm_texts = F.normalize(bmm_texts, dim=1)
    train_idx_t = torch.from_numpy(train_idx).long().to(device)
    held_out_idx_t = torch.from_numpy(held_out_idx).long().to(device)
    print('Precomputing teacher rankings ...')
    K_teacher = min(args.teacher_topk, N)
    teacher_topk_indices = torch.zeros(N, K_teacher, dtype=torch.long, device=device)
    teacher_topk_scores = torch.zeros(N, K_teacher, dtype=torch.float32, device=device)
    chunk = 512
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        sim_chunk = bmm_texts[start:end] @ bmm_motions.T
        (scores, indices) = sim_chunk.topk(K_teacher, dim=1)
        teacher_topk_indices[start:end] = indices
        teacher_topk_scores[start:end] = scores
    print(f'  Teacher top-{K_teacher} precomputed.')
    projector = QueryProjector(clip_dim=clip_dim, ze_dim=ze_dim, hidden=1024).to(device)
    n_params = sum((p.numel() for p in projector.parameters()))
    print(f'QueryProjector: {n_params:,} params  (clip_dim={clip_dim}, ze_dim={ze_dim}, objective=kl)')
    optimizer = torch.optim.Adam(projector.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_consistency = -1.0
    temperature = args.temperature
    log_records = []
    print(f'\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}, temperature={temperature}, n_train={n_train}, n_held_out={n_held_out}\n')
    for epoch in range(1, args.epochs + 1):
        projector.train()
        perm = train_idx_t[torch.randperm(n_train, device=device)]
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, args.batch_size):
            end = min(start + args.batch_size, n_train)
            batch_idx = perm[start:end]
            B = batch_idx.shape[0]
            batch_clip = clip_texts[batch_idx]
            projected = projector(batch_clip)
            batch_teacher_idx = teacher_topk_indices[batch_idx]
            batch_teacher_scores = teacher_topk_scores[batch_idx]
            flat_idx = batch_teacher_idx.reshape(-1)
            candidate_ze = ze_motions[flat_idx].reshape(B, K_teacher, ze_dim)
            student_sim = torch.bmm(projected.unsqueeze(1), candidate_ze.transpose(1, 2)).squeeze(1)
            student_log_prob = F.log_softmax(student_sim / temperature, dim=1)
            teacher_prob = F.softmax(batch_teacher_scores / temperature, dim=1)
            loss = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % args.eval_interval == 0 or epoch == 1 or epoch == args.epochs:
            r_metrics = recall_at_k(projector, clip_texts, ze_motions, eval_idx=held_out_idx_t, ks=(1, 5, 10), batch_size=512)
            r1_heldout = r_metrics['R@1']
            r5_heldout = r_metrics['R@5']
            r10_heldout = r_metrics['R@10']
            c_metrics = teacher_consistency_at_k(projector, clip_texts, ze_motions, teacher_topk_indices, eval_idx=held_out_idx_t, ks=(1, 5, 10), batch_size=512)
            consistency_at_1 = c_metrics['TeacherConsistency@1']
            consistency_at_5 = c_metrics['TeacherConsistency@5']
            consistency_at_10 = c_metrics['TeacherConsistency@10']
            lr_now = scheduler.get_last_lr()[0]
            log_line = f'Epoch {epoch:4d}/{args.epochs} | loss={avg_loss:.6f} | TeacherConsistency@1={consistency_at_1:.4f} @5={consistency_at_5:.4f} @10={consistency_at_10:.4f} | R@1(heldout)={r1_heldout:.4f} R@5={r5_heldout:.4f} R@10={r10_heldout:.4f} | lr={lr_now:.2e}'
            print(log_line)
            log_records.append({'epoch': epoch, 'loss': avg_loss, 'TeacherConsistency@1': consistency_at_1, 'TeacherConsistency@5': consistency_at_5, 'TeacherConsistency@10': consistency_at_10, 'R@1_heldout': r1_heldout, 'R@5_heldout': r5_heldout, 'R@10_heldout': r10_heldout, 'lr': lr_now})
            if consistency_at_1 > best_consistency:
                best_consistency = consistency_at_1
                best_path = pjoin(args.output_dir, 'best_projector.pt')
                save_projector(projector, best_path, extra={'epoch': epoch, 'TeacherConsistency@1': consistency_at_1, 'R@1_heldout': r1_heldout, 'R@5_heldout': r5_heldout, 'R@10_heldout': r10_heldout, 'held_out_frac': args.held_out_frac, 'split_seed': args.split_seed, 'loss': avg_loss, 'temperature': temperature, 'objective': 'kl'})
                print(f'  -> New best TeacherConsistency@1={consistency_at_1:.4f}, saved to {best_path}')
        elif epoch % 10 == 0:
            print(f'Epoch {epoch:4d}/{args.epochs} | loss={avg_loss:.6f}')
    log_path = pjoin(args.output_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump({'args': vars(args), 'best_TeacherConsistency@1': best_consistency, 'n_train': n_train, 'n_held_out': n_held_out, 'clip_dim': clip_dim, 'ze_dim': ze_dim, 'n_samples': N, 'n_params': n_params, 'records': log_records}, f, indent=2)
    print(f'Training log saved to {log_path}')
    last_path = pjoin(args.output_dir, 'last_projector.pt')
    save_projector(projector, last_path, extra={'epoch': args.epochs, 'loss': avg_loss, 'objective': 'kl'})
    print(f'Last checkpoint: {last_path}')
    print(f'\nDone. Best TeacherConsistency@1 = {best_consistency:.4f}')

def parse_args():
    parser = argparse.ArgumentParser(description='Train query projector')
    parser.add_argument('--database_ze_path', type=str, default='database_ze')
    parser.add_argument('--database_bmm_path', type=str, default='database')
    parser.add_argument('--output_dir', type=str, default='logs/query_projector')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--teacher_topk', type=int, default=256)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--held_out_frac', type=float, default=0.1)
    parser.add_argument('--split_seed', type=int, default=3407)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    train(args)
