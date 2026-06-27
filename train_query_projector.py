"""
Train QueryProjector: KL-divergence alignment from BMM teacher to z_e student.

The projector maps CLIP 512-d text embeddings into the VQ-VAE z_e 1024-d latent
space.  A frozen BMM (Part_TMR) retriever provides soft-target rankings as teacher
signal, and the projector learns to reproduce those rankings in z_e space.

Usage (after database_ze/ and database/ are both built):
    cd ReMoMask
    python train_query_projector.py \
        --database_ze_path database_ze \
        --database_bmm_path database \
        --output_dir logs/query_projector \
        --epochs 200 \
        --batch_size 128 \
        --device cuda:0
"""

import argparse
import json
import os
import sys
import time
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.rag.query_projector import QueryProjector, save_projector


# -------------------------------------------------------------------------
# Evaluation helpers
# -------------------------------------------------------------------------

@torch.no_grad()
def recall_at_k(projector: QueryProjector, clip_texts: torch.Tensor,
                ze_motions: torch.Tensor, ks=(1, 5, 10),
                batch_size: int = 256) -> dict:
    """Compute Recall@K: for each text, check if its paired motion (same index)
    appears in the top-K of the student's cosine ranking in z_e space."""
    projector.eval()
    N = clip_texts.shape[0]
    hits = {k: 0 for k in ks}

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_text = clip_texts[start:end]  # (B, 512)
        projected = projector(batch_text)   # (B, ze_dim)

        # Cosine similarity against entire library
        sim = projected @ ze_motions.T      # (B, N)

        # For each sample, ground truth is index (start + i)
        for i in range(sim.shape[0]):
            gt_idx = start + i
            _, topk_indices = sim[i].topk(max(ks))
            for k in ks:
                if gt_idx in topk_indices[:k]:
                    hits[k] += 1

    return {f"R@{k}": hits[k] / N for k in ks}


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load precomputed features
    # ------------------------------------------------------------------
    print("Loading z_e motion features ...")
    ze_motions_raw = np.load(pjoin(args.database_ze_path, "encoded_motions.npy"))
    ze_motions_raw = ze_motions_raw[:, 0, :]  # (N, code_dim2d)
    ze_dim = ze_motions_raw.shape[1]
    print(f"  z_e motions: {ze_motions_raw.shape}  (ze_dim = {ze_dim})")

    print("Loading CLIP text features ...")
    clip_texts_raw = np.load(pjoin(args.database_ze_path, "encoded_texts_clip.npy"))
    clip_texts_raw = clip_texts_raw[:, 0, :]  # (N, 512)
    clip_dim = clip_texts_raw.shape[1]
    print(f"  CLIP texts:  {clip_texts_raw.shape}  (clip_dim = {clip_dim})")

    print("Loading BMM (Part_TMR) features ...")
    bmm_motions_raw = np.load(pjoin(args.database_bmm_path, "encoded_motions.npy"))
    bmm_motions_raw = bmm_motions_raw[:, 0, :]  # (N, bmm_dim)
    bmm_texts_raw = np.load(pjoin(args.database_bmm_path, "encoded_texts.npy"))
    bmm_texts_raw = bmm_texts_raw[:, 0, :]  # (N, bmm_dim)
    print(f"  BMM motions: {bmm_motions_raw.shape}")
    print(f"  BMM texts:   {bmm_texts_raw.shape}")

    N = ze_motions_raw.shape[0]
    assert clip_texts_raw.shape[0] == N, (
        f"Sample count mismatch: z_e has {N}, CLIP has {clip_texts_raw.shape[0]}"
    )
    # BMM database may have different sample count if database/ was built
    # with a different run.  We align by taking the minimum and warning.
    N_bmm = bmm_motions_raw.shape[0]
    if N_bmm != N:
        print(f"  WARNING: BMM has {N_bmm} samples vs z_e has {N}; "
              f"using min({N_bmm}, {N}) = {min(N_bmm, N)}")
        N = min(N, N_bmm)
        ze_motions_raw = ze_motions_raw[:N]
        clip_texts_raw = clip_texts_raw[:N]
        bmm_motions_raw = bmm_motions_raw[:N]
        bmm_texts_raw = bmm_texts_raw[:N]

    # Move to GPU as tensors (total ~250 MB for N=23K, fine)
    ze_motions = torch.from_numpy(ze_motions_raw).float().to(device)   # (N, ze_dim)
    clip_texts = torch.from_numpy(clip_texts_raw).float().to(device)   # (N, 512)
    bmm_motions = torch.from_numpy(bmm_motions_raw).float().to(device) # (N, bmm_dim)
    bmm_texts = torch.from_numpy(bmm_texts_raw).float().to(device)     # (N, bmm_dim)

    # Ensure all features are L2-normalised
    ze_motions = F.normalize(ze_motions, dim=1)
    clip_texts = F.normalize(clip_texts, dim=1)
    bmm_motions = F.normalize(bmm_motions, dim=1)
    bmm_texts = F.normalize(bmm_texts, dim=1)

    # ------------------------------------------------------------------
    # 2. Precompute BMM teacher similarity matrix (N, N) — soft targets
    #    For memory, we compute row-by-row and keep top-K candidates.
    # ------------------------------------------------------------------
    print("Precomputing BMM teacher rankings ...")
    # Full teacher sim: (N, N).  N=23K -> 23K*23K*4 = ~2.1 GB.
    # That's borderline.  Use top-K sparse representation instead.
    K_teacher = min(args.teacher_topk, N)

    # Compute in chunks to avoid OOM
    teacher_topk_indices = torch.zeros(N, K_teacher, dtype=torch.long, device=device)
    teacher_topk_scores = torch.zeros(N, K_teacher, dtype=torch.float32, device=device)

    chunk = 512
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        sim_chunk = bmm_texts[start:end] @ bmm_motions.T  # (chunk, N)
        scores, indices = sim_chunk.topk(K_teacher, dim=1)
        teacher_topk_indices[start:end] = indices
        teacher_topk_scores[start:end] = scores

    print(f"  Teacher top-{K_teacher} precomputed.")

    # ------------------------------------------------------------------
    # 3. Build model, optimiser, scheduler
    # ------------------------------------------------------------------
    projector = QueryProjector(clip_dim=clip_dim, ze_dim=ze_dim).to(device)
    n_params = sum(p.numel() for p in projector.parameters())
    print(f"QueryProjector: {n_params:,} params  (clip_dim={clip_dim}, ze_dim={ze_dim})")

    optimizer = torch.optim.Adam(projector.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Index array for shuffled batching
    indices = torch.arange(N, device=device)

    best_r1 = -1.0
    temperature = args.temperature
    log_records = []

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"lr={args.lr}, temperature={temperature}\n")

    for epoch in range(1, args.epochs + 1):
        projector.train()
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            batch_idx = perm[start:end]
            B = batch_idx.shape[0]

            # Student forward
            batch_clip = clip_texts[batch_idx]              # (B, 512)
            projected = projector(batch_clip)                # (B, ze_dim)

            # Student similarity: projected text vs teacher's top-K z_e motions
            # For each sample i, gather the teacher's top-K candidate motions
            batch_teacher_idx = teacher_topk_indices[batch_idx]   # (B, K_teacher)
            batch_teacher_scores = teacher_topk_scores[batch_idx] # (B, K_teacher)

            # Gather candidate z_e motions
            flat_idx = batch_teacher_idx.reshape(-1)                     # (B*K_teacher,)
            candidate_ze = ze_motions[flat_idx].reshape(B, K_teacher, ze_dim)  # (B, K, ze_dim)

            # Student cosine sim within candidate set
            student_sim = torch.bmm(
                projected.unsqueeze(1),                      # (B, 1, ze_dim)
                candidate_ze.transpose(1, 2)                 # (B, ze_dim, K)
            ).squeeze(1)                                     # (B, K_teacher)

            # Temperature-scaled distributions
            student_log_prob = F.log_softmax(student_sim / temperature, dim=1)
            teacher_prob = F.softmax(batch_teacher_scores / temperature, dim=1)

            # KL divergence: KL(teacher || student)
            loss = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Evaluate every eval_interval epochs
        if epoch % args.eval_interval == 0 or epoch == 1 or epoch == args.epochs:
            metrics = recall_at_k(projector, clip_texts, ze_motions,
                                  ks=(1, 5, 10), batch_size=512)
            r1 = metrics["R@1"]
            r5 = metrics["R@5"]
            r10 = metrics["R@10"]
            lr_now = scheduler.get_last_lr()[0]

            log_line = (f"Epoch {epoch:4d}/{args.epochs} | "
                        f"loss={avg_loss:.6f} | "
                        f"R@1={r1:.4f} R@5={r5:.4f} R@10={r10:.4f} | "
                        f"lr={lr_now:.2e}")
            print(log_line)

            log_records.append({
                "epoch": epoch, "loss": avg_loss,
                "R@1": r1, "R@5": r5, "R@10": r10, "lr": lr_now
            })

            # Save best
            if r1 > best_r1:
                best_r1 = r1
                best_path = pjoin(args.output_dir, "best_projector.pt")
                save_projector(projector, best_path, extra={
                    "epoch": epoch, "R@1": r1, "R@5": r5, "R@10": r10,
                    "loss": avg_loss, "temperature": temperature,
                })
                print(f"  -> New best R@1={r1:.4f}, saved to {best_path}")
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch:4d}/{args.epochs} | loss={avg_loss:.6f}")

    # ------------------------------------------------------------------
    # 5. Final: project all texts and save to database_ze/encoded_texts.npy
    # ------------------------------------------------------------------
    print("\nProjecting all texts with best model ...")
    best_ckpt = pjoin(args.output_dir, "best_projector.pt")
    if os.path.exists(best_ckpt):
        from models.rag.query_projector import load_projector
        projector = load_projector(best_ckpt, map_location=args.device)
        projector.to(device)
    projector.eval()

    all_projected = []
    with torch.no_grad():
        for start in range(0, N, 512):
            end = min(start + 512, N)
            batch = clip_texts[start:end]
            proj = projector(batch)          # (B, ze_dim)
            all_projected.append(proj.cpu().numpy())

    all_projected = np.concatenate(all_projected, axis=0)  # (N, ze_dim)
    # Reshape to (N, 1, ze_dim) to match database format
    all_projected = all_projected[:, np.newaxis, :]        # (N, 1, ze_dim)

    out_path = pjoin(args.database_ze_path, "encoded_texts.npy")
    np.save(out_path, all_projected)
    print(f"Saved projected text features: {all_projected.shape} -> {out_path}")

    # ------------------------------------------------------------------
    # 6. Save training log
    # ------------------------------------------------------------------
    log_path = pjoin(args.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "args": vars(args),
            "best_R@1": best_r1,
            "clip_dim": clip_dim,
            "ze_dim": ze_dim,
            "n_samples": N,
            "n_params": n_params,
            "records": log_records,
        }, f, indent=2)
    print(f"Training log saved to {log_path}")

    # Save last checkpoint
    last_path = pjoin(args.output_dir, "last_projector.pt")
    save_projector(projector, last_path, extra={
        "epoch": args.epochs, "loss": avg_loss,
    })
    print(f"Last checkpoint: {last_path}")
    print(f"\nDone. Best R@1 = {best_r1:.4f}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train QueryProjector: KL alignment (BMM teacher -> z_e student)")
    parser.add_argument("--database_ze_path", type=str, default="database_ze",
                        help="Path to z_e database (encoded_motions.npy, encoded_texts_clip.npy)")
    parser.add_argument("--database_bmm_path", type=str, default="database",
                        help="Path to BMM (Part_TMR) database (encoded_motions.npy, encoded_texts.npy)")
    parser.add_argument("--output_dir", type=str, default="logs/query_projector",
                        help="Where to save checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (per D-15)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size (per D-15)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for Adam")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature for softmax (matches Part_TMR)")
    parser.add_argument("--teacher_topk", type=int, default=256,
                        help="Number of BMM teacher candidates per text (efficiency)")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Evaluate R@K every N epochs")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
