"""
z_e Geometry Verification (LA-01)
=================================
Read-only analysis script: extracts z_e from the VQ-VAE encoder2d,
then measures cosine similarity distribution, anisotropy, and
dimensional collapse.

Usage:
    cd ReMoMask
    python scripts/analyze_ze.py \
        --vq_checkpoint logs/humanml3d/pretrain_vq/model/net_best_fid.tar \
        --dataset_root dataset/HumanML3D \
        --output_dir results/ze_analysis \
        --max_samples 2000
"""

import argparse
import json
import os
import sys
from os.path import join as pjoin

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

# ---- project imports (run from ReMoMask/) --------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vq.model import RVQVAE
from utils.get_opt import get_opt


# ---------------------------------------------------------------------------
# Helper: construct x_joints (T, 22, 12) from 263-dim motion
# ---------------------------------------------------------------------------
def motion_to_x2d(motion: np.ndarray, n_j: int = 22) -> np.ndarray:
    """Convert 263-dim HumanML3D motion to per-joint (T, J, 12) tensor,
    matching the logic in data/t2m_dataset.py."""
    T = motion.shape[0]
    n_feat = 12
    x2 = motion[:, 4 : 4 + (n_j - 1) * 3]
    x3 = motion[:, 4 + (n_j - 1) * 3 : 4 + (n_j - 1) * 9]
    x4 = motion[:, 4 + (n_j - 1) * 9 : 4 + (n_j - 1) * 9 + n_j * 3]

    x_pos = x2.reshape(T, n_j - 1, 3)
    x_rot = x3.reshape(T, n_j - 1, 6)
    x_speed = x4.reshape(T, n_j, 3)

    x_joints = np.zeros([T, n_j, n_feat], dtype=np.float32)
    x_joints[:, 1:, :3] = x_pos
    x_joints[:, 1:, 3:9] = x_rot
    x_joints[:, :, 9:12] = x_speed
    return x_joints


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="z_e geometry analysis for LA-01")
    parser.add_argument("--vq_checkpoint", type=str, required=True,
                        help="Path to RVQVAE checkpoint (net_best_fid.tar)")
    parser.add_argument("--dataset_root", type=str, default="dataset/HumanML3D",
                        help="HumanML3D data root")
    parser.add_argument("--output_dir", type=str, default="results/ze_analysis",
                        help="Where to save analysis artifacts")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Max motions to process (for speed)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load RVQVAE via opt.txt
    # ------------------------------------------------------------------
    # opt.txt sits alongside the model/ directory
    ckpt_dir = os.path.dirname(os.path.dirname(args.vq_checkpoint))
    opt_path = pjoin(ckpt_dir, "opt.txt")
    assert os.path.isfile(opt_path), f"opt.txt not found at {opt_path}"

    vq_opt = get_opt(opt_path, device=args.device)

    print("\n========== VQ-VAE Key Parameters ==========")
    print(f"  code_dim1d   = {vq_opt.code_dim1d}")
    print(f"  code_dim2d   = {vq_opt.code_dim2d}")
    print(f"  nb_code1d    = {vq_opt.nb_code1d}")
    print(f"  nb_code2d    = {vq_opt.nb_code2d}")
    print(f"  num_quantizers = {vq_opt.num_quantizers}")
    print(f"  down_t       = {vq_opt.down_t}")
    print(f"  stride_t     = {vq_opt.stride_t}")
    print(f"  width        = {vq_opt.width}")
    print(f"  depth        = {vq_opt.depth}")
    print(f"  J (hardcoded)= 6")
    print("=" * 46)

    vq_model = RVQVAE(
        vq_opt,
        vq_opt.dim_pose,       # 263 for humanml3d
        vq_opt.down_t,
        vq_opt.stride_t,
        vq_opt.width,
        vq_opt.depth,
        vq_opt.dilation_growth_rate,
        vq_opt.vq_act,
        vq_opt.vq_norm,
    )

    ckpt = torch.load(args.vq_checkpoint, map_location="cpu")
    model_key = "vq_model" if "vq_model" in ckpt else "net"
    vq_model.load_state_dict(ckpt[model_key])
    vq_model.to(args.device)
    vq_model.eval()
    print(f"Loaded checkpoint: ep={ckpt.get('ep', '?')}, "
          f"best_fid={ckpt.get('value', '?')}")

    # ------------------------------------------------------------------
    # 2. Load Mean / Std and training split
    # ------------------------------------------------------------------
    mean = np.load(pjoin(args.dataset_root, "Mean.npy"))   # (263,)
    std  = np.load(pjoin(args.dataset_root, "Std.npy"))    # (263,)

    split_file = pjoin(args.dataset_root, "train.txt")
    with open(split_file, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    print(f"Training split has {len(names)} entries, using up to {args.max_samples}")

    motion_dir = pjoin(args.dataset_root, "new_joint_vecs")
    n_j = 22  # humanml3d
    max_motion_length = 196
    min_motion_length = 40

    # ------------------------------------------------------------------
    # 3. Extract z_e
    # ------------------------------------------------------------------
    ze_pooled_list = []
    ze_raw_shapes = []       # log a few raw shapes for diagnostics
    processed = 0

    for name in tqdm(names, desc="Extracting z_e"):
        if processed >= args.max_samples:
            break

        motion_path = pjoin(motion_dir, name + ".npy")
        if not os.path.exists(motion_path):
            continue

        raw_motion = np.load(motion_path)
        if len(raw_motion) < min_motion_length:
            continue
        if np.isnan(raw_motion).any():
            continue

        # Normalise
        motion = (raw_motion - mean) / std

        # Truncate / pad to max_motion_length
        m_length = motion.shape[0]
        if m_length > max_motion_length:
            motion = motion[:max_motion_length]
            m_length = max_motion_length
        else:
            pad_len = max_motion_length - m_length
            motion = np.concatenate(
                [motion, np.zeros((pad_len, 263), dtype=np.float32)], axis=0
            )

        # Build x2d: (T, 22, 12)
        x2d_np = motion_to_x2d(motion, n_j=n_j)
        x2d = torch.from_numpy(x2d_np).unsqueeze(0).to(args.device)  # (1, T, 22, 12)

        with torch.inference_mode():
            # Pad joints 22 -> 24 (same as RVQVAE.forward / encode)
            x2d_padded = torch.nn.functional.pad(x2d, (0, 0, 1, 1))  # (1, T, 24, 12)
            z_e = vq_model.encoder2d(
                rearrange(x2d_padded, "b t j d -> b d t j")
            )  # (1, code_dim2d, T/4, 6)

            if len(ze_raw_shapes) < 3:
                ze_raw_shapes.append(list(z_e.shape))

            # Mean pooling over time and joint dims -> (1, code_dim2d)
            z_e_pooled = z_e.mean(dim=[2, 3])  # (1, code_dim2d)

            # L2 normalize
            z_e_pooled = torch.nn.functional.normalize(z_e_pooled, dim=1)

            ze_pooled_list.append(z_e_pooled.cpu())

        processed += 1

    ze_all = torch.cat(ze_pooled_list, dim=0)  # (N, code_dim2d)
    N, D = ze_all.shape
    print(f"\nCollected {N} z_e vectors, dim = {D}")
    print(f"Raw z_e shapes (first 3): {ze_raw_shapes}")

    # ------------------------------------------------------------------
    # 4. Geometric Analysis
    # ------------------------------------------------------------------

    # --- 4a. Cosine similarity distribution ---
    n_pairs = min(10000, N * (N - 1) // 2)
    idx_a = torch.randint(0, N, (n_pairs,))
    idx_b = torch.randint(0, N, (n_pairs,))
    # Avoid identical pairs
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    cos_sims = torch.nn.functional.cosine_similarity(
        ze_all[idx_a], ze_all[idx_b], dim=1
    ).numpy()

    mean_cos = float(np.mean(cos_sims))
    std_cos  = float(np.std(cos_sims))
    median_cos = float(np.median(cos_sims))
    min_cos = float(np.min(cos_sims))
    max_cos = float(np.max(cos_sims))

    print(f"\n--- Cosine Similarity Distribution ---")
    print(f"  mean   = {mean_cos:.4f}")
    print(f"  std    = {std_cos:.4f}")
    print(f"  median = {median_cos:.4f}")
    print(f"  min    = {min_cos:.4f}")
    print(f"  max    = {max_cos:.4f}")

    # --- 4b. Anisotropy / isotropy score ---
    ze_np = ze_all.numpy()  # (N, D)
    ze_centered = ze_np - ze_np.mean(axis=0, keepdims=True)
    cov = np.cov(ze_centered, rowvar=False)  # (D, D)

    eigenvalues = np.linalg.eigvalsh(cov)    # sorted ascending
    eigenvalues = eigenvalues[::-1]           # descending
    max_eig = eigenvalues[0]
    min_eig = eigenvalues[-1]

    isotropy_score = float(min_eig / max_eig) if max_eig > 0 else 0.0

    # Effective dimension ratio: eigenvalues > 1% of max
    threshold = max_eig * 0.01
    n_effective = int(np.sum(eigenvalues > threshold))
    effective_dim_ratio = n_effective / D

    # Participation ratio (alternative effective dimensionality measure)
    eig_sum = eigenvalues.sum()
    eig_sq_sum = (eigenvalues ** 2).sum()
    participation_ratio = float((eig_sum ** 2) / eig_sq_sum) if eig_sq_sum > 0 else 0.0

    print(f"\n--- Anisotropy / Isotropy ---")
    print(f"  isotropy_score  = {isotropy_score:.6f}  (min_eig / max_eig)")
    print(f"  effective_dims  = {n_effective} / {D}  ({effective_dim_ratio:.1%})")
    print(f"  participation_ratio = {participation_ratio:.1f}")

    # --- 4c. Mean norm (before L2-normalisation, recompute) ---
    ze_unnorm_list = []
    processed2 = 0
    for name in names:
        if processed2 >= args.max_samples:
            break
        motion_path = pjoin(motion_dir, name + ".npy")
        if not os.path.exists(motion_path):
            continue
        raw_motion = np.load(motion_path)
        if len(raw_motion) < min_motion_length or np.isnan(raw_motion).any():
            continue
        motion = (raw_motion - mean) / std
        m_length = motion.shape[0]
        if m_length > max_motion_length:
            motion = motion[:max_motion_length]
        else:
            motion = np.concatenate(
                [motion, np.zeros((max_motion_length - m_length, 263), dtype=np.float32)], axis=0
            )
        x2d_np = motion_to_x2d(motion, n_j=n_j)
        x2d = torch.from_numpy(x2d_np).unsqueeze(0).to(args.device)
        with torch.inference_mode():
            x2d_padded = torch.nn.functional.pad(x2d, (0, 0, 1, 1))
            z_e = vq_model.encoder2d(rearrange(x2d_padded, "b t j d -> b d t j"))
            z_e_pooled = z_e.mean(dim=[2, 3])  # (1, D), unnormalised
            ze_unnorm_list.append(z_e_pooled.cpu())
        processed2 += 1
        if processed2 >= 100:  # only need ~100 for norm stats
            break

    ze_unnorm = torch.cat(ze_unnorm_list, dim=0)
    norms = ze_unnorm.norm(dim=1).numpy()
    mean_norm = float(np.mean(norms))
    std_norm  = float(np.std(norms))
    print(f"\n--- z_e Norm (before L2 normalisation) ---")
    print(f"  mean_norm = {mean_norm:.4f}")
    print(f"  std_norm  = {std_norm:.4f}")

    # ------------------------------------------------------------------
    # 5. Risk Assessment
    # ------------------------------------------------------------------
    high_anisotropy = isotropy_score < 0.01
    severe_collapse = mean_cos > 0.95

    print(f"\n--- Risk Assessment ---")
    print(f"  Anisotropy risk (isotropy < 0.01): {'HIGH' if high_anisotropy else 'OK'}")
    print(f"  Collapse risk (mean_cos > 0.95):   {'SEVERE' if severe_collapse else 'OK'}")

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    summary = {
        "code_dim2d": int(vq_opt.code_dim2d),
        "code_dim1d": int(vq_opt.code_dim1d),
        "nb_code2d": int(vq_opt.nb_code2d),
        "nb_code1d": int(vq_opt.nb_code1d),
        "num_quantizers": int(vq_opt.num_quantizers),
        "J": 6,
        "ze_raw_shape_example": ze_raw_shapes[0] if ze_raw_shapes else None,
        "num_samples": N,
        "cosine_similarity": {
            "mean": round(mean_cos, 6),
            "std": round(std_cos, 6),
            "median": round(median_cos, 6),
            "min": round(min_cos, 6),
            "max": round(max_cos, 6),
        },
        "isotropy_score": round(isotropy_score, 8),
        "effective_dim_ratio": round(effective_dim_ratio, 4),
        "effective_dims": n_effective,
        "total_dims": D,
        "participation_ratio": round(participation_ratio, 2),
        "mean_norm": round(mean_norm, 4),
        "std_norm": round(std_norm, 4),
        "risk_high_anisotropy": high_anisotropy,
        "risk_severe_collapse": severe_collapse,
        "top10_eigenvalues": [round(float(v), 6) for v in eigenvalues[:10]],
        "bottom5_eigenvalues": [round(float(v), 8) for v in eigenvalues[-5:]],
    }

    summary_path = pjoin(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary -> {summary_path}")

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Cosine similarity histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(cos_sims, bins=100, density=True, alpha=0.75, color="steelblue",
                edgecolor="white", linewidth=0.3)
        ax.axvline(mean_cos, color="red", linestyle="--", linewidth=1.5,
                   label=f"mean={mean_cos:.4f}")
        ax.set_xlabel("Cosine Similarity", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"z_e Pairwise Cosine Similarity (N={N}, D={D})", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        hist_path = pjoin(args.output_dir, "cosine_sim_hist.png")
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"Saved histogram -> {hist_path}")

        # Eigenvalue spectrum
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.semilogy(eigenvalues, color="darkorange", linewidth=1.5)
        ax.axhline(threshold, color="red", linestyle=":", linewidth=1,
                   label=f"1% threshold ({threshold:.2e})")
        ax.axvline(n_effective, color="green", linestyle="--", linewidth=1,
                   label=f"eff. dims = {n_effective}")
        ax.set_xlabel("Component index", fontsize=12)
        ax.set_ylabel("Eigenvalue (log scale)", fontsize=12)
        ax.set_title("Eigenvalue Spectrum of z_e Covariance", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        cumvar = np.cumsum(eigenvalues) / eigenvalues.sum()
        ax.plot(cumvar, color="teal", linewidth=1.5)
        ax.axhline(0.95, color="red", linestyle=":", linewidth=1, label="95% variance")
        idx_95 = int(np.searchsorted(cumvar, 0.95))
        ax.axvline(idx_95, color="green", linestyle="--", linewidth=1,
                   label=f"95% @ dim {idx_95}")
        ax.set_xlabel("Component index", fontsize=12)
        ax.set_ylabel("Cumulative variance ratio", fontsize=12)
        ax.set_title("Cumulative Variance Explained", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        eig_path = pjoin(args.output_dir, "eigenvalue_spectrum.png")
        fig.savefig(eig_path, dpi=150)
        plt.close(fig)
        print(f"Saved eigenvalue plot -> {eig_path}")

    except ImportError:
        print("matplotlib not available; skipping plots.")

    # ------------------------------------------------------------------
    # Final summary to terminal
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  z_e GEOMETRY VERIFICATION RESULTS")
    print("=" * 60)
    print(f"  code_dim2d          = {vq_opt.code_dim2d}")
    print(f"  z_e shape           = (B, {vq_opt.code_dim2d}, T/4, 6)")
    print(f"  z_e pooled shape    = (N, {D})")
    print(f"  mean cosine sim     = {mean_cos:.4f}")
    print(f"  isotropy score      = {isotropy_score:.6f}")
    print(f"  effective dims      = {n_effective}/{D} ({effective_dim_ratio:.1%})")
    print(f"  participation ratio = {participation_ratio:.1f}")
    print(f"  mean norm (raw)     = {mean_norm:.4f}")
    if severe_collapse:
        print("  ** WARNING: severe collapse detected (mean cos > 0.95) **")
    if high_anisotropy:
        print("  ** NOTE: high anisotropy (isotropy < 0.01) — consider ArcVQ **")
    else:
        print("  Anisotropy level acceptable for cosine retrieval")
    print("=" * 60)


if __name__ == "__main__":
    main()
