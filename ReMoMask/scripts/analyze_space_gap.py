"""
Cross-Space Rank Correlation Analysis (E03: representation-gap quantification)
===============================================================================
Read-only analysis script: compares the neighbor structure induced by the BMM
retrieval space (database/encoded_motions, 512-d) against the z_e retrieval
space (database_ze/encoded_motions, 1024-d) on the SAME set of unique motions.

For each query motion we compute cosine-similarity rankings of all other
motions in both spaces, then measure:
  (a) Spearman rank correlation between the two similarity vectors, and
  (b) top-k neighbor overlap |A ∩ B| / k for k in --topk.

Low correlation / overlap is the empirical evidence for the "representation
gap" between the two spaces that motivates the alignment (query projector).

Usage:
    cd ReMoMask
    python scripts/analyze_space_gap.py \
        --database_bmm database --database_ze database_ze \
        --sample 2000 --seed 3407 --topk 1 5 10 50 \
        --out results/space_gap
"""

import argparse
import json
import os
import re
import sys
from os.path import join as pjoin

import numpy as np
from scipy.stats import spearmanr

# ---- project imports (run from ReMoMask/) --------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Matches HumanML3D-style motion ids used in the RAG databases, e.g.
# "000002_0" (motion 000002, caption index 0) or "M000002_3" (mirrored motion).
MOTION_ID_RE = re.compile(r"^(M?\d+)_(\d+)$")


def split_motion_id(full_id):
    """Split a full database row id into (base_motion_id, caption_index).

    Falls back to treating the whole string as the base id (caption_index=0)
    if it does not match the expected "<id>_<caption_idx>" pattern.
    """
    m = MOTION_ID_RE.match(full_id)
    if m is None:
        return full_id, 0
    return m.group(1), int(m.group(2))


def load_database(db_dir):
    """Load motion_ids + encoded_motions from a RAG database directory."""
    ids_path = pjoin(db_dir, "motion_ids.npy")
    vec_path = pjoin(db_dir, "encoded_motions.npy")
    assert os.path.isfile(ids_path), f"motion_ids.npy not found in {db_dir}"
    assert os.path.isfile(vec_path), f"encoded_motions.npy not found in {db_dir}"

    motion_ids = np.load(ids_path, allow_pickle=True)
    motion_ids = np.array([str(x) for x in motion_ids])

    vecs = np.load(vec_path, allow_pickle=True)
    vecs = np.asarray(vecs, dtype=np.float32).reshape(len(motion_ids), -1)
    return motion_ids, vecs


def l2_normalize(x, eps=1e-8):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def build_unique_motion_index(motion_ids):
    """Group row indices by base motion id, keeping the first-seen caption
    (lowest caption index, ties broken by array order) as the representative
    row for that unique motion.

    Returns: dict base_motion_id -> (row_index, full_id)
    """
    best = {}  # base_id -> (caption_idx, row_index, full_id)
    for row_idx, full_id in enumerate(motion_ids):
        base_id, cap_idx = split_motion_id(full_id)
        cur = best.get(base_id)
        if cur is None or cap_idx < cur[0]:
            best[base_id] = (cap_idx, row_idx, full_id)
    return {base_id: (row_idx, full_id) for base_id, (_, row_idx, full_id) in best.items()}


def main():
    parser = argparse.ArgumentParser(
        description="E03: cross-space (BMM vs z_e) rank-correlation analysis")
    parser.add_argument("--database_bmm", type=str, default="database",
                        help="BMM (Part_TMR) RAG database directory (512-d encoded_motions)")
    parser.add_argument("--database_ze", type=str, default="database_ze",
                        help="z_e RAG database directory (1024-d encoded_motions)")
    parser.add_argument("--sample", type=int, default=2000,
                        help="Number of unique motions to sample")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10, 50],
                        help="k values for neighbor-overlap computation")
    parser.add_argument("--out", type=str, default="results/space_gap",
                        help="Output directory for summary.json + plots")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # 1. Load both databases
    # ------------------------------------------------------------------
    print(f"Loading BMM database from {args.database_bmm} ...")
    bmm_ids, bmm_vecs = load_database(args.database_bmm)
    print(f"  {len(bmm_ids)} rows, dim={bmm_vecs.shape[1]}")

    print(f"Loading z_e database from {args.database_ze} ...")
    ze_ids, ze_vecs = load_database(args.database_ze)
    print(f"  {len(ze_ids)} rows, dim={ze_vecs.shape[1]}")

    # ------------------------------------------------------------------
    # 2. Verify row order / align by motion_id
    # ------------------------------------------------------------------
    same_order = (len(bmm_ids) == len(ze_ids)) and bool(np.array_equal(bmm_ids, ze_ids))
    if same_order:
        print("motion_ids arrays are identical (same order) across both databases; "
              "no alignment needed.")
        common_full_ids = list(bmm_ids)
        bmm_row_of = {fid: i for i, fid in enumerate(bmm_ids)}
        ze_row_of = {fid: i for i, fid in enumerate(ze_ids)}
    else:
        print("motion_ids arrays differ (order and/or content); aligning by motion_id "
              "(+ caption index) ...")
        bmm_row_of = {fid: i for i, fid in enumerate(bmm_ids)}
        ze_row_of = {fid: i for i, fid in enumerate(ze_ids)}
        common_full_ids = sorted(set(bmm_row_of.keys()) & set(ze_row_of.keys()))
        print(f"  {len(common_full_ids)} full (motion, caption) ids in common "
              f"out of {len(bmm_ids)} (BMM) / {len(ze_ids)} (z_e)")
        assert len(common_full_ids) > 0, (
            "No overlapping motion_ids between the two databases -- cannot align.")

    # ------------------------------------------------------------------
    # 3. Deduplicate to one row per unique (base) motion id
    # ------------------------------------------------------------------
    common_ids_set = set(common_full_ids)
    unique_index = {}  # base_id -> (caption_idx, full_id)
    for full_id in common_full_ids:
        base_id, cap_idx = split_motion_id(full_id)
        cur = unique_index.get(base_id)
        if cur is None or cap_idx < cur[0]:
            unique_index[base_id] = (cap_idx, full_id)

    unique_base_ids = sorted(unique_index.keys())
    n_unique = len(unique_base_ids)
    print(f"Unique motions after dedup: {n_unique} "
          f"(from {len(common_ids_set)} aligned motion-caption rows)")

    # ------------------------------------------------------------------
    # 4. Sample N unique motions (seeded)
    # ------------------------------------------------------------------
    n_sample = args.sample
    if n_sample > n_unique:
        print(f"WARNING: --sample {n_sample} > available unique motions ({n_unique}); "
              f"clipping to {n_unique}.")
        n_sample = n_unique
    assert n_sample >= 2, "Need at least 2 unique motions to compute rank correlation."

    sampled_bases = rng.choice(unique_base_ids, size=n_sample, replace=False)

    bmm_rows = []
    ze_rows = []
    for base_id in sampled_bases:
        _, full_id = unique_index[base_id]
        bmm_rows.append(bmm_row_of[full_id])
        ze_rows.append(ze_row_of[full_id])

    bmm_sub = bmm_vecs[bmm_rows]   # (N, 512)
    ze_sub = ze_vecs[ze_rows]      # (N, 1024)
    N = n_sample
    print(f"Sampled N={N} unique motions "
          f"(BMM dim={bmm_sub.shape[1]}, z_e dim={ze_sub.shape[1]})")

    # ------------------------------------------------------------------
    # 5. L2-normalize (defensive) + cosine similarity matrices
    # ------------------------------------------------------------------
    bmm_sub = l2_normalize(bmm_sub)
    ze_sub = l2_normalize(ze_sub)

    sim_bmm = bmm_sub @ bmm_sub.T   # (N, N)
    sim_ze = ze_sub @ ze_sub.T      # (N, N)

    # ------------------------------------------------------------------
    # 6. Per-query Spearman rho + top-k overlap (excluding self)
    # ------------------------------------------------------------------
    topk_list = sorted(set(args.topk))
    max_k = N - 1
    clipped_topk = []
    for k in topk_list:
        if k > max_k:
            print(f"WARNING: topk={k} exceeds available neighbors ({max_k}); clipping to {max_k}.")
            k = max_k
        clipped_topk.append(k)
    topk_list = sorted(set(clipped_topk))

    self_mask = ~np.eye(N, dtype=bool)
    rhos = np.empty(N, dtype=np.float64)
    overlaps = {k: np.empty(N, dtype=np.float64) for k in topk_list}

    for i in range(N):
        row_mask = self_mask[i]
        a = sim_bmm[i, row_mask]
        b = sim_ze[i, row_mask]
        rho, _ = spearmanr(a, b)
        rhos[i] = rho

        # candidate indices (excluding self), mapped back into full index space
        cand_idx = np.nonzero(row_mask)[0]
        order_a = cand_idx[np.argsort(-a)]
        order_b = cand_idx[np.argsort(-b)]
        for k in topk_list:
            set_a = set(order_a[:k].tolist())
            set_b = set(order_b[:k].tolist())
            overlaps[k][i] = len(set_a & set_b) / k

    # ------------------------------------------------------------------
    # 7. Aggregate + save
    # ------------------------------------------------------------------
    rho_mean = float(np.nanmean(rhos))
    rho_std = float(np.nanstd(rhos))
    rho_median = float(np.nanmedian(rhos))
    n_nan_rho = int(np.isnan(rhos).sum())

    overlap_mean = {int(k): float(np.mean(overlaps[k])) for k in topk_list}
    overlap_std = {int(k): float(np.std(overlaps[k])) for k in topk_list}

    summary = {
        "database_bmm": args.database_bmm,
        "database_ze": args.database_ze,
        "same_order": same_order,
        "n_unique_available": n_unique,
        "n_sample": N,
        "seed": args.seed,
        "topk": topk_list,
        "spearman_rho": {
            "mean": round(rho_mean, 6),
            "std": round(rho_std, 6),
            "median": round(rho_median, 6),
            "n_nan": n_nan_rho,
        },
        "topk_overlap": {
            "mean": {str(k): round(v, 6) for k, v in overlap_mean.items()},
            "std": {str(k): round(overlap_std[k], 6) for k in topk_list},
        },
    }

    summary_path = pjoin(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary -> {summary_path}")

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # overlap vs k
        fig, ax = plt.subplots(figsize=(7, 5))
        ks = topk_list
        means = [overlap_mean[k] for k in ks]
        stds = [overlap_std[k] for k in ks]
        ax.errorbar(ks, means, yerr=stds, marker="o", color="darkorange",
                    ecolor="lightgray", capsize=3, linewidth=1.5)
        ax.set_xlabel("k", fontsize=12)
        ax.set_ylabel("Top-k neighbor overlap (BMM vs z_e)", fontsize=12)
        ax.set_title(f"Cross-space top-k overlap (N={N})", fontsize=13)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        overlap_path = pjoin(args.out, "overlap_curve.png")
        fig.savefig(overlap_path, dpi=150)
        plt.close(fig)
        print(f"Saved overlap curve -> {overlap_path}")

        # rho histogram
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(rhos[~np.isnan(rhos)], bins=50, density=True, alpha=0.75,
                color="steelblue", edgecolor="white", linewidth=0.3)
        ax.axvline(rho_mean, color="red", linestyle="--", linewidth=1.5,
                   label=f"mean={rho_mean:.4f}")
        ax.set_xlabel("Per-query Spearman rho (BMM vs z_e)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Cross-space rank correlation (N={N})", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        rho_path = pjoin(args.out, "rho_hist.png")
        fig.savefig(rho_path, dpi=150)
        plt.close(fig)
        print(f"Saved rho histogram -> {rho_path}")

    except ImportError:
        print("matplotlib not available; skipping plots.")

    # ------------------------------------------------------------------
    # 9. Human-readable summary (5 lines)
    # ------------------------------------------------------------------
    overlap_str = ", ".join(f"@{k}={overlap_mean[k]:.3f}" for k in topk_list)
    print("\n" + "=" * 60)
    print("  CROSS-SPACE REPRESENTATION GAP (E03)")
    print("=" * 60)
    print(f"  N={N} unique motions sampled (seed={args.seed}) from {n_unique} available")
    print(f"  Spearman rho: mean={rho_mean:.4f} std={rho_std:.4f} median={rho_median:.4f}")
    print(f"  Top-k overlap: {overlap_str}")
    print(f"  Low rho / overlap => empirical evidence of a BMM <-> z_e representation gap")
    print("=" * 60)


if __name__ == "__main__":
    main()
