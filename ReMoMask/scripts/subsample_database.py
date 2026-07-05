"""
Coverage-vs-quality database sub-sampling (E06).
=================================================
Builds nested, seeded sub-samples of a retrieval database (e.g. database_ze/)
by stratifying on the *unique motion* (not the per-caption entry): for a
fraction f we keep floor(f * n_unique_motions) motions and ALL of their
caption entries. Fractions share one seeded permutation of motions, so
smaller fractions are always subsets of larger ones (monotone coverage
curve, cf. fig:db_coverage).

A database directory looks like (see build_rag_database_ze.py):
    encoded_motions.npy      (N, 1, D)   float, aligned per entry
    encoded_texts.npy        (N, 1, D)   float, aligned per entry (optional)
    encoded_texts_clip.npy   (N, 1, 512) float, aligned per entry
    all_captions.npy         (N,)        str,   aligned per entry
    motion_ids.npy           (N,)        str,   aligned per entry, "{motion}_{caption_idx}"
    tag_lists.npy            (N, 2)      float, aligned per entry
    motion_tokens.npy        dict[str, np.ndarray], keyed by the same id as motion_ids
    metadata.json

Usage:
    cd ReMoMask
    python scripts/subsample_database.py \
        --database database_ze \
        --fractions 0.1,0.25,0.5,0.75 \
        --seed 3407
    # -> writes database_ze_p10/, database_ze_p25/, database_ze_p50/, database_ze_p75/
"""

import argparse
import json
import os
import re
import time
from os.path import join as pjoin

import numpy as np

# Per-entry arrays that must be subset consistently (row 0 = database entry).
ALIGNED_FILES = [
    "encoded_motions.npy",
    "encoded_texts.npy",
    "encoded_texts_clip.npy",
    "all_captions.npy",
    "motion_ids.npy",
    "tag_lists.npy",
]

BASE_ID_RE = re.compile(r"_(\d+)$")


def base_motion_id(entry_id):
    """Strip the trailing '_<caption_idx>' suffix, e.g. '000002_1' -> '000002'."""
    return BASE_ID_RE.sub("", entry_id)


def format_suffix(fraction, suffix_style):
    if suffix_style == "pNN":
        pct = int(round(fraction * 100))
        return "p{:02d}".format(pct)
    raise ValueError("Unsupported --suffix_style '{}' (supported: pNN)".format(suffix_style))


def parse_fractions(s):
    fractions = [float(x) for x in s.split(",") if x.strip() != ""]
    for f in fractions:
        if not (0.0 < f <= 1.0):
            raise ValueError("Each fraction must be in (0, 1], got {}".format(f))
    return fractions


def load_metadata(database_dir):
    meta_path = pjoin(database_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Stratified, nested sub-sampling of a retrieval database "
                     "by unique motion (E06 coverage-vs-quality).")
    parser.add_argument("--database", type=str, default="database_ze",
                        help="Input database directory (e.g. database_ze).")
    parser.add_argument("--fractions", type=str, default="0.1,0.25,0.5,0.75",
                        help="Comma-separated fractions in (0,1], e.g. '0.1,0.25,0.5,0.75'.")
    parser.add_argument("--seed", type=int, default=3407,
                        help="Seed for the motion permutation (shared across fractions "
                             "so smaller fractions nest inside larger ones).")
    parser.add_argument("--suffix_style", type=str, default="pNN",
                        help="Output directory suffix style. Currently only 'pNN' "
                             "(e.g. database_ze_p10) is supported.")
    parser.add_argument("--out_root", type=str, default=".",
                        help="Root directory under which '<database>_<suffix>' output "
                             "dirs are created. Defaults to the current directory "
                             "(repo root, when invoked from there). Override for smoke "
                             "tests so the real repo is not touched.")
    args = parser.parse_args()

    fractions = parse_fractions(args.fractions)

    database_dir = args.database
    motion_ids = np.load(pjoin(database_dir, "motion_ids.npy"), allow_pickle=True)
    n_total = len(motion_ids)
    base_ids = np.array([base_motion_id(str(m)) for m in motion_ids])
    unique_bases = np.unique(base_ids)
    n_unique = len(unique_bases)
    print("Database '{}': {} entries, {} unique motions".format(database_dir, n_total, n_unique))

    # Single seeded permutation shared by all fractions -> nested subsets.
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n_unique)

    metadata_in = load_metadata(database_dir)
    db_basename = os.path.basename(os.path.normpath(database_dir))

    prev_selected = None
    prev_count = -1
    for fraction in sorted(fractions):
        count = int(np.floor(fraction * n_unique))
        if count <= 0:
            print("  WARNING: fraction {} selects 0 motions, clamping to 1".format(fraction))
            count = 1
        selected_bases = set(unique_bases[perm[:count]].tolist())

        # Nesting sanity check: this fraction's selection must contain the
        # previous (smaller) fraction's selection, by construction of `perm`.
        if prev_selected is not None and count >= prev_count:
            assert prev_selected <= selected_bases, \
                "nesting violated between fractions (seed/perm bug)"
        prev_selected, prev_count = selected_bases, count

        mask = np.array([b in selected_bases for b in base_ids])
        n_entries_selected = int(mask.sum())

        suffix = format_suffix(fraction, args.suffix_style)
        out_dir = pjoin(args.out_root, "{}_{}".format(db_basename, suffix))
        os.makedirs(out_dir, exist_ok=True)

        print("fraction={:.4f} -> {} motions, {} entries -> {}".format(
            fraction, count, n_entries_selected, out_dir))

        for fname in ALIGNED_FILES:
            fpath = pjoin(database_dir, fname)
            if not os.path.exists(fpath):
                continue
            arr = np.load(fpath, allow_pickle=True)
            if arr.shape[0] != n_total:
                print("  WARNING: {} has {} rows (expected {}), skipping "
                      "(stale/mismatched file, not written to subset)".format(
                          fname, arr.shape[0], n_total))
                continue
            np.save(pjoin(out_dir, fname), arr[mask])

        tokens_path = pjoin(database_dir, "motion_tokens.npy")
        if os.path.exists(tokens_path):
            tokens = np.load(tokens_path, allow_pickle=True).item()
            if isinstance(tokens, dict):
                selected_entry_ids = set(motion_ids[mask].tolist())
                tokens_out = {k: v for k, v in tokens.items() if k in selected_entry_ids}
                np.save(pjoin(out_dir, "motion_tokens.npy"), tokens_out)
                print("  motion_tokens.npy: kept {}/{} entries".format(
                    len(tokens_out), len(tokens)))
            else:
                print("  WARNING: motion_tokens.npy is not a dict "
                      "(type={}), copied as-is".format(type(tokens)))
                np.save(pjoin(out_dir, "motion_tokens.npy"), tokens)

        metadata_out = dict(metadata_in)
        metadata_out.update({
            "parent": os.path.abspath(database_dir),
            "subsample_fraction": fraction,
            "subsample_seed": args.seed,
            "subsample_suffix_style": args.suffix_style,
            "n_unique_motions_total": int(n_unique),
            "n_unique_motions_selected": int(count),
            "n_entries_total": int(n_total),
            "n_entries_selected": n_entries_selected,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        with open(pjoin(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata_out, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()
