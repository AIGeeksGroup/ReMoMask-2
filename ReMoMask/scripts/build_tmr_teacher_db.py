"""
Build a database/-format teacher feature library using the *official* TMR
encoder, for the E08 teacher-choice ablation (tab:ablation_v2_teacher).

Investigation summary (why this is a standalone script, not a flag on
build_rag_database.py):
    Part_TMR/models/builder_bimoco.py (`MoCoTMR`) is this project's own
    from-scratch retriever: a 6-body-part MoCo-contrastive architecture with
    a CLIP-style text tower and an HBM (hierarchical body-part matching)
    loss. It is architecturally unrelated to the original TMR (Petrovich et
    al., "TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion
    Synthesis", ICCV 2023 -- a single global TEMOS-style VAE motion encoder +
    DistilBERT text encoder). Official-TMR checkpoints CANNOT be loaded into
    MoCoTMR (module names, per-part branching, and momentum-queue buffers
    all differ), so this script does not route through Part_TMR at all --
    it talks to the official TMR inference API directly via --tmr_repo.

TMR HumanML3D checkpoint source (TODO for operator -- manual, not automated
by this script):
    1. Clone the official repo:  git clone https://github.com/Mathux/TMR
    2. Install its deps (see TMR's own requirements.txt / README).
    3. Download a HumanML3D-trained run from the TMR README's "Pretrained
       models" section (uses the Guo et al. HumanML3D "guoh3dfeats" 263-dim
       representation -- the same one this repo stores in new_joint_vecs/,
       which is why per-frame features are reused as-is below).
    4. Pass the run directory (containing its Hydra config + checkpoint,
       see TMR's src/load.py:load_model_from_cfg / read_config) as
       --tmr_ckpt, and the repo clone as --tmr_repo.
    This script only *consumes* an already-downloaded checkpoint; it does
    not fetch or convert one.

Usage (real, once the TODO above is done):
    python scripts/build_tmr_teacher_db.py \
        --tmr_repo /path/to/TMR \
        --tmr_ckpt /path/to/TMR/pretrained_models/tmr_humanml3d_guoh3dfeats \
        --dataset_root dataset/HumanML3D \
        --output_dir database_tmr \
        --device cuda:0
    # then: train_query_projector.py --database_bmm_path database_tmr ...

Usage (mock encoder, no TMR install required -- exercises the exact same
dataset-iteration + output-writing code path with random unit-norm
features; this is what smoke-tests this script):
    python scripts/build_tmr_teacher_db.py --mock --mock_dim 512 \
        --dataset_root dataset/HumanML3D --split_file train_tiny.txt \
        --output_dir /tmp/database_tmr_mock

Output format (matches build_rag_database.py; consumed by
train_query_projector.py via --database_bmm_path):
    <output_dir>/motion_ids.npy        List[str]              "{name}_{caption_idx}"
    <output_dir>/all_captions.npy      List[str]
    <output_dir>/encoded_motions.npy   np.ndarray (N, 1, D)    L2-normalised
    <output_dir>/encoded_texts.npy     np.ndarray (N, 1, D)    L2-normalised
    <output_dir>/tag_lists.npy         List[[float, float]]
    <output_dir>/metadata.json         {"teacher": "tmr" | "mock", ...}

Note on positional alignment: dataset entries are filtered/iterated with the
exact same rule as build_rag_database_ze.py (drop raw motions shorter than
MIN_MOTION_LENGTH, drop captions whose cropped [f_tag, to_tag] segment is
still shorter than MIN_MOTION_LENGTH) so that, for the same split file, the
row order here lines up with database_ze/ as closely as possible. Sample
counts can still differ across database_* directories (e.g. stale rebuilds);
train_query_projector.py already truncates to min(N) across databases and
warns when it does so.
"""

import argparse
import itertools
import json
import os
import shutil
import sys
import time
import codecs as cs
from os.path import join as pjoin

import numpy as np
import torch
from tqdm import tqdm

MIN_MOTION_LENGTH = 40  # frames; original motions shorter than this are ignored
FPS = 20                # HumanML3D frame rate


# -------------------------------------------------------------------------
# Encoders: MockTMREncoder (smoke-testable, no deps) and OfficialTMREncoder
# (real TMR checkpoint, guarded import). Both expose the same tiny interface
# so the dataset-iteration / output-writing code below is identical either
# way.
# -------------------------------------------------------------------------

class MockTMREncoder:
    """Deterministic random unit-norm features, for output-format testing.

    NOT a real teacher -- only exists so this script's file-writing logic
    can be smoke-tested without cloning TMR / downloading a checkpoint.
    """

    def __init__(self, dim=512, seed=0):
        self.dim = dim
        self.rng = np.random.RandomState(seed)

    def encode_motion(self, motion, length):
        v = self.rng.randn(self.dim).astype(np.float32)
        return v

    def encode_text(self, caption):
        v = self.rng.randn(self.dim).astype(np.float32)
        return v


_TMR_IMPORT_ERROR = """\
Could not import the official TMR inference API from --tmr_repo={tmr_repo!r}.

This script expects a local clone of the official TMR repo (Petrovich et
al., ICCV 2023): https://github.com/Mathux/TMR

    git clone https://github.com/Mathux/TMR
    cd TMR && pip install -r requirements.txt   # see TMR README for exact deps
    # download a HumanML3D-trained checkpoint (TMR README "Pretrained models")

Then re-run with:
    --tmr_repo /path/to/TMR --tmr_ckpt /path/to/TMR/<run_dir>

Underlying error: {err}
"""


class OfficialTMREncoder:
    """Wraps an official-TMR checkpoint for motion/text encoding.

    Best-effort adapter against TMR's public API as of the ICCV'23 release
    (src/load.py:load_model_from_cfg, src/data/collate.py:collate_x_dict,
    model.encode(x_dict, sample_mean=True)). TMR module/config layouts have
    shifted across forks/versions -- if instantiation or encode() raises
    here, check the installed --tmr_repo's own encode_text.py / retrieval.py
    scripts for the exact call signature and adjust this class accordingly
    (kept deliberately small and isolated for that purpose).

    x_dict convention: {"x": (T, D) float tensor, "length": int}. HumanML3D
    TMR checkpoints use the Guo et al. "guoh3dfeats" 263-dim representation,
    i.e. the same mean/std-normalised vectors this repo stores under
    new_joint_vecs/ -- so motion features are passed through unchanged.
    """

    def __init__(self, tmr_repo, tmr_ckpt, device):
        tmr_repo = os.path.abspath(tmr_repo)
        if tmr_repo not in sys.path:
            sys.path.insert(0, tmr_repo)
        try:
            from src.config import read_config          # noqa: F401
            from src.load import load_model_from_cfg     # noqa: F401
            from src.data.collate import collate_x_dict  # noqa: F401
            from hydra.utils import instantiate          # noqa: F401
        except ImportError as e:
            raise SystemExit(_TMR_IMPORT_ERROR.format(tmr_repo=tmr_repo, err=e))

        self.device = device
        cfg = read_config(tmr_ckpt)
        self.model = load_model_from_cfg(cfg, device=device, eval_mode=True)
        self._collate_x_dict = collate_x_dict
        try:
            self.text_model = instantiate(cfg.data.text_to_token_emb)
        except Exception as e:
            raise SystemExit(
                "Failed to instantiate the TMR text-to-token-embedding model "
                f"from cfg.data.text_to_token_emb ({e}). Inspect the installed "
                "TMR repo's encode_text.py for the exact config key/loader in "
                "your version and adjust OfficialTMREncoder.__init__ accordingly."
            )

    def _encode_x_dict(self, items):
        x_dict = self._collate_x_dict(items)
        x_dict = {
            k: (v.to(self.device) if torch.is_tensor(v) else v)
            for k, v in x_dict.items()
        }
        with torch.inference_mode():
            latent = self.model.encode(x_dict, sample_mean=True)
        return latent[0].detach().cpu().numpy()

    def encode_motion(self, motion, length):
        item = {"x": torch.from_numpy(motion).float(), "length": int(length)}
        return self._encode_x_dict([item])

    def encode_text(self, caption):
        items = self.text_model([caption])
        return self._encode_x_dict(items)


# -------------------------------------------------------------------------
# Dataset iteration (mirrors build_rag_database_ze.py's filtering so rows
# line up with database_ze/ as closely as possible; see module docstring).
# -------------------------------------------------------------------------

def iter_caption_entries(dataset_root, split_file, max_motion_length=None):
    motion_dir = pjoin(dataset_root, "new_joint_vecs")
    text_dir = pjoin(dataset_root, "texts")
    mean = np.load(pjoin(dataset_root, "Mean.npy"))
    std = np.load(pjoin(dataset_root, "Std.npy"))

    with cs.open(split_file, encoding="utf-8", errors="replace") as f:
        names = [n.strip() for n in f.readlines() if n.strip()]

    for name in names:
        motion_path = pjoin(motion_dir, name + ".npy")
        text_path = pjoin(text_dir, name + ".txt")
        if not os.path.exists(motion_path) or not os.path.exists(text_path):
            continue

        raw_motion = np.load(motion_path)
        if len(raw_motion) < MIN_MOTION_LENGTH or np.isnan(raw_motion).any():
            continue

        with cs.open(text_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split("#")
            if len(parts) < 4:
                continue
            caption = parts[0]
            f_tag = float(parts[2]) if parts[2] else 0.0
            to_tag = float(parts[3]) if parts[3] else 0.0
            f_tag = 0.0 if np.isnan(f_tag) else f_tag
            to_tag = 0.0 if np.isnan(to_tag) else to_tag

            if f_tag == 0.0 and to_tag == 0.0:
                motion = raw_motion
            else:
                motion = raw_motion[int(f_tag * FPS): int(to_tag * FPS)]

            if len(motion) < MIN_MOTION_LENGTH:
                continue

            motion_norm = ((motion - mean) / std).astype(np.float32)
            if max_motion_length and len(motion_norm) > max_motion_length:
                motion_norm = motion_norm[:max_motion_length]

            yield name, idx, caption, motion_norm, f_tag, to_tag


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a database/-format TMR teacher feature library (E08 teacher ablation).")
    parser.add_argument("--tmr_repo", type=str, default=None,
                        help="Path to a local clone of the official TMR repo (Mathux/TMR). "
                             "Required unless --mock is set.")
    parser.add_argument("--tmr_ckpt", type=str, default=None,
                        help="Path to a TMR run_dir (Hydra config + checkpoint) for a "
                             "HumanML3D-trained model. Required unless --mock is set.")
    parser.add_argument("--mock", action="store_true",
                        help="Use MockTMREncoder (random unit-norm features) instead of "
                             "loading a real TMR checkpoint. For smoke-testing the output "
                             "format only -- never use for real experiments.")
    parser.add_argument("--mock_dim", type=int, default=512,
                        help="Feature dim for --mock.")
    parser.add_argument("--mock_seed", type=int, default=0,
                        help="RNG seed for --mock.")
    parser.add_argument("--dataset_root", type=str, default="dataset/HumanML3D",
                        help="HumanML3D data root (must contain new_joint_vecs/, texts/, "
                             "Mean.npy, Std.npy).")
    parser.add_argument("--split_file", type=str, default="train.txt",
                        help="Split file name, resolved relative to --dataset_root.")
    parser.add_argument("--max_motion_length", type=int, default=196,
                        help="Optional truncation (frames) for very long motions; TMR "
                             "handles variable-length input natively so this is only a "
                             "safety cap, not padding (unlike build_rag_database.py).")
    parser.add_argument("--max_captions", type=int, default=None,
                        help="Optional cap on number of (motion, caption) pairs processed "
                             "(debug / smoke-test only).")
    parser.add_argument("--output_dir", type=str, default="database_tmr",
                        help="Output directory for the database files.")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    if args.mock:
        encoder = MockTMREncoder(dim=args.mock_dim, seed=args.mock_seed)
        teacher_name = "mock"
    else:
        if not args.tmr_repo or not args.tmr_ckpt:
            raise SystemExit(
                "--tmr_repo and --tmr_ckpt are required unless --mock is set. "
                "See the script header docstring for setup instructions."
            )
        encoder = OfficialTMREncoder(args.tmr_repo, args.tmr_ckpt, args.device)
        teacher_name = "tmr"

    split_file = pjoin(args.dataset_root, args.split_file)
    print(f"Reading split file: {split_file}")

    entries = iter_caption_entries(args.dataset_root, split_file, args.max_motion_length)
    if args.max_captions is not None:
        entries = itertools.islice(entries, args.max_captions)

    motion_embeddings, text_embeddings = [], []
    caption_list, motion_ids, tag_lists = [], [], []

    for name, idx, caption, motion_norm, f_tag, to_tag in tqdm(entries, desc="Encoding (TMR teacher)"):
        motion_vec = encoder.encode_motion(motion_norm, len(motion_norm))
        motion_vec = motion_vec / (np.linalg.norm(motion_vec) + 1e-8)

        text_vec = encoder.encode_text(caption)
        text_vec = text_vec / (np.linalg.norm(text_vec) + 1e-8)

        motion_embeddings.append(motion_vec)
        text_embeddings.append(text_vec)
        caption_list.append(caption)
        motion_ids.append(f"{name}_{idx}")
        tag_lists.append([f_tag, to_tag])

    total = len(caption_list)
    if total == 0:
        raise SystemExit(
            f"No (motion, caption) pairs encoded from {split_file} -- check "
            "--dataset_root/--split_file."
        )

    motion_embeddings = np.stack(motion_embeddings, axis=0)[:, np.newaxis, :]  # (N, 1, D)
    text_embeddings = np.stack(text_embeddings, axis=0)[:, np.newaxis, :]      # (N, 1, D)
    D = motion_embeddings.shape[-1]
    print(f"Total samples: {total}")
    print(f"encoded_motions: {motion_embeddings.shape}, encoded_texts: {text_embeddings.shape}")

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(pjoin(args.output_dir, "all_captions.npy"), caption_list)
    np.save(pjoin(args.output_dir, "motion_ids.npy"), motion_ids)
    np.save(pjoin(args.output_dir, "tag_lists.npy"), tag_lists)
    np.save(pjoin(args.output_dir, "encoded_motions.npy"), motion_embeddings)
    np.save(pjoin(args.output_dir, "encoded_texts.npy"), text_embeddings)

    # Optional: carry over motion_tokens.npy (generation-time token lookup)
    # from the existing BMM database/, if present. Not required by
    # train_query_projector.py, but kept for parity with build_rag_database.py
    # / build_rag_database_ze.py's output directories.
    existing_tokens = pjoin("database", "motion_tokens.npy")
    if os.path.exists(existing_tokens):
        shutil.copy2(existing_tokens, pjoin(args.output_dir, "motion_tokens.npy"))
        print(f"Copied motion_tokens.npy from {existing_tokens}")

    metadata = {
        "teacher": teacher_name,
        "encoder_dim": int(D),
        "tmr_repo": args.tmr_repo,
        "tmr_ckpt": args.tmr_ckpt,
        "dataset_root": args.dataset_root,
        "split_file": args.split_file,
        "max_motion_length": args.max_motion_length,
        "num_samples": total,
        "motion_embedding_shape": list(motion_embeddings.shape),
        "text_embedding_shape": list(text_embeddings.shape),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": (
            "MOCK encoder: random unit-norm features for output-format "
            "smoke-testing only -- NOT a real teacher, do not use for E08."
            if args.mock else
            "Official TMR (Mathux/TMR) encoder output. Verify --tmr_ckpt is "
            "a HumanML3D/guoh3dfeats run before using as the E08 teacher "
            "(--database_bmm_path database_tmr)."
        ),
    }
    with open(pjoin(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    duration = time.time() - t_start
    print(f"Done in {duration:.2f}s ({duration / 60:.2f} min). "
          f"{total} pairs -> {args.output_dir}/ (teacher={teacher_name}, dim={D})")


if __name__ == "__main__":
    main()
