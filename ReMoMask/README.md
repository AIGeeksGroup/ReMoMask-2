# ReMoMask-2 — Working Repository (TPAMI Extension)

Working codebase for **ReMoMask-2: Latent Retrieval-Augmented Masked Motion Generation** (TPAMI extension, in progress), built on the conference version:

> **ReMoMask: Retrieval-Augmented Masked Motion Generation**
> Zhengdao Li\*, Siheng Wang\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, and [Hao Tang](https://ha0tang.github.io/)<sup>#</sup>
> [Paper](https://arxiv.org/abs/2508.02605) | [Website](https://aigeeksgroup.github.io/ReMoMask) | [V1 Model](https://huggingface.co/lycnight/ReMoMask) | [HF Paper](https://huggingface.co/papers/2508.02605)

ReMoMask-2 migrates the retrieval space from the standalone contrastive semantic space (Part_TMR/HBM) into the generator's own pre-quantization RVQ-VAE latent space $z_e$, via a lightweight query projector distilled from the HBM retriever. The V2 pipeline is fully gated behind `--use_ze_retrieval`; the V1 path is untouched.

---

## 1. Environment

```bash
conda create -n remomask python=3.10 -y
conda activate remomask
# CUDA 11.8 build used throughout the project
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Verified on: L40 / A40 / A800 (training & eval). Note: `models/vq/quantizer.py` hard-codes `.cuda()` — **a GPU is required even to construct the RVQ-VAE**; CPU-only machines can only run mock-level tests.

## 2. Prepare assets

All downloads work through the HF mirror. Set the endpoint first:

```bash
# Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com
```
```powershell
# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

### 2.1 V1 official bundle (base weights)

```bash
hf download lycnight/ReMoMask --local-dir hf_download
```

Place into the repo:

| From bundle | To | Contents |
|---|---|---|
| `hf_download/checkpoints/` | `./checkpoints/` | evaluators (`Comp_v6_KLD005`, `text_mot_match`, `length_estimator`), glove, kit |
| `hf_download/logs/` | `./logs/` | `pretrain_vq` (2D-RVQ-VAE), `pretrain_mtrans`, `pretrain_rtrans` |
| `hf_download/Part_TMR/` | `./Part_TMR/` | HBM/BMM retriever checkpoint (`checkpoints/exp_for_mtrans/HumanML3D/best_model.pt`) |
| `hf_download/ViT-B-32.pt` | `./ViT-B-32.pt` | CLIP weights |

> ⚠️ **Do NOT use `hf_download/database/` — it is a 32-entry debug stub** (shape `(32,1,512)`, a `config_small` leftover in the official bundle). Using it silently cripples retrieval. Use the full rebuilt database from the V2 asset repo below, or rebuild it yourself (Sec. 3.0).

### 2.2 V2 assets (retrained checkpoints, full databases, projector)

```bash
hf download yiranranranra/my_dataset --repo-type dataset --include "remomask2/*" --local-dir hf_v2
```

| From `hf_v2/remomask2/` | To | Contents |
|---|---|---|
| `database/` | `./database/` | **full** BMM retrieval database (66,912 entries, rebuilt from the complete train split) |
| `database_ze/` | `./database_ze/` | $z_e$ retrieval database (66,912 × 1024-d, L2-normalized) |
| `query_projector/` | `./logs/query_projector/` | trained query projector (`best_projector.pt`, CLIP 512d → $z_e$ 1024d) |
| `checkpoints/humanml3d/v1_retrain_rtval/` | `./logs/humanml3d/v1_retrain_rtval/model/` | controlled-baseline Mask Transformer (`net_best_fid_ep0316.tar` + `opt.txt`) |
| `checkpoints/humanml3d/v2_ze_rtval/` | `./logs/humanml3d/v2_ze_rtval/model/` | ReMoMask-2 Mask Transformer (`net_best_fid_ep0409.tar` + `opt.txt`) |

### 2.3 Dataset (HumanML3D)

Expected at `./dataset/HumanML3D/` (standard T2M layout; feature dim 263; the train split used for databases is the mirror-augmented `train.txt`, 23,384 sequences).

Recommended: build it from the official pipeline [EricGuo5513/HumanML3D](https://github.com/EricGuo5513/HumanML3D) (AMASS license terms apply — AMASS data must be obtained from the source and may not be redistributed).

### 2.4 Expected layout after preparation

```
ReMoMask/
├── checkpoints/{glove, humanml3d/{Comp_v6_KLD005,text_mot_match,length_estimator}, kit}
├── logs/humanml3d/{pretrain_vq, pretrain_mtrans, pretrain_rtrans, v1_retrain_rtval, v2_ze_rtval}
├── logs/query_projector/best_projector.pt
├── database/          # full BMM database (66,912) — NOT the 32-entry stub
├── database_ze/       # z_e database (66,912 × 1024d)
├── dataset/HumanML3D/
├── Part_TMR/checkpoints/exp_for_mtrans/HumanML3D/best_model.pt
└── ViT-B-32.pt
```

## 3. ReMoMask-2 pipeline (from scratch)

Skip any step whose artifact you downloaded in Sec. 2.2.

**3.0 (Re)build the BMM database** (only if not using the downloaded one):

```bash
python build_rag_database.py            # full train.txt -> database/ (66,912 entries)
```

**3.1 Build the $z_e$ database** (frozen 2D-RVQ-VAE encoder, pre-quantization latents, spatio-temporal mean-pool + L2 norm):

```bash
python build_rag_database_ze.py --vq_name pretrain_vq --output_dir database_ze
# variant for the z_q ablation: add --use_quantized --output_dir database_zq
```

**3.2 Train the query projector** (KL distillation from the BMM teacher, top-256 soft targets, τ=0.07):

```bash
python train_query_projector.py \
    --database_ze_path database_ze --database_bmm_path database \
    --output_dir logs/query_projector \
    --epochs 200 --batch_size 128 --lr 1e-4 --temperature 0.07 \
    --teacher_topk 256 --eval_interval 10 --device cuda:0
# ablation flags: --objective {kl,infonce,mse}   --hidden {0,1024,2048}   --teacher_topk {64,256,1024}
```

**3.3 Train the Mask Transformer (V2)** — the controlled protocol keeps every hyper-parameter identical to the V1 `opt.txt` (latent_dim 512, 8 heads, 8 layers, ff 1024, batch 64, lr 2e-4 constant, seed 3407); the retrieval space is the only change:

```bash
MASTER_PORT=12584 python -m torch.distributed.launch --nproc_per_node=<N> \
  train_mask_transformer_ddp.py \
  --name v2_ze_rtval --dataset_name humanml3d --vq_name pretrain_vq \
  --use_ze_retrieval --ze_database_path database_ze \
  --projector_path logs/query_projector/best_projector.pt \
  --rt_in_value --batch_size 64 --max_epoch 2000
# V1 controlled baseline: same command without the three ze flags (keep --rt_in_value)
```

Always pass hyper-parameters explicitly — several code defaults (e.g. `latent_dim=384`) differ from the paper configuration.

**3.4 Evaluation** (paper protocol: `cond_scale=4, time_steps=10, seed=10107`, 20 repeats):

```bash
# full pipeline (paper numbers): Mask + Residual transformer
python eval_res.py \
    --mtrans_name v2_ze_rtval --which_epoch net_best_fid_ep0409 \
    --rtrans_name pretrain_rtrans --which_ckpt net_best_fid.tar \
    --dataset_name humanml3d --vq_name pretrain_vq --gpu_id 0 \
    --repeat_times 20 --cond_scale 4 --time_steps 10 --seed 10107 \
    --use_ze_retrieval --ze_database_path database_ze \
    --projector_path logs/query_projector/best_projector.pt \
    --rt_in_value --retrieval_dim 1024 --ext my_eval
# mask-only diagnosis: eval_mask.py with the same flags (drop --rtrans_name/--which_ckpt)
```

**Checkpoint-selection traps (read before trusting any number):**
- `--which_epoch` substring-matches files in the model dir. After a resumed run, the un-suffixed `net_best_fid.tar` is overwritten by a sub-optimal tracker reset — **always pass the epoch-suffixed name** (`net_best_fid_ep0409`), never `best_fid`.
- `--which_ckpt` (residual transformer) defaults to `net_best_fid2d.tar`, which does not exist for `pretrain_rtrans` — pass `net_best_fid.tar` explicitly.
- `eval_res.py` runs 20 repeats regardless of `--repeat_times` (hard-coded); `eval_mask.py` honors the flag.
- The residual transformer is retrieval-free (no `re_dict` input) — sharing the frozen `pretrain_rtrans` across V1/V2 is intentional; it is the controlled protocol.

## 4. Experiment suite (TPAMI Phase 4)

Ready-to-submit SLURM jobs for the journal experiments live in `scripts/slurm/phase4/` (E01 full eval, E02 mask-only, E03 cross-space rank correlation, E04 z_q database, E08 κ sensitivity). Analysis utilities: `scripts/analyze_space_gap.py` (representation-gap quantification), `scripts/subsample_database.py` (coverage ablation), `scripts/build_tmr_teacher_db.py` (teacher ablation).

Transferring scripts from Windows to Linux? Strip CRLF first: `sed -i 's/\r$//' <file>`.

## Citation

```bibtex
@article{li2025remomask,
  title={ReMoMask: Retrieval-Augmented Masked Motion Generation},
  author={Li, Zhengdao and Wang, Siheng and Zhang, Zeyu and Tang, Hao},
  journal={arXiv preprint arXiv:2508.02605},
  year={2025}
}
```

ReMoMask-2 (TPAMI extension) citation: TBA.
