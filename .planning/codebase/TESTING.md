# Testing Patterns

**Analysis Date:** 2026-06-27

## Test Framework

**Current Status:** No traditional unit testing framework

- **Framework:** Not pytest, unittest, or similar
- **Testing approach:** End-to-end evaluation via shell scripts and Python scripts
- **Validation:** Embedded in training pipeline (epoch-based validation)
- **No formal test suite:** Codebase relies on evaluation metrics computed during inference

## Test Files & Organization

**Test File Location:**
- `Part_TMR/scripts/test.py` — Only dedicated test file (evaluates retriever model)
- Not co-located with source; separate `scripts/` directory

**File Structure:**
```
Part_TMR/
├── scripts/
│   ├── test.py          # Retriever model evaluation
│   ├── train.py         # Model training
│   └── utils.py         # Helper functions
├── models/
├── datasets/
└── conf/
    └── config.yaml      # Hydra config for test/train
```

## Test Structure & Patterns

### Part_TMR Retriever Test (`Part_TMR/scripts/test.py`)

**Pattern:**
```python
@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg: DictConfig) -> None:
    test_dataloader = prepare_test_dataset(cfg)
    model = prepare_test_model(cfg)
    eval(cfg, test_dataloader, model)
```

**Stages:**
1. **Setup:** Load config via Hydra, set random seed
2. **Data Loading:** Prepare test dataset with DataLoader (batch_size from config)
3. **Model Preparation:** Instantiate MoCoTMR, load checkpoint
4. **Evaluation:** Run inference on test split, compute metrics

**Key Functions:**
- `prepare_test_dataset()` - loads Mean/Std normalization, constructs DataLoader
- `prepare_test_model()` - initializes model, loads best_model.pt or last_model.pt
- `eval()` - executes inference loop (not shown in sampled code)

## Training & Validation Pipeline

### Shell Scripts

**Training Commands:**
- `run_mtrans.sh` - Train 2D Mask Transformer (DDP)
- `run_rtrans.sh` - Train 2D Residual Transformer (DDP)
- `run_rvq.sh` - Train RVQ-VAE quantizer

**Pattern in `run_mtrans.sh`:**
```bash
CUDA_VISIBLE_DEVICES=$GPU_IDX \
python -m torch.distributed.launch \
    --master_port $MASTER_PORT \
    --nproc_per_node=$GPU \
    train_mask_transformer_ddp.py \
    --name $NAME \
    --dataset_name $DATASET \
    ${@:6}  # Additional hyperparameters
```

**Validation During Training:**
- Frequency controlled by `--eval_every_e` (epoch-based)
- Called within training loop in `train_mask_transformer_ddp.py`
- Generates visualizations (`.mp4` videos via `plot_t2m()`)

### Trainer Validation Pattern

**File:** `models/transformer/transformer_trainer_ddp.py`

**Methods:**
- `forward()` - Encode motion, compute loss via both transformers
- `update()` - Zero grad, backward, step, return metrics
- `validation()` - Run on validation split, compute evaluation metrics

**Logging:**
- TensorBoard via `SummaryWriter` (rank 0 only in DDP)
- Loss/accuracy tracked per iteration
- Checkpoint saved based on best metric (`net_best_fid.tar`)

## Evaluation Scripts

### 1. RVQ-VAE Evaluation (`eval_vq.py`)

**Purpose:** Assess VQ quantizer reconstruction quality

**Usage:**
```bash
python eval_vq.py \
    --gpu_id 0 \
    --name pretrain_vq \
    --dataset_name humanml3d \
    --ext eval \
    --which_epoch net_best_fid.tar
```

**Workflow:**
1. Load VQ model from checkpoint
2. Load test dataset
3. Encode → Quantize → Decode motion
4. Compute reconstruction metrics (FID, etc.)

**Metrics Location:** Computed in eval modules (not visible in sampled code)

### 2. Mask Transformer Evaluation (`eval_mask.py`)

**Purpose:** Evaluate masked transformer generation quality

**Usage:**
```bash
python eval_mask.py \
    --dataset_name humanml3d \
    --mtrans_name pretrain_mtrans \
    --gpu_id 0 \
    --cond_scale 4 \
    --time_steps 10 \
    --ext eval \
    --repeat_times 1 \
    --which_epoch net_best_fid.tar
```

**Workflow:**
1. Load VQ model (for quantization)
2. Load Mask Transformer checkpoint
3. Load text encodings (from retriever or precomputed)
4. Iterative generation with classifier-free guidance
5. Decode to motion space
6. Compute metrics (FID, Inception, Frechet, etc.)

**Key Parameters:**
- `--cond_scale`: Guidance scale (4 typical)
- `--time_steps`: Diffusion steps (10 typical)
- `--repeat_times`: Multiple generations per text

### 3. Residual Transformer Evaluation (`eval_res.py`)

**Purpose:** End-to-end generation quality (Mask + Residual)

**Usage:**
```bash
python eval_res.py \
    --gpu_id 0 \
    --dataset_name humanml3d \
    --mtrans_name pretrain_mtrans \
    --rtrans_name pretrain_rtrans \
    --cond_scale 4 \
    --time_steps 10 \
    --ext eval \
    --which_ckpt net_best_fid.tar \
    --which_epoch fid \
    --traverse_res
```

**Workflow:**
1. Load VQ model + both transformers
2. Load retriever + RAG database
3. For each test sample:
   - Encode text to embedding
   - Retrieve top-K similar motions
   - Generate coarse codes via Mask Transformer
   - Refine with Residual Transformer
   - Decode to motion space
4. Compute full pipeline metrics

**Flags:**
- `--traverse_res`: Include motion-space traversal analysis
- `--which_epoch fid`: Use FID-best checkpoint

### 4. Retriever Evaluation (`Part_TMR/scripts/test.py`)

**Purpose:** Text-motion retrieval performance

**Usage:**
```bash
python Part_TMR/scripts/test.py \
    device=cuda:0 \
    train=train \
    exp_name=exp_pretrain
```

**Metrics Computed:**
- R@1, R@5, R@10 (retrieval recall)
- mAP (mean average precision)
- Uses Hydra config from `Part_TMR/conf/`

## Mocking & Test Data

**Status:** No mocking framework found

- No unittest.mock or pytest-mock imports
- Test data comes from actual HumanML3D/KIT-ML datasets
- Fixtures not used; evaluation uses production dataset splits

## How to Verify Changes Work

### Full Pipeline Verification (Recommended)

**Step 1: Check individual components**
```bash
# VQ quantizer reconstruction
python eval_vq.py \
    --gpu_id 0 \
    --name <your_vq_name> \
    --dataset_name humanml3d \
    --ext test

# Mask Transformer generation
python eval_mask.py \
    --dataset_name humanml3d \
    --mtrans_name <your_mtrans_name> \
    --gpu_id 0 \
    --ext test \
    --which_epoch net_best_fid.tar

# Residual Transformer end-to-end
python eval_res.py \
    --gpu_id 0 \
    --dataset_name humanml3d \
    --mtrans_name <your_mtrans_name> \
    --rtrans_name <your_rtrans_name> \
    --ext test \
    --which_epoch fid
```

**Step 2: Verify retriever integration**
```bash
# Retriever model test
python Part_TMR/scripts/test.py \
    device=cuda:0 \
    train=train \
    exp_name=<your_rag_name>
```

**Step 3: Check inference correctness**
```bash
# Quick demo with single text prompt
python demo.py \
    --gpu_id 0 \
    --ext debug \
    --text_prompt "A person walks forward." \
    --checkpoints_dir logs \
    --dataset_name humanml3d \
    --mtrans_name <your_mtrans> \
    --rtrans_name <your_rtrans>

# Output: ./outputs/debug/*.mp4 (visual inspection)
```

### Quick Smoke Tests (During Development)

**For transformer changes:**
1. Check shape consistency through `forward()` pass
2. Run single batch through trainer `.forward()` → verify loss shape is scalar
3. Verify gradients propagate: `.backward()` without error

**For retrieval changes (Plan A):**
1. Check latent dimension consistency: `latent_dim == 512` (see CONVENTIONS.md)
2. Verify `re_dict` structure reaches `SemanticsModulatedAttention` correctly
3. Run inference on 1 sample: `python eval_mask.py --repeat_times 1 --ext debug`

### Integration Test Sequence

**Training → Validation → Evaluation → Demo:**

1. **Train for 1 epoch (debug):**
```bash
bash run_mtrans.sh \
    debug_mtrans 1 0 12345 humanml3d \
    --vq_name pretrain_vq \
    --batch_size 4 \
    --max_epoch 1 \
    --eval_every_e 1
```

2. **Check checkpoint created:**
```bash
ls logs/humanml3d/debug_mtrans/model/
# Should see: net_best_fid.tar
```

3. **Evaluate on subset:**
```bash
python eval_mask.py \
    --dataset_name humanml3d \
    --mtrans_name debug_mtrans \
    --gpu_id 0 \
    --ext debug \
    --which_epoch net_best_fid.tar \
    --time_steps 1  # Minimal steps for speed
```

4. **Visual inspection:**
```bash
ls outputs/debug/
# Review *.mp4 for reasonable motion (not constant/random)
```

## Metrics & Evaluation Modules

**Location:** `utils/eval_t2m_ddp.py`

**Functions (referenced but not fully visible):**
- `evaluation_mask_transformer()` - Compute generation metrics
- `evaluation_res_transformer()` - Compute end-to-end metrics

**Typical Metrics:**
- **FID (Fréchet Inception Distance):** Motion distribution match
- **Inception Score:** Diversity & realism
- **MultiMatch:** Text-motion alignment
- **Diversity:** Motion variety within class

**Calculation Pattern:**
- Compute on entire test split
- Aggregate per text class or globally
- Log to TensorBoard + stdout

## Checkpoint Management

**Checkpoint Structure:**
```python
state = {
    'mask_transformer_aux': state_dict,      # 1D transformer
    'mask_transformer_ts': state_dict,       # 2D transformer
    'opt_mask_transformer_aux': optimizer,   # 1D optimizer
    'scheduler_aux': scheduler,              # 1D scheduler
    'opt_mask_transformer_ts': optimizer,    # 2D optimizer
    'scheduler_ts': scheduler,               # 2D scheduler
    'ep': epoch_number,
    'best_value': metric_value,
}
```

**Best Checkpoint:** `net_best_fid.tar` (based on FID score)

**Location:** `logs/<dataset>/<model_name>/model/`

## Reproduction Steps (Full Run)

### Phase 1: Train Retriever (Optional if using pretrain)
```bash
python Part_TMR/scripts/train.py \
    device=cuda:0 \
    exp_name=exp_rag_v1 \
    train.optimizer.motion_lr=1.0e-05
```

### Phase 2: Build RAG Database
```bash
python build_rag_database.py \
    device=cuda:0 \
    exp_name=exp_rag_v1
# Creates ./database/
```

### Phase 3: Train VQ Quantizer
```bash
bash run_rvq.sh \
    vq_v1 \
    1 0 12345 humanml3d \
    --batch_size 256 \
    --num_quantizers 6 \
    --max_epoch 50
```

### Phase 4: Train Mask Transformer
```bash
bash run_mtrans.sh \
    mtrans_v1 \
    1 0 12345 humanml3d \
    --vq_name vq_v1 \
    --batch_size 64 \
    --max_epoch 200 \
    --latent_dim 512 \
    --n_heads 8 \
    --attnj --attnt
```

### Phase 5: Train Residual Transformer
```bash
bash run_rtrans.sh \
    rtrans_v1 \
    1 0 12345 humanml3d \
    --vq_name vq_v1 \
    --mtrans_name mtrans_v1 \
    --batch_size 64 \
    --max_epoch 200
```

### Phase 6: Evaluate All Stages
```bash
# Retriever
python Part_TMR/scripts/test.py device=cuda:0 exp_name=exp_rag_v1

# VQ
python eval_vq.py --gpu_id 0 --name vq_v1 --dataset_name humanml3d

# Mask Transformer
python eval_mask.py --dataset_name humanml3d --mtrans_name mtrans_v1 --gpu_id 0

# End-to-End
python eval_res.py \
    --gpu_id 0 \
    --dataset_name humanml3d \
    --mtrans_name mtrans_v1 \
    --rtrans_name rtrans_v1

# Visual Demo
python demo.py \
    --gpu_id 0 \
    --text_prompt "A person jumps high." \
    --mtrans_name mtrans_v1 \
    --rtrans_name rtrans_v1
```

## Test Data & Fixtures

**Datasets:**
- `HumanML3D` - Primary dataset (training)
- `KIT-ML` - Secondary dataset (evaluation only)
- Dataset splits: `train.txt`, `val.txt`, `test.txt`

**Normalization:**
- Mean/Std precomputed: `<dataset>/Mean.npy`, `<dataset>/Std.npy`
- Loaded in eval scripts: `mean = np.load(...), std = np.load(...)`

**External Models (Pre-trained):**
- CLIP (frozen): `ViT-B-32.pt`
- TMR retriever: `Part_TMR/checkpoints/exp_pretrain/HumanML3D/`
- VQ codebook: Dynamic from checkpoint

## Known Testing Gaps

1. **No unit tests:** No isolated component testing (transformers, quantizers, retrieval)
2. **No integration tests:** No automated pipeline verification
3. **No regression tests:** No metrics tracking across versions
4. **No edge cases:** No handling of variable-length sequences, empty batches, OOM scenarios
5. **Limited error messages:** Failures often silent or cryptic

---

*Testing analysis: 2026-06-27*
