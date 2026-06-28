---
phase: 02-latent-aligned-retrieval-plan-a
plan: 05
subsystem: training-pipeline
tags: [training, integration, retrieval, v2-pipeline]
dependency_graph:
  requires: [02-02, 02-03, 02-04]
  provides: [v2-training-pipeline, v2-eval-pipeline]
  affects: [train_mask_transformer_ddp.py, eval_mask.py, eval_res.py]
tech_stack:
  added: []
  patterns: [conditional-retriever-loading, cli-flag-gating, strict-false-checkpoint-loading]
key_files:
  created: []
  modified:
    - ReMoMask/train_mask_transformer_ddp.py
    - ReMoMask/models/transformer/transformer_trainer_ddp.py
    - ReMoMask/models/rag/ze_retriever.py
    - ReMoMask/eval_mask.py
    - ReMoMask/eval_res.py
    - ReMoMask/options/eval_option.py
decisions:
  - "V2 pipeline gated behind --use_ze_retrieval flag; V1 path untouched"
  - "ZeRetriever gets tokenize()/encode_text() matching MocoTmrRetriever interface; trainer code needs no branching"
  - "retrieval_dim auto-detected from vq_model.code_dim2d (1024) when V2; CLI override available"
  - "Checkpoint loading uses strict=False with relaxed assertions for new SSTA projection layer keys"
metrics:
  duration: 8m
  completed: 2026-06-28
  tasks_completed: 4
  tasks_total: 4
  files_modified: 6
status: complete
---

# Phase 2 Plan 5: End-to-End Training Pipeline Integration Summary

V2 z_e retrieval pipeline integrated into training and eval scripts via --use_ze_retrieval flag, loading ZeRetriever with 1024d z_e features and passing retrieval_dim to SSTA projection layers.

## Task Completion

| Task | Name | Commit | Key Changes |
|------|------|--------|-------------|
| 1 | Training entry point V2 support | 948f370 | CLI args, ZeRetriever loading, retrieval_dim passthrough, DDP safety check, rt_in_value |
| 2 | Trainer + ZeRetriever adaptation | 4a60158 | tokenize()/encode_text() on ZeRetriever, re_text projection in forward(), relaxed resume assertions |
| 3 | Eval scripts V2 support | cb99894 | eval_mask.py/eval_res.py: V2 retriever loading, retrieval_dim, EvalT2MOptions CLI args |
| 4 | Smoke test | (verified inline) | 7 assertions pass: arg parsing, interface checks, SSTA 1024d forward pass |

## Implementation Details

### CLI Interface (all flags backward-compatible, V1 is default)

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| --use_ze_retrieval | bool | False | Switch from MocoTmrRetriever to ZeRetriever |
| --ze_database_path | str | database_ze | z_e database directory |
| --projector_path | str | logs/query_projector/best_projector.pt | Trained query projector checkpoint |
| --rt_in_value | bool | False | Phase 1 ABL-02: R_t in SSTA Value branch |
| --retrieval_dim | int | None (auto) | SSTA retrieval feature dimension |

### Key Integration Points

1. **Retriever loading** (train_mask_transformer_ddp.py lines 159-184): conditional branch loads ZeRetriever when --use_ze_retrieval, with DDP safety assertion checking database_ze/metadata.json exists before proceeding.

2. **SSTA retrieval_dim** (lines 203-220): auto-detects code_dim2d=1024 from VQ model when V2 enabled, passes to MaskTransformer2D constructor. SSTA creates Linear(1024->384) projection layers for re_motion and re_text.

3. **ZeRetriever text encoding** (ze_retriever.py): new tokenize() and encode_text() methods delegate to internal CLIP model, matching MocoTmrRetriever's interface so trainer code is unchanged.

4. **Checkpoint compatibility** (transformer_trainer_ddp.py): resume() assertions allow missing keys for `re_motion_proj.*` and `re_text_proj.*` in addition to `clip_model.*`, enabling V1 checkpoint loading into V2 model (new projection layers init randomly).

### V2 Training Command (8-GPU DDP)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
  train_mask_transformer_ddp.py \
  --name v2_ze_exp \
  --dataset_name humanml3d \
  --use_ze_retrieval \
  --ze_database_path database_ze \
  --projector_path logs/query_projector/best_projector.pt \
  --rt_in_value \
  --vq_name {VQ_NAME} \
  --batch_size 64 \
  --max_epoch 2000
```

### V2 Eval Command

```bash
python eval_mask.py \
  --name v2_ze_exp \
  --dataset_name humanml3d \
  --use_ze_retrieval \
  --ze_database_path database_ze \
  --projector_path logs/query_projector/best_projector.pt \
  --rt_in_value \
  --retrieval_dim 1024 \
  --mtrans_name v2_ze_exp
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing] ZeRetriever tokenize/encode_text interface**
- **Found during:** Task 2
- **Issue:** Trainer calls self.retriever.encode_text(text_ids) but ZeRetriever only had encode_query_text() for string input; no pre-tokenized text ID encoding.
- **Fix:** Added tokenize() and encode_text() methods to ZeRetriever delegating to internal CLIP model, matching MocoTmrRetriever's interface exactly.
- **Files modified:** models/rag/ze_retriever.py
- **Commit:** 4a60158

**2. [Rule 2 - Missing] Checkpoint loading assertions too strict for V2**
- **Found during:** Task 2
- **Issue:** MaskTransformerTrainer.resume() asserted all missing keys start with 'clip_model.' but V2's new SSTA retrieval projection layers would also be missing when loading V1 checkpoints.
- **Fix:** Relaxed assertions to also allow re_motion_proj.*/re_text_proj.* missing keys.
- **Files modified:** models/transformer/transformer_trainer_ddp.py
- **Commit:** 4a60158

**3. [Rule 2 - Missing] Flexible checkpoint key lookup in eval scripts**
- **Found during:** Task 3
- **Issue:** Eval scripts hardcoded ckpt['t2m_transformer_ts'] but trainer saves as ckpt['mask_transformer_ts']. Different checkpoint formats from eval-time vs train-time saves.
- **Fix:** Added fallback: tries 't2m_transformer_ts' first, falls back to 'mask_transformer_ts'.
- **Files modified:** eval_mask.py, eval_res.py
- **Commit:** cb99894

## Verification Results

All 7 smoke test assertions passed:
1. TrainT2MOptions V2 args parsed correctly
2. EvalT2MOptions V2 args parsed correctly
3. ZeRetriever has tokenize/encode_text/forward methods
4. MaskTransformer2D accepts retrieval_dim parameter
5. SemanticsModulatedAttention accepts retrieval_dim and rt_in_value
6. MaskTransformerTrainer imports without error
7. SSTA forward pass with 1024d retrieval features produces correct shape (B, N, 384)

## Known Stubs

None -- all integration points are wired with actual implementation.

## Self-Check: PASSED
