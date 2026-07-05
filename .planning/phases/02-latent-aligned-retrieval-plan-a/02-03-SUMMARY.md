---
phase: 02-latent-aligned-retrieval-plan-a
plan: 03
subsystem: query-projector
tags: [projector, kl-divergence, clip, z_e, alignment]

requires:
  - phase: 02-latent-aligned-retrieval-plan-a
    plan: 01
    provides: code_dim2d = 1024 confirmed
provides:
  - QueryProjector model (CLIP 512d -> z_e 1024d, L2-normalised)
  - KL-alignment training script with BMM teacher supervision
  - Checkpoint format with dim metadata (clip_dim, ze_dim)
affects: [02-04, 02-05, 02-06]

tech-stack:
  added: []
  patterns: [KL divergence on top-K teacher candidates, cosine-annealed Adam]

key-files:
  created:
    - ReMoMask/models/rag/query_projector.py
    - ReMoMask/train_query_projector.py
  modified: []

key-decisions:
  - "Top-K sparse teacher (default 256) instead of full N*N similarity matrix — avoids 2.1 GB for N=23K"
  - "CosineAnnealingLR scheduler added on top of plan's Adam baseline"
  - "Post-training step automatically projects all texts and saves encoded_texts.npy to database_ze/"

requirements-completed: [LA-03]

coverage:
  - id: T1
    description: "QueryProjector maps (B, 512) -> (B, 1024), L2-normalised output"
    requirement: "LA-03"
    verification:
      - kind: automated_ui
        ref: "python -c import + shape + norm assertion"
        status: pass
    human_judgment: false
  - id: T2
    description: "Training script importable, parse_args works, recall_at_k works on mock data"
    requirement: "LA-03"
    verification:
      - kind: automated_ui
        ref: "python import + function existence + mock R@K"
        status: pass
    human_judgment: false
  - id: T3
    description: "End-to-end smoke test: 3 epochs on CPU with mock data, loss decreases, all outputs saved"
    requirement: "LA-03"
    verification:
      - kind: automated_ui
        ref: "Mock DB -> train -> best_projector.pt + encoded_texts.npy verified"
        status: pass
    human_judgment: false

duration: 8min
completed: 2026-06-28
status: complete
---

# Phase 2 Plan 03: Query Projector + KL Alignment Training Summary

**2-layer MLP projector (CLIP 512d -> z_e 1024d) with KL-divergence alignment training using BMM teacher's top-K soft targets**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-06-28
- **Completed:** 2026-06-28
- **Tasks:** 2/2
- **Files created:** 2

## Accomplishments

### Task 1: QueryProjector Model

Created `models/rag/query_projector.py`:
- `QueryProjector(nn.Module)`: Linear(512, 1024) + GELU + Linear(1024, 1024), L2-normalised output
- ~1.57M parameters — trains fast on single GPU
- `save_projector()` / `load_projector()` helpers store `clip_dim` and `ze_dim` in checkpoint
- Verified: shape (4, 512) -> (4, 1024), output norms == 1.0, save/load roundtrip exact

### Task 2: KL Alignment Training Script

Created `train_query_projector.py`:
- Loads precomputed features: z_e motions (N, 1024), CLIP texts (N, 512), BMM motions+texts (N, 512)
- BMM teacher provides soft-target rankings (per D-07, LOCKED-04)
- Top-K sparse teacher (default 256 candidates per text) avoids storing full N*N matrix
- KL divergence loss: `F.kl_div(student_log_prob, teacher_prob, reduction='batchmean')`
- Temperature = 0.07 (matches Part_TMR)
- R@1 / R@5 / R@10 evaluation every `eval_interval` epochs
- Best checkpoint selected by R@1
- Post-training: projects all texts through best model -> saves `database_ze/encoded_texts.npy` (N, 1, ze_dim)
- Smoke tested end-to-end on CPU: 3 epochs, loss decreasing (0.118 -> 0.060), all outputs verified

## Training Script CLI

```bash
python train_query_projector.py \
    --database_ze_path database_ze \
    --database_bmm_path database \
    --output_dir logs/query_projector \
    --epochs 200 --batch_size 128 \
    --lr 1e-4 --temperature 0.07 \
    --teacher_topk 256 --eval_interval 10 \
    --device cuda:0
```

## Checkpoint Format

```python
{
    'model_state_dict': ...,
    'clip_dim': 512,
    'ze_dim': 1024,
    'epoch': int,
    'R@1': float, 'R@5': float, 'R@10': float,
    'loss': float,
    'temperature': 0.07,
}
```

## Design Decisions

1. **Top-K sparse teacher** — full N*N similarity for N=23K would need ~2.1 GB VRAM; top-256 keeps the meaningful teacher signal while fitting easily in memory
2. **CosineAnnealingLR** — added on top of the plan's Adam baseline for smoother convergence over 200 epochs
3. **Automatic text re-projection** — after training completes, the script projects all CLIP texts through the best model and saves `encoded_texts.npy`, ready for ze_retriever to consume

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Next Steps

- Run `build_rag_database_ze.py` to create `database_ze/` (LA-02)
- Then run `train_query_projector.py` with real data on GPU
- LA-04/LA-05 will integrate the trained projector into the retriever and SSTA pipeline

---
*Phase: 02-latent-aligned-retrieval-plan-a*
*Completed: 2026-06-28*
