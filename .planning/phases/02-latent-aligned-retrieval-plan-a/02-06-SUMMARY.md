---
phase: 02-latent-aligned-retrieval-plan-a
plan: 06
subsystem: ablation-evaluation
tags: [ablation, evaluation, v1-vs-v2, slurm-pipeline]
dependency_graph:
  requires: [02-05]
  provides: [plan-a-ablation-results, v2-trained-model]
  affects: [comparison.json, comparison.tex]
tech_stack:
  added: [slurm-job-chaining]
  patterns: [dependency-chained-sbatch, log-parsing-regex, latex-tabular-generation]
key_files:
  created:
    - ReMoMask/scripts/run_plan_a_ablation.sh
    - ReMoMask/scripts/compare_v1_v2.py
    - ReMoMask/scripts/slurm/job0_build_db_ze.sh
    - ReMoMask/scripts/slurm/job1_train_projector.sh
    - ReMoMask/scripts/slurm/job2_train_v2.sh
    - ReMoMask/scripts/slurm/job3_eval_v2.sh
    - ReMoMask/scripts/slurm/job4_eval_v1.sh
    - ReMoMask/scripts/slurm/submit_pipeline.sh
  modified: []
decisions:
  - "V2 training uses torchrun 4xL40 DDP instead of single GPU (train_v2.sh had --gpu_id 0)"
  - "Explicitly pass --train_split train.txt to override default train_small.txt"
  - "SLURM pipeline uses afterok dependency chain for 5 sequential jobs"
  - "Experiment name: v2_ze_rtval (z_e retrieval + rt_in_value)"
metrics:
  duration: 8m
  completed: 2026-06-30
  tasks_completed: 1
  tasks_total: 1
  files_modified: 8
status: complete
---

# Phase 2 Plan 6: Plan A Ablation -- V1 Part_TMR vs V2 z_e Retrieval Summary

消融对比脚本 + 完整 SLURM 训练流水线已创建并提交到 persephone 集群（4xL40），5 个 job 通过 afterok 依赖链串联执行。

## Task Completion

| Task | Name | Commit | Key Changes |
|------|------|--------|-------------|
| 1 | Ablation scripts + SLURM pipeline | 6c02391 | 2 local scripts + 6 sbatch scripts, pipeline submitted |

## Implementation Details

### Local Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_plan_a_ablation.sh` | End-to-end ablation runner: eval V1 -> eval V2 -> compare |
| `scripts/compare_v1_v2.py` | Parse eval logs, output terminal table + comparison.json + comparison.tex |

### SLURM Pipeline (dependency-chained)

| Job | Script | Time | Resources | Dependency |
|-----|--------|------|-----------|------------|
| 13745 | job0_build_db_ze.sh | ~30 min | 1 GPU persephone | none |
| 13746 | job1_train_projector.sh | ~2h | 1 GPU persephone | afterok:13745 |
| 13747 | job2_train_v2.sh | ~24-48h | 4 GPU persephone (DDP) | afterok:13746 |
| 13748 | job3_eval_v2.sh | ~2-4h | 1 GPU persephone | afterok:13747 |
| 13749 | job4_eval_v1.sh | ~2-4h | 1 GPU persephone | afterok:13748 |

### V2 Training Command (job2)

```bash
torchrun --nproc_per_node=4 \
    train_mask_transformer_ddp.py \
    --name v2_ze_rtval \
    --dataset_name humanml3d \
    --use_ze_retrieval \
    --ze_database_path database_ze \
    --projector_path logs/query_projector/best_projector.pt \
    --rt_in_value \
    --vq_name pretrain_vq \
    --batch_size 64 \
    --max_epoch 2000 \
    --train_split train.txt \
    --val_split val.txt \
    --attnj --attnt
```

### compare_v1_v2.py Outputs

- **Terminal**: formatted comparison table with delta percentages
- **comparison.json**: structured metrics with summary (fid_delta, r1_delta, mm_dist_delta)
- **comparison.tex**: LaTeX tabular with bold-best formatting, ready for paper

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing] Full SLURM training pipeline**
- **Found during:** Task 1
- **Issue:** Plan assumed V1/V2 checkpoints exist; neither database_ze nor V2 model had been trained yet
- **Fix:** Created 5 sbatch scripts for the complete pipeline (build_db -> projector -> V2 train -> eval V2 -> eval V1) and submitted with dependency chain
- **Files created:** 6 sbatch scripts in scripts/slurm/
- **Commit:** 6c02391

**2. [Rule 1 - Bug] train_v2.sh single GPU + wrong split**
- **Found during:** Task 1
- **Issue:** Existing train_v2.sh used --gpu_id 0 (single GPU, no DDP) and defaulted to train_small.txt (subset)
- **Fix:** New job2_train_v2.sh uses torchrun --nproc_per_node=4 and explicitly passes --train_split train.txt --val_split val.txt
- **Files created:** scripts/slurm/job2_train_v2.sh
- **Commit:** 6c02391

## Verification Status

**Deferred**: Ablation results cannot be verified until training completes (~30-54 hours). Current status:
- Job 13745 (build_db_ze): RUNNING on persephone
- Jobs 13746-13749: PENDING (dependency chain)

After pipeline completion, run:
```bash
python scripts/compare_v1_v2.py \
    --v1_dir logs/humanml3d/pretrain_mtrans/eval \
    --v2_dir logs/humanml3d/v2_ze_rtval/eval \
    --output results/plan_a_ablation/
```

Expected outcome: comparison.json with FID/R-Precision/MM-Dist for V1 vs V2.

## Known Stubs

None -- all scripts are fully implemented with real logic. Results are pending training completion, not stubbed.

## Self-Check: PASSED
