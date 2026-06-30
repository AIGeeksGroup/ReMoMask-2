---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 2
current_phase_name: Latent-Aligned Retrieval (Plan A)
status: executing
stopped_at: "Phase 2 Wave 3 complete. Wave 4 (ablation) blocked on full training. Remote server (diana.acfr.usyd.edu.au) configured and verified."
last_updated: "2026-06-30"
last_activity: 2026-06-30
last_activity_desc: "Phase 2 Wave 3 training integration complete; remote server setup + demo verified"
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 7
  completed_plans: 5
  percent: 71
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-27)

**Core value:** 检索空间与生成空间统一后 FID 必须优于 V1
**Current focus:** Phase 2: Latent-Aligned Retrieval (Plan A)

## Current Position

Phase: 2 of 4 (Latent-Aligned Retrieval)
Status: Wave 1-3 complete, Wave 4 blocked on full training
Last activity: 2026-06-30 — Remote server setup + demo verified

Progress: [███████░░░] 71%

## Phase Status

| Phase | Status | Summary |
|-------|--------|---------|
| 1. Zero-Cost Ablations | **COMPLETE** | ABL-01 CFG schedule 信号模糊; ABL-02 R_t-in-Value FID -13% ← 采纳 |
| 2. Latent-Aligned Retrieval | **IN PROGRESS** | Wave 1-3 done (z_e分析/数据库/projector/SSTA/训练脚本); Wave 4 待完整训练 |
| 3. Iterative Dynamic Retrieval | Not started | 依赖 Phase 2 |
| 4. Comprehensive Evaluation | Not started | 依赖 Phase 2+3 |

## Phase 1 Results

| Metric | Baseline | ABL-01 (CFG schedule) | ABL-02 (R_t in V) |
|---|---|---|---|
| FID ↓ | 0.102 | 0.098 (-4%) | **0.089 (-13%)** |
| Top-1 ↑ | 0.478 | 0.476 | 0.479 |

**Decision:** Phase 2 默认 rt_in_value=True

## Phase 2 Progress

| Wave | Plans | Status |
|------|-------|--------|
| 1 | 02-01 z_e 几何验证 | ✓ code_dim2d=1024, cosine retrieval viable, 严重各向异性但不阻塞 |
| 2 | 02-02 数据库重建 | ✓ database_ze/ 66912样本 1024d |
| 2 | 02-03 Query Projector | ✓ 1.57M params, 初步训练完成(小数据集) |
| 2 | 02-04 SSTA 适配 | ✓ retrieval_dim=1024 投影层 |
| 3 | 02-05 训练集成 | ✓ train/eval 脚本支持 --use_ze_retrieval |
| 4 | 02-06 消融对比 | ✓ 脚本已创建，SLURM 流水线已提交（13750-13754），等待训练完成 |

## Key Decisions (accumulated)

- z_e 预量化连续 latent（不是 z_q 离散 token）— 4/4 ideation agent 共识
- code_dim2d = 1024（LA-01 验证）→ projector: Linear(512→1024)
- 冻结 VQ-VAE + KL 对齐 + BMM teacher
- rt_in_value=True（Phase 1 ABL-02 验证 FID -13%）
- 远程训练环境：diana.acfr.usyd.edu.au SLURM 集群
- V2 实验名：v2_ze_rtval，DDP 4×L40 训练

## Remote Server

- Host: diana.acfr.usyd.edu.au (USYD ACFR)
- Code: ~/ReMoMask-2 (git synced with local)
- Environment: conda remomask (Python 3.10 + PyTorch 2.1.0+cu118)
- Data: checkpoints + database + HumanML3D + database_ze 全部就位

## SLURM Training Pipeline (2026-06-30 提交)

| Job ID | 名称 | 预计时长 | 节点 | 状态 |
|--------|------|---------|------|------|
| 13750 | rebuild_bmm (重建 BMM teacher 全量数据库) | ~5 min | persephone | PENDING |
| 13751 | proj_train (200 epochs KL 对齐) | ~2 h | persephone | afterok:13750 |
| 13752 | v2_train (2000 epochs, 4×L40 DDP) | ~24-48 h | persephone | afterok:13751 |
| 13753 | eval_v2 (20 repeats) | ~2-4 h | persephone | afterok:13752 |
| 13754 | eval_v1 baseline (20 repeats) | ~2-4 h | persephone | afterok:13753 |

修复的 bug：BMM 数据库只有 32 条（已重建）+ DDP os.makedirs 竞争条件（已修）

## Next Steps

1. 等待 SLURM 流水线完成（~30-54 小时）
2. 训练完成后运行 `python scripts/compare_v1_v2.py` 产出消融对比表
3. Phase 2 验证：检查 FID ≤ 0.099
4. Phase 3 (Plan B iterative retrieval)

## Blockers

- 等待 SLURM 训练流水线（13750-13754）完成

## Session Continuity

Last session: 2026-06-30
Resume with: /gsd-progress (check SLURM status + phase state)
Monitor: ssh diana.acfr.usyd.edu.au "squeue -u ywan0794"
