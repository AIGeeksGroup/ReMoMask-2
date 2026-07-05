---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 4
current_phase_name: Comprehensive Evaluation & Paper Experiments
status: executing
stopped_at: "Phase 4 Wave 1 在飞（jobs 13885-13892，见 phases/04-*/RUNS.md）；Phase 2 训练仍在跑（提前停待拍板）；论文期刊版扩写已完成第一轮（v2working）。"
last_updated: "2026-07-05"
last_activity: 2026-07-05
last_activity_desc: "论文写作启动：深读 v1 全文、表格改名+ReMoMask-2 占位行、方法/intro/消融起草中"
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
| 2. Latent-Aligned Retrieval | **训练收尾中** | 6/6 plans done;V1/V2 对照训练 ~65%,best 已定型(ep316/ep409),提前停待拍板 |
| 3. Iterative Dynamic Retrieval | **DEFERRED** | Plan B 搁置(2026-07-05),论文仅 future work |
| 4. Comprehensive Eval & Paper Experiments | **IN PROGRESS** | E01-E08 在飞/排队(RUNS.md);E03 ✅(ρ=0.58, overlap@10=18.5%);E09 压轴 800ep;E12/KIT/Snap 随大服务器批 |

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
- V1/V2 对照训练：单卡 L40 × 2，配置与 V1 原版 opt.txt 完全一致 + 双方均开 rt_in_value，唯一变量 = 检索模块
- rt_in_value 定位：独立 minor contribution（Phase 1 ABL-02，原始 V1 无此项），论文中作正交消融维度

## Remote Server

- Host: diana.acfr.usyd.edu.au (USYD ACFR)
- Code: ~/ReMoMask-2 (git synced with local)
- Environment: conda remomask (Python 3.10 + PyTorch 2.1.0+cu118)
- Data: checkpoints + database (全量重建 23384 条) + HumanML3D + database_ze 全部就位

## SLURM Training (2026-07-05 快照)

| Job ID | 名称 | 进度 | 全局 best FID | 最优 ckpt |
|--------|------|------|--------------|----------|
| 13845 | v2_train (resume 自 13784@ep766) | ep ~1271/2000 | **0.0991 @ ep409** | net_best_fid_ep0409.tar |
| 13846 | v1_retrain (resume 自 13791@ep762) | ep ~1268/2000 | 0.1273 @ ep316 | net_best_fid_ep0316.tar |

- lr 恒定 2e-4，FID 已饱和 → 剩余 epoch 大概率不刷新 best，可考虑提前停
- 看门狗 train_watchdog.sh 在登录节点自动重交（脚本未实战验证）
- ⚠️ resume 后 net_best_fid.tar 被次优覆盖，正式评估显式用 ep0409/ep0316 文件

## Next Steps

1. **Phase 4 已重定义并写好执行规格**:`.planning/phases/04-comprehensive-evaluation/EXPERIMENTS-SPEC.md`(E01-E11 逐条命令/gap/成本)+ 04-CONTEXT.md(wave 结构 + 4 个拍板项 D-P4-01..04);Phase 3 (Plan B) 正式 DEFERRED
2. 训练停止(D-P4-01 拍板)→ Wave 1:E01 正式评估 + E02 mask-only + E03/E04 离线分析(E03 无前置,现在就能跑)
3. 代码 gap G1-G9(~435 LOC,spec 内清单)可先行实现+冒烟,分发 Sonnet xhigh
4. Phase 2 正式验证(/gsd-verify-work 2)在 E01 数字落地后跑
5. 论文：**写作已启动（2026-07-05）**。唯一可信 LaTeX = `D:\tpami\currentversion\v2working\`（解压自 `currentversion/_TPAMI_2026__ReMoMask_2 (1).zip`；旧路径 `_TPAMI_2026__ReMoMask_2/` 作废）。写作文档：`.planning/paper-v2/`（PLAN-A-FACTS / V2-REVISION-PLAN / FIG1-DESIGN / ABLATION-DESIGN）。v1 全文已深读；表格 Ours→ReMoMask + ReMoMask-2 高亮占位行已完成；方法/intro/消融扩写进行中

## Blockers

- 等待 V1/V2 训练（13845/13846）完成或做提前停决策

## Session Continuity

Last session: 2026-07-05(论文扩写 + Phase 4 开工)
**完整 handoff 见: .planning/phases/02-latent-aligned-retrieval-plan-a/.continue-here.md(顶部「傍晚更新」段)**
**在飞实验登记: .planning/phases/04-comprehensive-evaluation/RUNS.md(新 session 必读)**
Resume with: /gsd-progress 或 /gsd-resume-work
Monitor(训练): ssh diana.acfr.usyd.edu.au "sacct -j 13845,13846 --format=JobID,JobName%18,State%12,Elapsed -n"
Monitor(Phase4): ssh diana.acfr.usyd.edu.au "sacct -j 13885,13886,13887,13888,13889,13890,13891,13892 --format=JobID,JobName%14,State%12,Elapsed -n | grep -v batch"
⚠️ 上一 session 的进程内 Monitor 不跨 session,接手后需重挂或手动轮询。
