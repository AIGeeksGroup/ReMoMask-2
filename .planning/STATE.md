---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 4
current_phase_name: Comprehensive Evaluation & Paper Experiments
status: executing
stopped_at: "2026-07-06 深夜收数 sync 完成;E00/E00b/归因训练在飞"
last_updated: "2026-07-06T23:59:00.000Z"
last_activity: 2026-07-06
last_activity_desc: "E00 中期裁决(官方 ckpt ~0.122≠0.026)+ 三警报(E04/E05 无效、projector R@k chance、encoded_texts 污染)+ 修复重交 13916-13919"
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 7
  completed_plans: 7
  percent: 65
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-27)

**Core value:** 检索空间与生成空间统一后 FID 必须优于 V1
**Current focus:** Phase 4: Comprehensive Evaluation(E 系列实验 + 复现口径归因)+ 论文 V2 写作

## Current Position

Phase: 4 of 4 (Comprehensive Evaluation & Paper)
Status: E01/E02/E03 已出数;E00 口径裁决在飞;E04/E05 数字作废待代码调查;归因训练(v1_orig/T1)刚真正开跑
Last activity: 2026-07-06 深夜 — 收数 sync + 修复重交(详见 RUNS.md 顶部深夜段)

Progress: [██████▌░░░] 65%

## Phase Status

| Phase | Status | Summary |
|-------|--------|---------|
| 1. Zero-Cost Ablations | **COMPLETE** | ABL-01 CFG schedule 信号模糊; ABL-02 R_t-in-Value FID -13% ← 采纳(⚠️ 训练版 rt_in_value 嫌疑升级,见 V1-REPRO-GAP §3.59) |
| 2. Latent-Aligned Retrieval | **COMPLETE** | 6/6 plans;训练 TIMEOUT 终局 ep1676/1672,best 坐实 V2 0.0991@ep409 / V1 0.1273@ep316;正式验证挂 E 系列数字后 |
| 3. Iterative Dynamic Retrieval | **DEFERRED** | Plan B 搁置(2026-07-05),论文仅 future work |
| 4. Comprehensive Eval & Paper Experiments | **IN PROGRESS** | E01✅ E02✅ E03✅;E00 口径裁决在飞(中期:官方 ckpt ~0.122≠论文 0.026);E04/E05 数字作废(eval 检索无效警报);κ 表暂缓(R@k 全 chance);E09 压轴 |

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

## SLURM 在飞(2026-07-06 深夜快照;历史训练 13845/13846 已 TIMEOUT 终局)

| Job ID | 内容 | 节点 | 备注 |
|--------|------|------|------|
| 13909 | E00 官方 ckpt full ×20(口径裁决) | hades L4 | 中期 10/20:FID ~0.122,MMod ~1.39 |
| 13918 | E00b 官方 ckpt mask-only ×20 | hades L4 | 补齐 2×2 口径矩阵 |
| 13916 | v1_orig_single(D2+D3 归因,800ep) | persephone | 13908 秒崩修复后重交 |
| 13917 | T1 v1_lowlr(噪声地板,800ep) | persephone | 13910 秒崩修复后重交 |
| 13906/13907 | V1/V2 resume → 2000ep | persephone | 无望刷新 best;13907 吃了污染 encoded_texts,**去留待用户** |
| 13919 | encoded_texts.npy 恢复(CPU) | any | 基线 projector 确定性重投影 |

- 最优 ckpt 不变:**V2 0.0991@ep0409 / V1 0.1273@ep0316**(mask-only 单次口径);正式数字见 RUNS.md E01/E02 表
- ⚠️ 磁盘 /home 97%(余 210G)

## Next Steps

1. **收 E00 final + E00b** → 口径裁决落锤 → 整理「与一作对齐 0.026 评估命令」的问题清单(最高优先级)
2. **代码调查 ×2**:①eval_mask 生成路径是否消费 re_dict(E04/E05 四配置字节级相同的根因);②projector R@k 实现 + teacher 自身 R@1 基准(全 chance 之谜)
3. 收 13916/13917(v1_orig/T1)训练曲线 → rt_in_value-vs-batch 归因(V1-REPRO-GAP §3 三点定位)
4. T2(梯度累积 ×8 等价 8 卡)实现与提交,~20 LOC
5. 论文:叙事方案 A 等用户点头后回填;PLACEHOLDERS 回填遵循「口径统一」原则(E00 裁决影响主表口径设计)
6. Phase 2 正式验证(/gsd-verify-work 2)在口径问题厘清后跑

## Blockers

- E00/E00b 口径裁决(在飞,数小时)
- 一作侧信息:0.026 的确切评估命令/脚本(E00 落锤后发问)
- 用户拍板:13906/13907 去留;叙事方案 A;E04/E05 调查 vs 论文推进的优先级

## Session Continuity

**Stopped at:** 2026-07-06 深夜收数 sync 完成

Last session: 2026-07-06(深夜)
**完整 handoff 见: .planning/phases/02-latent-aligned-retrieval-plan-a/.continue-here.md(顶部「深夜」段)**
**在飞实验登记: .planning/phases/04-comprehensive-evaluation/RUNS.md(新 session 必读,顶部深夜 sync 段)**
**口径归因档案: .planning/phases/04-comprehensive-evaluation/V1-REPRO-GAP.md(§3.8 E00 中期裁决)**
Resume with: /gsd-progress 或 /gsd-resume-work
Monitor: ssh diana.acfr.usyd.edu.au "sacct -j 13906,13907,13909,13916,13917,13918,13919 --format=JobID,JobName%16,State%12,Elapsed -n | grep -v batch"
⚠️ 上一 session 的进程内 Monitor 不跨 session,接手后需重挂或手动轮询。
