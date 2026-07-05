# Phase 4: Comprehensive Evaluation & Paper Experiments — Context

**Gathered:** 2026-07-05
**Status:** Ready for planning(等 Phase 2 训练停止即可开跑 Wave 1)

## Phase Boundary(2026-07-05 重定义)

原 Phase 4 定义(2×2 = Part_TMR/z_e × 静态/动态)已过时:Plan B 搁置(Phase 3 DEFERRED),
2×2 重定义为 **{Semantic(Part_TMR), Latent-aligned(z_e)} × {rt_in_value off/on}**。
本 phase 承接论文(`D:\tpami\currentversion\v2working\pami.tex`)全部 45 处占位符的数据生产:
正式对照评估、V2 消融全家桶(含 2026-07-05 用户批准的候选扩充 1-8)、图资产数据。

**唯一事实来源**:`.planning/paper-v2/PLAN-A-FACTS.md`;
**逐实验执行规格**:本目录 `EXPERIMENTS-SPEC.md`(E01-E11);
**论文落点登记**:`.planning/paper-v2/PLACEHOLDERS.md`。

## 继承的锁定决策

- 论文数字纪律:只有完整 pipeline(eval_res.py)× 20 repeats 可进主表;诊断表(mask-only)
  用 eval_mask.py × 20 repeats,caption 注明口径;所有跨口径混用禁止。
- checkpoint 显式用 `net_best_fid_ep0409.tar`(V2)/`net_best_fid_ep0316.tar`(V1);
  无后缀 net_best_fid.tar 已被 resume 污染,禁止使用。
- 集群纪律照旧(erinyes 禁用、persephone mem≤15G、sbatch 必须 set -e、超参显式传全)。
- rt_in_value 叙事红线:premise-changed,消融结果不符时按 ABLATION-DESIGN.md 预案改写。

## 决策待定(BLOCKERS,进 GOAL.md §5)

- **D-P4-01** 提前停(触发 Wave 1 的前置);
- ~~**D-P4-02** KIT-ML / SnapMoGen~~ → **已拍板(2026-07-05):做,但延后**——等 HumanML3D
  全线结束后执行;**长时间训练可能切换到卡更多的新服务器**(届时重新核对环境/数据/
  ckpt 搬运,04-07 的资产盘点步骤照旧适用);主表两数据集的 ReMoMask-2 占位行**保留**;
- ~~**D-P4-03** 2×2 补格训练长度~~ → **已拍板(2026-07-05):800ep 截断;E09 排在全部
  eval 批次之后最后跑**(执行时以主线 sbatch 为模板:--max_epoch 800、去 --rt_in_value、
  实验名 v1_retrain_nortval / v2_ze_nortval,persephone 2×L40 并行,~2.7 天);
- ~~**D-P4-04** 效率图硬件口径~~ → **已拍板(2026-07-05):方案 A,整图 L40 重测**,
  caption 改 "a single NVIDIA L40"。执行顺序:① E07 先测 V1+V2;② 盘点基线
  (MoMask/ReMoDiffuse/MDM/T2M-GPT/MoGenTS/MoRAG/ReMoGPT)推理环境与 ckpt——集群上
  现成的直接测,缺的列清单向用户要 ECCV † 复现资产;个别实在拉不起来的基线,该点
  沿用旧值并在 caption 单独脚注(部分降级,不阻塞)。

## Wave 结构(依赖顺序)

| Wave | Plans | 前置 | GPU 需求 |
|---|---|---|---|
| 1 | 04-01 正式评估+mask-only;04-02 空间 gap 离线分析+z_q;04-03 推理旋钮(k/coverage/效率) | 训练停止(04-02 无此前置,随时可跑) | eval 级(单卡,小时级) |
| 2 | 04-04 projector 变体消融(objective/teacher/κ/容量) | 04-02 的 TMR 特征库(仅 teacher 行) | 单卡,每变体 <1h |
| 3 | 04-05 正交 2×2 补两格训练 | D-P4-03 拍板 | 2×L40,天级 |
| 4 | 04-06 定性可视化+图资产 | 04-01 数字(fig1 面板) | 轻 |
| — | 04-07 KIT/SnapMoGen 扩展 | D-P4-02 拍板(数据集侧留白) | 大,天-周级 |

## Downstream Notes

- for planner:各实验的命令/代码 gap/产出物/论文落点已在 EXPERIMENTS-SPEC.md 逐条写死,
  plan 只需按 wave 打包 + 排 SLURM;
- for verifier:验收 = PLACEHOLDERS.md 对应条目回填 + pami.tex 重编译零错误 + 数字与
  results/ 产出物一致。
