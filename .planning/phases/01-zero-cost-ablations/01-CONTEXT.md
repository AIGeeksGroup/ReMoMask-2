# Phase 1: Zero-Cost Ablations - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning

## Phase Boundary

在 V1 代码上用最小改动（~30 LOC）验证两个设计前提假设，为 Phase 2 (Plan A) 和 Phase 3 (Plan B) 提供实证依据。不改架构，只改超参数和 SSTA 信息路由。

## Implementation Decisions

### ABL-01: Step-Dependent RAG-CFG Schedule

- **D-01:** Baseline 是 V1 的固定 `cond_scale=4`（`transformer_ts.py` 中 `forward_with_cond_scale` 的 `cond_scale` 参数）
- **D-02:** 对比变体：10 步中 s 值线性递减，范围由我根据代码中 CFG 的实际行为确定（预计 s=6→2 或类似）。只测一组 schedule，不做网格搜索
- **D-03:** 实现方式：在 `generate()` 循环中将固定 `cond_scale` 替换为步相关的值，从一个预定义数组读取。不增加可学习参数
- **D-04:** 其他所有超参数（学习率、batch size、训练 epochs、top-K 检索数等）与 V1 完全一致

### ABL-02: R_t in SSTA Key vs Key+Value

- **D-05:** 单变量对比：V1 baseline（R_t 只在 K 中，不在 V 中）vs R_t 同时进入 K 和 V
- **D-06:** 实现方式：修改 `semantics_modulated.py` 的 V 分支构造，将 `R_t_pooled` 加入 `value()` 的拼接输入
- **D-07:** 不测其他变体（如 V-only 或去掉 R_t）——单变量足以回答「SSTA 的 K-only 设计是否是性能瓶颈」

### 评估范围

- **D-08:** 只在 HumanML3D 上评估，单次运行，看 FID/R-Precision/MM-Dist 的方向（升/降/持平）
- **D-09:** 完整 20 次重复 + 三数据集留到 Phase 4 综合评估
- **D-10:** 结果记录到 `.planning/phases/01-zero-cost-ablations/` 下，格式可直接复用到论文消融表

### 前置条件

- **D-11:** 需要 V1 的预训练 checkpoint（Part_TMR + MaskTransformer2D + ResidualTransformer + RVQVAE）和 RAG 数据库
- **D-12:** 需要 GPU 服务器在线。如果服务器暂时不可用，先完成代码修改，推迟训练/评估

## Deferred Ideas

- Step-dependent s 值如果有效，可以考虑在 Plan A 之后做可学习 schedule（当前阶段只做固定 schedule）
- R_t 的消融结果可能暗示 SSTA 需要更复杂的重设计，但不在 Phase 1 范围内

## Downstream Notes

- **For researcher:** Phase 1 不需要额外研究，改动完全基于已有代码
- **For planner:** 两个消融实验可以并行准备代码，但评估需要串行（共享 GPU）
- **For verifier:** 成功标准是消融结果产出且方向明确，不是 FID 必须降低
