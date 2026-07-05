# Roadmap: ReMoMask V2

## Overview

ReMoMask V2 将检索增强动作生成（RAG-T2M）的检索空间从独立的 Part_TMR 对比空间迁移到 VQ-VAE latent space（Plan A），并引入迭代动态检索机制（Plan B），构成两个正交 contribution。四个阶段依次推进：先零成本消融验证前提假设，再实现 Plan A 核心改造，接着叠加 Plan B 动态检索，最后完成 2x2 消融矩阵和三数据集完整评估。

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Zero-Cost Ablations** - V1 上验证 RAG-CFG schedule 和 SSTA Value 分支假设（~30 LOC）
- [x] **Phase 2: Latent-Aligned Retrieval (Plan A)** - z_e 空间检索替换 Part_TMR，实现检索-生成空间统一（~200 LOC） (completed 2026-06-30)
- [ ] **Phase 3: Iterative Dynamic Retrieval (Plan B)** — **DEFERRED(2026-07-05 用户决定搁置,论文仅 future work 提及;不阻塞 Phase 4)**
- [ ] **Phase 4: Comprehensive Evaluation & Paper Experiments** - 论文全部占位符的数据生产:正式对照评估 + V2 消融全家桶(E01-E11)+ 图资产;详见 `.planning/phases/04-comprehensive-evaluation/EXPERIMENTS-SPEC.md`

## Phase Details

### Phase 1: Zero-Cost Ablations

**Goal**: 在 V1 代码上用最小改动验证两个设计前提：step-dependent RAG-CFG schedule 是否优于固定 s 值，以及 SSTA Value 分支加入 R_t 是否有益
**Depends on**: Nothing (first phase)
**Requirements**: ABL-01, ABL-02
**Success Criteria** (what must be TRUE):

  1. Step-dependent RAG-CFG schedule（s 值从高到低递减）vs 固定 s=4 的 FID/R-Precision 对比结果已产出，差异方向明确
  2. R_t in K+V vs K-only 的 FID/R-Precision 对比结果已产出，差异方向明确
  3. 两组消融结果已记录，可直接复用到论文消融表

**Plans**: TBD
**Estimated effort**: ~30 LOC, low complexity

### Phase 2: Latent-Aligned Retrieval (Plan A)

**Goal**: 用冻结 VQ-VAE 的预量化连续 latent z_e 替换 Part_TMR 编码，统一检索空间和生成空间，端到端训练后 FID 不劣于 V1 基线
**Depends on**: Phase 1 (消融结果指导 RAG-CFG 和 K/V 设计选择)
**Requirements**: LA-01, LA-02, LA-03, LA-04, LA-05, LA-06
**Success Criteria** (what must be TRUE):

  1. z_e 几何分布已验证：cosine similarity 分布合理，无严重各向异性或 dimensional collapse
  2. 新检索数据库已构建：RVQVAE encoder 输出的 z_e 向量替代 Part_TMR 512d 编码，retriever 正常返回 re_dict
  3. query_projector（CLIP 512d -> z_e 空间）KL 对齐训练收敛，BMM 当 teacher
  4. 端到端训练完成，HumanML3D 上 FID 不劣于 V1 基线（0.099 或更低）
  5. Plan A 消融（V1 Part_TMR 检索 vs V2 z_e 检索，其他不变）结果已产出

**Plans**: 6/6 plans complete
Plans:

- [x] 02-01-PLAN.md — LA-01: z_e 几何分布验证（Wave 1）
- [x] 02-02-PLAN.md — LA-02: z_e 数据库重建 + ZeRetriever（Wave 2）
- [x] 02-03-PLAN.md — LA-03: QueryProjector KL 对齐训练（Wave 2）
- [x] 02-04-PLAN.md — LA-04: SSTA 维度适配（Wave 2）
- [x] 02-05-PLAN.md — LA-05: 端到端训练（Wave 3）
- [x] 02-06-PLAN.md — LA-06: Plan A 消融对比（Wave 4）

**Estimated effort**: ~200 LOC, high complexity (z_e 几何未知、维度耦合风险)

### Phase 3: Iterative Dynamic Retrieval (Plan B)

**Goal**: 在去掩码生成过程的 50% 时做一次定点精检索，用已确定 token 构造更精准的 query，并在训练时模拟该行为消除 train-test mismatch
**Depends on**: Phase 2 (基于 Plan A 的新检索基础设施)
**Requirements**: IR-01, IR-02, IR-03
**Success Criteria** (what must be TRUE):

  1. generate() 在第 5 步（10 步中的 50%）可用已确定 token 重新编码 query 并执行精检索，re_dict 被更新
  2. 训练时通过随机模拟动态检索（替换 re_dict），train-test mismatch 消除
  3. Plan B 消融（静态 1 次 vs 动态 1 次 50% vs 动态 2 次 33%+67%）结果已产出，最优策略确定

**Plans**: TBD
**Estimated effort**: ~150 LOC, medium complexity

### Phase 4: Comprehensive Evaluation & Paper Experiments(2026-07-05 重定义)

**Goal**: 生产论文(currentversion/v2working/pami.tex)全部 45 处占位符的数据:V1/V2 正式对照评估、重定义 2×2 正交消融({Part_TMR, z_e} × {rt_in_value off/on})、候选扩充实验 E03-E08/E10、图资产;逐实验规格见 `04-comprehensive-evaluation/EXPERIMENTS-SPEC.md`
**Depends on**: Phase 2(训练停止);Phase 3 已 DEFERRED,不再是依赖
**Requirements**: COMB-01, COMB-02(重解释为 E01/E09), EV-01..EV-08(对应 E03-E10)
**Success Criteria** (what must be TRUE):

  1. E01:eval_res.py × 20 repeats × {V1@ep0316, V2@ep0409} 完成,comparison.json/tex 产出,主表 HumanML3D 行回填
  2. E09:2×2 正交消融四格齐(补训 Semantic✗ / z_e✗ 两格),tab:ablation_v2_orthogonal 回填
  3. E02-E08、E10 按 spec 完成,对应消融表/图回填;PLACEHOLDERS.md 相应条目清账
  4. pami.tex 重编译零错误,数字与 results/phase4/ 产出物一致(verifier 检查项)
  5. KIT-ML/SnapMoGen(04-07)按 D-P4-02 拍板结果执行或从主表移除对应占位行

**Plans**(wave 结构见 04-CONTEXT.md):

- [ ] 04-01 — E01 正式评估 + E02 mask-only 诊断(Wave 1)
- [ ] 04-02 — E03 秩相关 + E04 z_q 对照(Wave 1,离线,无训练前置)
- [ ] 04-03 — E05 top-k + E06 coverage-V2 + E07 效率(Wave 1)
- [ ] 04-04 — E08 projector 变体消融(Wave 2)
- [ ] 04-05 — E09 2×2 补两格训练(Wave 3,等 D-P4-03)
- [ ] 04-06 — E10 定性可视化 + 图资产(Wave 4)
- [ ] 04-07 — KIT-ML/SnapMoGen 扩展(已拍板:延后,随换大服务器的长训批执行)
- [ ] 04-08 — E12 rtrans 完整重训(条件性,与 04-07 同期;rtrans 与检索无关,一次重训全配置共享,见 EXPERIMENTS-SPEC E12)

**Estimated effort**: 代码 gap 共 9 项 ~435 LOC(可分发 Sonnet xhigh);算力大头 = E09 两格训练(~2.7 天并行)与 04-07(未定)

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Zero-Cost Ablations | 0/TBD | Complete | 2026-06-28 |
| 2. Latent-Aligned Retrieval | 6/6 | Complete   | 2026-06-30 |
| 3. Iterative Dynamic Retrieval | 0/TBD | Not started | - |
| 4. Comprehensive Evaluation | 0/TBD | Not started | - |
