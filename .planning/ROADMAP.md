# Roadmap: ReMoMask V2

## Overview

ReMoMask V2 将检索增强动作生成（RAG-T2M）的检索空间从独立的 Part_TMR 对比空间迁移到 VQ-VAE latent space（Plan A），并引入迭代动态检索机制（Plan B），构成两个正交 contribution。四个阶段依次推进：先零成本消融验证前提假设，再实现 Plan A 核心改造，接着叠加 Plan B 动态检索，最后完成 2x2 消融矩阵和三数据集完整评估。

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Zero-Cost Ablations** - V1 上验证 RAG-CFG schedule 和 SSTA Value 分支假设（~30 LOC）
- [ ] **Phase 2: Latent-Aligned Retrieval (Plan A)** - z_e 空间检索替换 Part_TMR，实现检索-生成空间统一（~200 LOC）
- [ ] **Phase 3: Iterative Dynamic Retrieval (Plan B)** - 50% 时定点精检索 + 训练模拟（~150 LOC）
- [ ] **Phase 4: Comprehensive Evaluation** - 2x2 消融矩阵 + 三数据集完整评估 + 结果归档

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
**Plans**: TBD
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

### Phase 4: Comprehensive Evaluation
**Goal**: 完成 2x2 消融矩阵的全部四组配置评估，并在三数据集上跑完整 20 次重复评估，输出可直接用于论文的实验结果
**Depends on**: Phase 2, Phase 3
**Requirements**: COMB-01, COMB-02
**Success Criteria** (what must be TRUE):
  1. 2x2 消融矩阵四组配置（Part_TMR/z_e x 静态/动态）全部在 HumanML3D 上评估完成，FID/R-Precision/MM-Dist 数值齐全
  2. 三数据集（HumanML3D + KIT-ML + SnapMoGen）完整评估完成，每组 20 次重复，均值和标准差已计算
  3. 最优配置（预期为 z_e + 动态）的 FID 优于 V1 Part_TMR 基线
**Plans**: TBD
**Estimated effort**: 主要是训练/评估时间，代码改动极少

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Zero-Cost Ablations | 0/TBD | Not started | - |
| 2. Latent-Aligned Retrieval | 0/TBD | Not started | - |
| 3. Iterative Dynamic Retrieval | 0/TBD | Not started | - |
| 4. Comprehensive Evaluation | 0/TBD | Not started | - |
