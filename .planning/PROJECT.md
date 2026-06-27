# ReMoMask V2 — Latent-Aligned Retrieval for Text-to-Motion RAG

## What This Is

ReMoMask V2 是 ECCV 2025 中稿论文 ReMoMask 的 TPAMI 期刊扩展版。核心升级：将检索增强动作生成（RAG-T2M）的检索空间从独立的 Part_TMR 对比空间迁移到 VQ-VAE latent space（Plan A），并引入迭代动态检索机制（Plan B），构成两个正交 contribution 和 2×2 消融矩阵。

## Core Value

**检索空间与生成空间统一后，RAG-T2M 的生成质量（FID）必须优于 V1 的独立空间方案。** 如果 FID 不降反升，整个 V2 的技术叙事就站不住。

## Requirements

### Validated

- ✓ HBM 层级双向动量对比检索（Part_TMR）— ECCV 2025 已验证
- ✓ SSTA 非对称 Q/K/V 融合 — ECCV 2025 已验证
- ✓ TSM 拓扑结构化掩码训练策略 — ECCV 2025 已验证
- ✓ 2D 空间-时间 VQ-VAE 动作表示 — ECCV 2025 已验证
- ✓ 三数据集评估流程（HumanML3D / KIT-ML / SnapMoGen）— ECCV 2025 已验证

### Active

- [ ] **PLAN-A-01**: 在 VQ-VAE 预量化连续 latent z_e 上构建检索数据库
- [ ] **PLAN-A-02**: 实现 query_projector（CLIP 512d → z_e 空间），KL 散度对齐训练
- [ ] **PLAN-A-03**: 调整 SSTA 融合维度适配新检索空间
- [ ] **PLAN-A-04**: 端到端训练 + V1 vs V2 对比消融
- [ ] **PLAN-B-01**: 在 10 步去掩码中实现定点精检索（~50% 时一次）
- [ ] **PLAN-B-02**: 训练时模拟动态检索避免 train-test mismatch
- [ ] **ABL-01**: 2×2 消融矩阵（空间统一 × 动态检索）
- [ ] **ABL-02**: Step-dependent RAG-CFG schedule 消融（~20 LOC，零成本）
- [ ] **ABL-03**: R_t in K vs K+V 消融（~10 LOC，零成本）

### Out of Scope

- 重训 VQ-VAE — Plan A 冻结 VQ-VAE，只训 projector；联合训练仅在冻结方案效果不够时考虑
- 替换 SSTA 为全新融合架构 — V1 的 SSTA 已被审稿认可，只调维度不换架构
- 新数据集采集 — 用现有 HumanML3D/KIT-ML/SnapMoGen
- Plan B 的每步检索 — 收窄为 50% 时一次定点精检索，非每步都检索
- LAnR 式 entropy control head — 过于复杂，收益不确定

## Context

- **代码库**: ReMoMask (D:\tpami\ReMoMask\)，基于 MoMask 框架改造，PyTorch + DDP
- **知识库**: paper/wiki/ 下 21 篇论文、7 个 concept、5 个 gap、8 个 notes
- **已验证新颖性**: gap-latent-aligned-retrieval-for-motion 经 13 组 query 搜索确认 CONFIRMED
- **创新标准**: TPAMI 期刊拓展版，核心创新已被 ECCV 审稿认可，V2 新增部分标准可降低
- **codebase mapping**: .planning/codebase/ 下 7 个文档，2230 行
- **关键风险**: z_e 几何结构未知（各向异性/dimensional collapse）、23K 样本数据稀疏、SSTA 维度耦合

## Constraints

- **改动量**: Plan A 约 200 LOC，Plan B 约 150 LOC，总计不超过 400 LOC
- **架构不动**: VQ-VAE / Residual Transformer / 评估流程 / 数据加载不改
- **训练环境**: 本地写代码，远程 GPU 服务器跑训练（服务器待上线）
- **维度耦合**: SSTA 要求 latent_dim = clip_dim = 512（代码硬约束）
- **License**: CC BY-NC-SA 4.0

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 操作 z_e（预量化连续 latent）而非 z_q（离散 token） | 4/4 ideation agent 共识；绕开离散梯度问题；commitment loss 提供隐式聚类 | — Pending |
| 冻结 VQ-VAE，仅训练 projector | 低风险先行验证；联合训练作为 fallback | — Pending |
| KL 对齐而非 InfoNCE | LatentRAG 消融显示 KL 更鲁棒（43.46 > 41.86） | — Pending |
| BMM 当 teacher 而非 TMR | BMM R@1=13.76 远优于 TMR R@1=5.68 | — Pending |
| 先跑零成本消融（RAG-CFG schedule + R_t K/V） | 验证前提假设，~30 LOC，不依赖 Plan A | — Pending |
| Plan B 收窄为 50% 时一次定点精检索 | SARDI 跨域验证支持；降低复杂度 | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-27 after initialization*
