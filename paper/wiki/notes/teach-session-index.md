---
type: note
date: 2026-06-27
---

## Teach Session 课程索引

ReMoMask V1 代码深度教学课程（HTML 交互式），覆盖架构全景到 TPAMI 论文叙事。

### 课程目录

| # | 文件 | 主题 |
|---|---|---|
| 01 | `D:\tpami\lessons\0001-system-architecture.html` | 系统架构全景：五级流水线、数据流向 |
| 02 | `D:\tpami\lessons\0002-dual-branch-rvqvae.html` | 双分支 RVQ-VAE：1D 全局 + 2D 关节分支量化 |
| 03 | `D:\tpami\lessons\0003-mask-transformer-2d.html` | 2D Mask Transformer：时间-关节掩码、cosine schedule |
| 04 | `D:\tpami\lessons\0004-ssta-semantics-modulated-attention.html` | SSTA 融合模块：非对称 Q/K/V、info_mlp、维度耦合约束 |
| 05 | `D:\tpami\lessons\0005-rag-retrieval-part-tmr.html` | RAG + Part_TMR：HBM 对比学习、6 部位编码、数据库构建 |
| 06 | `D:\tpami\lessons\0006-residual-transformer-training.html` | 残差 Transformer 训练策略：逐层预测、四阶段训练 |
| 07 | `D:\tpami\lessons\0007-evaluation-and-v2-directions.html` | 评估指标体系与 V2 方向初探 |
| 08 | `D:\tpami\lessons\0008-rag-comparison-four-papers.html` | V1 RAG 与四篇 latent retrieval 论文的五维事实对比 |
| 09 | `D:\tpami\lessons\0009-v2-plan-ab-combination.html` | V2 Plan A+B 组合方案详解：改动清单、架构图、消融矩阵 |
| 10 | `D:\tpami\lessons\0010-tpami-paper-overview.html` | TPAMI 论文全景：叙事结构、方法论证、实验数据、消融 |

### 参考材料

| 文件 | 内容 |
|---|---|
| `D:\tpami\reference\glossary.html` | 术语表 |
| `D:\tpami\reference\architecture-cheatsheet.html` | 架构速查表 |

### 学习进度记录（已迁入 wiki/notes/）

| 记录 | wiki/notes/ 对应文件 |
|---|---|
| 0001 先验知识基线 | [[prior-knowledge-baseline]] |
| 0002 V1 RAG 架构理解 | [[v1-rag-architecture-understood]] |
| 0003a TPAMI 论文叙事理解 | [[tpami-paper-narrative-understood]] |
| 0003b V2 Plan A 确认 | [[v2-plan-a-confirmed]] |

### 其他分析产物（已迁入 wiki/notes/）

| 来源 | wiki/notes/ 对应文件 |
|---|---|
| 代码库深度解构报告 | [[codebase-deep-analysis]] |
| TPAMI 论文事实摘要（17-agent 验证） | [[paper-summary-factcheck]] |
| V2 Plan A 实施进度 | [[v2-plan-a-progress]] |
