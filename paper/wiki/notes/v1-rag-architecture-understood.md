---
type: note
source: D:\tpami\learning-records\0002-v1-rag-architecture-understood.md
date: 2026-06-26
---

## V1 RAG 架构已完全理解

已完整理解 ReMoMask V1 的 RAG 实现：
- Part_TMR 独立空间检索（512 维对比空间）
- 一次性检索不更新（re_dict 在 10 步循环内固定）
- SSTA 的 mean pooling + K/V 分离注入

通过与四篇 latent retrieval 论文（LongLive-RAG、LatentRAG、LAnR、RADiAnce）的事实对比（Lesson 08），已清楚 V1 在五个维度上的定位：检索空间、检索时机、融合粒度、检索库内容、训练关系。

### 关键结论
- V1 是五个方法中唯一「检索空间与生成空间完全割裂」的
- V1 是唯一对检索结果做 mean pooling 的
- 这两点正是 Plan A 要解决的核心问题

### 更新（2026-07-05）
Plan A 已实现，实际解决的是第一点（检索空间迁移到生成器自身的预量化 z_e 空间，冻结 VQ-VAE + KL 蒸馏 query projector）；mean pooling 粒度**未**改——z_e 检索键本身就是对 (T/4, 6) 做时空 mean pool + L2 norm 的产物，粒度问题留待后续。另补一条 V1 事实：SSTA 发表版信息路由为 K-only w.r.t. R_t——Q = W_q z；K = W_k concat(z, h_sem)，h_sem = MLP(concat[t; R_t; R_m])；V = W_v concat(z, R_m)，t、R_t 明确不进 Value。

### 相关
- [[latent-space-retrieval-alignment]]
- [[gap-latent-aligned-retrieval-for-motion]]
