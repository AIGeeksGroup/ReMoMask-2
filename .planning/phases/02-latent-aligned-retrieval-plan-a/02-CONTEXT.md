# Phase 2: Latent-Aligned Retrieval (Plan A) - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning

## Phase Boundary

将 ReMoMask 的检索空间从独立的 Part_TMR 对比空间迁移到 VQ-VAE 的预量化连续 latent z_e 空间，使检索和生成共享同一表征。冻结 VQ-VAE，只训练 query_projector。~200 LOC 改动，4-5 个文件。

## Implementation Decisions

### 从 Phase 1 和 ideation 继承的锁定决策

- **LOCKED-01:** 操作 z_e（预量化连续 latent），不操作 z_q（离散 token）— 4/4 ideation agent 共识，绕开离散梯度问题
- **LOCKED-02:** 冻结 VQ-VAE，仅训练 projector — 低风险先行验证，联合训练作为 fallback
- **LOCKED-03:** KL 散度对齐而非 InfoNCE — LatentRAG 消融数据支持（KL 43.46 > InfoNCE 41.86）
- **LOCKED-04:** BMM 当 teacher 而非 TMR — BMM R@1=13.76 远优于 TMR R@1=5.68
- **LOCKED-05:** SSTA 架构不换，只调维度适配

### z_e 提取与聚合

- **D-01:** 从 2D 分支（joint-time grid）提取 z_e，不用 1D 全局分支 — 与 SSTA 的 2D spatial-temporal token grid 一致
- **D-02:** z_e 形状预计为 (B, code_dim, T/4, 6)，通过时间维+关节维 mean pooling 压缩为 (B, code_dim) 作为检索向量
- **D-03:** 提取位置：`models/vq/model.py` 中 RVQVAE encoder 的输出，量化操作之前

### Query Projector 架构

- **D-04:** 2 层 MLP：Linear(512→D) + GELU + Linear(D→D)，D = z_e 的 code_dim
- **D-05:** 输入是 CLIP ViT-B/32 的 512 维 text embedding（已有，与 V1 共享）
- **D-06:** 对齐训练目标：KL(projector(clip_text) || z_e_motion.mean())，用训练集 (text, motion) 对训练
- **D-07:** BMM retriever 的检索结果作为 teacher signal（soft target），不直接用 BMM 的权重

### SSTA 维度适配

- **D-08:** 依赖 LA-01 的 z_e 维度验证结果：
  - 如果 code_dim = 512（= clip_dim）→ SSTA 的 info_mlp 输入维度不变，只需改 re_dict 的来源
  - 如果 code_dim ≠ 512 → projector 输出端加一个 Linear(code_dim→512) 对齐到 SSTA 期望的维度
- **D-09:** info_mlp 可能从 3 路输入（text + R_m + R_t）变为 2 路（text + R_latent），因为在 z_e 空间里 motion 和 text 的检索结果合一了。或者保留 3 路结构但 R_m 和 R_t 都来自 z_e 空间。具体方案在 planner 读完代码后确定

### 检索数据库重建

- **D-10:** 新数据库替换 `encoded_motions.npy`：从 Part_TMR 512d → z_e code_dim 维
- **D-11:** 文本侧 `encoded_texts.npy` 改为 projector 输出（训练好 projector 后重建）
- **D-12:** 保留 `motion_ids.npy`、`all_captions.npy` 格式不变
- **D-13:** 数据库重建需要 ~112 秒（23K 样本），可接受

### 训练策略

- **D-14:** 分两阶段：(a) 先训 projector 对齐（独立脚本，快速收敛），(b) 再端到端训 MaskTransformer（用新检索库 + projector）
- **D-15:** 阶段 (a) 超参数：从 Part_TMR 的训练配置参考（200 epochs, batch 128, 单卡）
- **D-16:** 阶段 (b) 超参数：与 V1 MaskTransformer 训练完全一致（2000 epochs, 8 卡 DDP, batch 64）

### 前置条件

- **D-17:** Phase 1 的消融结果可能影响 SSTA 的 R_t 处理方式（ABL-02 如果 K+V 更好，Phase 2 应采用 K+V）
- **D-18:** 需要 V1 预训练 RVQVAE checkpoint（用于提取 z_e）
- **D-19:** 需要 GPU 服务器（训练 projector + 重训 MaskTransformer）

## Deferred Ideas

- 联合训练 VQ-VAE + 对比损失（如果冻结方案的检索 ceiling 不够，Phase 2 结束后评估）
- ArcVQ-VAE codebook 正则化（如果 z_e 各向异性严重）
- Confidence-gated SSTA（独立改进，可在任何 phase 后加入）

## Downstream Notes

- **For researcher:** 需要研究 z_e 的实际维度和分布特征、KL 对齐的训练细节、以及 SSTA info_mlp 适配的具体方案
- **For planner:** 改动涉及 build_rag_database.py、t2m_retriever.py、semantics_modulated.py、新增 train_query_projector.py。CONCERNS.md 标记了维度耦合和 DDP 同步风险
- **For verifier:** 核心验收标准是 HumanML3D FID 不劣于 V1 baseline (0.099)
