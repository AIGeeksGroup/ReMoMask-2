---
type: note
source: D:\tpami\learning-records\0003-v2-plan-a-confirmed.md
date: 2026-06-26
---

## V2 方案确认：先 Plan A，再 Plan B

用户确认采用 A+B 组合方案，优先实现 Plan A。

### Plan A — Latent-Aligned Retrieval（主推）
将检索库从 Part_TMR 对比空间迁移到 VQ-VAE latent space，使检索结果和生成器在同一空间。对标 RADiAnce 的共享 contrastive space 思路。

改动量约 200 行，4-5 个文件。VQ-VAE / Residual Transformer / 评估流程不动。

### Plan B — 迭代动态检索（后续补充）
在 10 步去掩码循环中，每隔 N 步用已确定 token 更新检索 query。对标 LongLive-RAG 的逐步检索。

### 论文 Story
两个 contribution 正交，支撑 2x2 消融矩阵：
- C1: Latent-aligned retrieval（空间统一）
- C2: Dynamic iterative retrieval（时间动态）

### 更新（2026-07-05）

- Plan A 已实现：检索库建在冻结 VQ-VAE 的**预量化 z_e 空间**（code_dim2d=1024，66,912 对）；文本 query 经 QueryProjector（Linear 512→1024 + GELU + Linear 1024→1024，KL 蒸馏自 BMM teacher top-256，τ=0.07）投入同一空间；SSTA 不变，仅加 Linear(1024→512) 适配层。
- **Plan B 搁置**，论文最多在 future work 提一句。论文 story 相应改为：C1 = latent-aligned retrieval（主贡献），C2 = **rt_in_value**（R_t 进 SSTA Value，独立 minor contribution，「前提变化」叙事）；2x2 消融矩阵变为 {Part_TMR, z_e} × {rt_in_value on/off}。
- 训练对照进行中：V2 best FID 0.0991@ep409 vs V1 0.1273@ep316（V1 job 13846 / V2 job 13845；MaskTransformer 单独的中间数字，**非论文数字**，论文数字待 eval_res.py × 20 repeats）。
- 权威事实见 `.planning/paper-v2/PLAN-A-FACTS.md`。

### 相关
- [[gap-latent-aligned-retrieval-for-motion]]
- [[gap-iterative-retrieval-in-motion-generation]]
