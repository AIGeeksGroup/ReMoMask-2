---
type: concept
related_papers: [[2508.02605]], [[2304.01116]], [[2409.12140]], [[remogpt-aaai2025]], [[2508.12081]], [[2412.04343]]
---

## 领域现状

Text-to-motion 的 RAG 方法从 2023 年 ReMoDiffuse 首创至今已形成一个方法族，6 篇代表工作覆盖了检索源、检索粒度、融合机制和生成架构四个维度的设计空间：

| 方法 | 检索源 | 检索空间 | 检索粒度 | 融合方式 | 生成架构 | 训练需求 |
|---|---|---|---|---|---|---|
| **ReMoDiffuse** ([[2304.01116]]) | 3D motion 库 | CLIP 文本 + 运动学 | 全身 | SMA cross-attn | Diffusion | 端到端 |
| **MoRAG** ([[2409.12140]]) | 3D motion 库 | TMR 部位级 | 3 部位 | SMA (复用) | Diffusion | 检索器独立训练 |
| **ReMoMask** ([[2508.02605]]) | 3D motion 库 | Part_TMR (BMM) | 5 部位组 + 全身 | SSTA cross-attn | Masked Transformer | 端到端 |
| **ReMoGPT** ([[remogpt-aaai2025]]) | 3D motion 库 | PL-TMR (部位级对比) | 6 部位 | Prompt context 拼接 | T5 autoregressive | 检索器独立 |
| **VimoRAG** ([[2508.12081]]) | 视频库 (425k) | Gemini-MVR (双通道) | 全身 (视频) | LLM prompt + McDPO | Phi3-3.8B AR | 检索器+DPO 分阶段 |
| **RMD** ([[2412.04343]]) | 3D motion 库 | CLIP 文本 | 层级 (全身/半身/6部位) | SDEdit 去噪 | Diffusion (冻结) | 无需训练 |

## 前人忽略的问题

1. **检索空间与生成空间始终分离**：6 种方法全部在独立的检索空间（CLIP/TMR/PL-TMR/InternVideo）中完成检索，再将结果注入另一个空间的生成器。ReMoGPT Table 6 证实检索质量（PL-TMR > TMR > MotionPatches）直接影响生成质量，但检索-生成空间对齐的缺失被所有工作忽略。参见 [[gap-latent-aligned-retrieval-for-motion]]。

2. **全部采用一次性检索**：6 种方法在生成开始前做一次检索，整个生成过程检索条件固定。SARDI ([[2606.06474]]) 和 SPREAD ([[2601.11342]]) 已在文本 DLM 上证明逐步检索的优势，但 motion 领域尚未引入。参见 [[gap-iterative-retrieval-in-motion-generation]]。

3. **检索结果不做质量评估**：所有方法将检索到的动作无条件注入生成器。VimoRAG 通过 McDPO 教 LLM "何时忽略坏的检索"，但这是事后补救而非事前门控。参见 [[gap-retrieval-quality-aware-fusion]]。

4. **检索库规模受限于 3D 动作数据**：VimoRAG 率先突破这一限制（425k 视频 vs 14k 3D 动作），证明视频检索可行但面临跨模态鸿沟。RMD 则证明即使库内没有完全匹配的全身动作，部位级组合也能合成 OOD 运动。

## 共存的挑战

1. **检索对 in-domain 增益有限但对 OOD 关键**：ReMoMask 在 HumanML3D 上 FID 0.099，劣于无检索的 MoMask（0.045），暗示检索噪声可能抵消增益。（注：0.099 是 arXiv 版数字；可信版 pami.tex 报 FID 0.026、全场最佳，「劣于 MoMask」在可信版本中不再成立，本条仅保留 OOD 论证部分，updated 2026-07-05。）但在 OOD 场景（VimoRAG IDEA400 FID 2.388 vs 无 RAG 5.410，RMD Mixamo 泛化实验），RAG 优势显著。检索对长尾/罕见动作最有价值——ReMoGPT 在 top-5% 罕见动作上 balanced MMDist 从 4.161 降至 3.056。

2. **融合机制的计算开销**：cross-attention 融合（SMA/SSTA）引入大量额外参数和计算。ReMoMask 的 SSTA 2D attention 使总参数达 238M，ReMoDiffuse 需要 4 种条件组合的 grid search。RMD 选择冻结 diffusion 模型 + SDEdit 绕过了训练开销，但推理时 $t_0$ 需手动调节。

3. **检索候选数量的甜区很窄**：ReMoGPT 发现 $k=1+1$ 最优（text-to-motion + text-to-text 各取 1），$k=2+2$ 反而略降——T5 对超长 motion token prompt 处理能力有限。LongLive-RAG ([[2606.02553]]) 同样发现 $K=8$ 时质量下降。

## 主要解决方法族（按方法分类,不按论文分类）

### 方法族 A：学习式检索 + 端到端融合

检索器和生成器联合训练或分阶段训练，检索结果通过 cross-attention 注入。

- **全身级检索 + SMA**（ReMoDiffuse）：CLIP text-to-text + 运动学相似度混合检索，$K=2$，通过 Condition Mixture 4 种权重组合做 CFG。
- **部位级检索 + SSTA**（ReMoMask）：BMM 动量对比学习（65536 负样本队列），HBM 5 部位组编码，SSTA 2D cross-attention + RAG-CFG（$s=4$）。
- **部位级检索 + SMA**（MoRAG）：LLM 分解文本 → 3 个独立 TMR 检索 → 关节空间组合 → 送入 ReMoDiffuse pipeline。

### 方法族 B：检索 + prompt context 注入

检索结果以 token 序列形式拼入语言模型的 prompt。

- **PL-TMR + T5 instruction tuning**（ReMoGPT）：6 部位 Part-Level TMR 做双模态检索（T2M + T2T），$k=1+1$。检索到的 motion-caption 对作为 `### Context` 拼入 T5 prompt。
- **视频 + LLM + McDPO**（VimoRAG）：Gemini-MVR 双通道检索 rank-1 视频，文本 + 视频一同输入 Phi3-3.8B。McDPO 基于偏好学习抑制检索噪声传播。

### 方法族 C：免训练检索 + diffusion refinement

冻结预训练生成器，检索结果直接注入生成过程（无需额外训练）。

- **层级分解 + SDEdit**（RMD）：LLM 将文本按全身/半身/6 部位分解，CLIP 检索 + SLERP 组合，SDEdit 从中间 $t_0=0.96$ 去噪精炼。$k=5$ 次 LLM 分解 + 1 次 LLM 选择。越 OOD 越依赖细粒度分解（HumanML3D 37.8% 全身 vs Mixamo 0% 全身）。

## 开放问题 / Gap

1. **检索空间统一到生成空间**：6 种方法无一将检索空间与生成空间对齐。ReMoMask V2 Plan A 的核心提议——在 VQ-VAE latent space 内同时做检索和生成——如果成功，将是该方法族的范式突破。参见 [[gap-latent-aligned-retrieval-for-motion]]。（updated 2026-07-05：已实现为 ReMoMask-2——检索库建在冻结 VQ-VAE 的预量化 z_e 空间（66,912 对，1024 维），文本 query 经 KL 蒸馏 QueryProjector（BMM teacher，top-256，τ=0.07）投入同一空间，SSTA 仅加 Linear(1024→512) 适配；中间结果 V2 FID 0.0991@ep409 vs V1 0.1273@ep316，MaskTransformer 单独、**非论文数字**。）

2. **迭代检索在 motion 中的首次实现**：ReMoMask 的 10 步去掩码天然适合嵌入逐步检索，但技术前提是检索空间能接受部分生成状态作为 query。参见 [[gap-iterative-retrieval-in-motion-generation]]。（updated 2026-07-05：Plan B 已搁置，论文最多在 future work 提一句。）

3. **检索增强 vs 数据扩展的帕累托前沿**：VimoRAG 用 425k 视频扩展检索库，RMD 用部位组合合成 OOD 动作。两条路径是否可以叠加？一条 OOD 动作既有视频参考又有部位级组合时，融合策略如何设计？

4. **长序列和组合动作的检索粒度**：ReMoGPT 对"先走再蹲再转"只做一次全局检索，无法为各段子动作分别找参考。RMD 按身体部位分解但不按时间分解。时空联合的分段检索未被探索。

5. **VQ-VAE tokenizer 限制的根本瓶颈**：ReMoGPT 明确指出（Table 8），yoga 等全新动作域即使检索到了参考，VQ-VAE decoder 也无法重建分布外动作。检索增强的上限由 tokenizer 的重建能力决定。
