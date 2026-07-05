---
type: gap
seeded_from: [[2508.02605]], [[retrieval-generation-fusion]], [[2511.09803]]
novelty_verified: false
---

## 问题陈述

当前所有 motion RAG 方法的融合层（SSTA、SMA、prompt 拼接、SDEdit）对检索结果一视同仁——不论检索到的动作与 query 的匹配程度如何，都以相同的权重注入生成过程。这导致两类失败：

1. **低质量检索反噬生成**：RADiAnce ([[2510.10480]]) 发现检索质量低时性能下降（其局限 1）。LongLive-RAG ([[2606.02553]]) 在 $K=8$ 时 background consistency 和 imaging quality 反而低于 $K=6$。ReMoMask ([[2508.02605]]) 在 HumanML3D 上 FID 0.099 劣于无检索的 MoMask 0.045，暗示检索噪声可能是原因之一。

2. **不区分"需要检索"和"不需要检索"的 query**：TARG ([[2511.09803]]) 证明 Always-RAG 可能劣于 Never-RAG（Qwen2.5-7B NQ-Open 37.4 vs 38.8），并建立了 usefulness calibration 理论——选择性检索可以同时优于 Always 和 Never 两个 baseline（Proposition 1）。但 motion 领域从未做过类似分析。

3. **检索结果的质量信号被浪费**：ReMoGPT ([[remogpt-aaai2025]]) 的 PL-TMR 检索返回余弦相似度分数，VimoRAG ([[2508.12081]]) 的 Gemini-MVR 有检索打分，RMD ([[2412.04343]]) 有层级阈值 $\tau$，但这些检索质量信号从未传递到融合层来调节注入强度。

## 为什么前人没解决

1. **检索通常被当作预处理步骤**：motion RAG 方法将检索和生成视为严格串行的两阶段，检索完成后其置信度信息就被丢弃，只保留 top-K 的 motion 特征送入融合层。

2. **Baseline 检索质量尚可**：在 HumanML3D 等 in-domain 数据集上，TMR/Part_TMR/BMM 的检索质量较高，低质量检索的比例不大。问题在 OOD 场景或长尾 query 中才凸显。

3. **缺少"检索有害"的量化分析**：ReMoMask FID 0.099 > MoMask 0.045 被归因于其他因素（如 SSTA 参数量），未系统拆分出检索噪声的贡献。VimoRAG 通过 McDPO 间接处理了这一问题，但 McDPO 的改善到底来自"抑制坏检索"还是"整体 DPO 对齐"无法分离。

## 可能的切入点

1. **检索置信度加权融合**：将检索相似度分数 $s(q, m_i)$ 作为 attention 的偏置项注入 SSTA/SMA 的 cross-attention 权重，使高置信检索结果获得更高注意力。具体实现：$\text{Attn}(Q, K, V) = \text{softmax}(QK^\top / \sqrt{d} + \alpha \cdot s) V$，其中 $s$ 是检索相似度向量。

2. **TARG 风格的 motion 门控**：在 VQ-VAE codebook 距离上定义类似 TARG Margin Gate 的不确定性分数——如果 query latent 对 codebook 的 top-1 和 top-2 距离差很小，说明模型对匹配不确定，更需要检索辅助。距离差大则直接生成。

3. **可学习的检索质量评估器**：训练一个轻量 MLP，输入为检索到的 motion latent 和 text query embedding，输出检索质量分数 $r \in [0,1]$。$r$ 调制融合层的权重：$r \to 0$ 时退化为无检索生成，$r \to 1$ 时全力融合。

4. **RAG-CFG 的自适应版本**：ReMoMask 的 RAG-CFG 使用固定 guidance scale $s=4$。可以让 $s$ 依赖于检索质量——高质量检索时 $s$ 增大（更多依赖检索条件），低质量时 $s$ 减小（更多依赖无条件先验）。

## 相关文献

- [[2508.02605]] — ReMoMask SSTA 融合不感知检索质量
- [[2510.10480]] — RADiAnce 发现检索质量差时性能下降
- [[2606.02553]] — LongLive-RAG $K=8$ 过度检索导致质量下降
- [[2511.09803]] — TARG 证明选择性检索优于 Always-RAG
- [[2508.12081]] — VimoRAG McDPO 间接抑制检索噪声
- [[remogpt-aaai2025]] — ReMoGPT $k=2+2$ 不涨反跌
- [[2412.04343]] — RMD 层级阈值做粗粒度检索质量判断
- [[retrieval-generation-fusion]] — 融合机制综述
- [[adaptive-retrieval-gating]] — 自适应检索门控综述
