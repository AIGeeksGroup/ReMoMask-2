---
type: concept
related_papers: [[2508.02605]], [[2304.01116]], [[2409.12140]], [[2510.10480]], [[2606.02553]], [[remogpt-aaai2025]], [[2508.12081]], [[2412.04343]], [[2401.02142]]
---

## 领域现状

检索增强生成的核心问题之一是：检索到的信息如何注入生成过程？不同于纯文本 RAG 可以直接拼接检索文档到 prompt，结构化生成（动作序列、蛋白结构、视频帧）需要在连续空间中融合检索条件与生成状态。五篇论文给出了五种不同的融合机制：

| 方法 | 融合方式 | 生成架构 | Q/K/V 设计 |
|---|---|---|---|
| **ReMoDiffuse** ([[2304.01116]]) | SMA cross-attention | Diffusion Transformer | Q=噪声动作, K=[self;prompt;$R_m$;$R_t$], V=[self;prompt;$R_m$] |
| **ReMoMask** ([[2508.02605]]) | SSTA cross-attention | Masked Transformer | Q=motion, K=concat($z$,$t$,[$R_m$;$R_t$]), V=concat($z$,$t$,$R_m$) |
| **MoRAG** ([[2409.12140]]) | SMA (复用 ReMoDiffuse) | Diffusion Transformer | 同 ReMoDiffuse，但 $R_t$ 用输入文本特征替代 |
| **RADiAnce** ([[2510.10480]]) | Cross-attention in LDM | Latent Diffusion (EPT) | Q=去噪状态 $H$, K=$T_v$ (检索模板), V=$T_v$ |
| **LongLive-RAG** ([[2606.02553]]) | 注意力上下文拼接 | AR Diffusion | 将检索 latent 直接拼入注意力的 KV 序列 |
| **ReMoGPT** ([[remogpt-aaai2025]]) | Prompt context 拼接 | T5 autoregressive | 检索 motion-caption 对作为 `### Context` 拼入 prompt |
| **VimoRAG** ([[2508.12081]]) | LLM prompt + McDPO | Phi3-3.8B AR | 文本+视频拼入 LLM，DPO 抑制噪声传播 |
| **RMD** ([[2412.04343]]) | SDEdit 去噪精炼 | Diffusion (冻结) | 检索组合后加噪到 $t_0$，DDIM 去噪 |
| **GUESS** ([[2401.02142]]) | 动态多条件融合 | 级联 Latent Diffusion | Channel-wise + cross-modal attention 加权 |

## 前人忽略的问题

1. **检索文本特征 $R_t$ 的角色不清晰**：ReMoDiffuse 和 ReMoMask 都在 K 中加入检索文本特征 $R_t$ 但 V 中不含 $R_t$——即 $R_t$ 只参与注意力权重计算但不直接贡献输出值。这一设计的消融未被充分讨论。MoRAG 因空间组合后的动作没有配对文本，被迫用输入文本特征替代 $R_t$，间接暴露了 $R_t$ 角色的模糊性。
2. **融合位置的选择**：RADiAnce 的 cross-attention 置于 self-attention 和 FFN 之间，比较了 cross-attention、AdaLN-Zero 和 in-context 三种方式，发现 cross-attention 整体最优但 ISM 指标（交互模式恢复）反而略低（71.64% vs AdaLN-Zero 的 73.69%）。动作生成领域未做类似系统比较。
3. **无检索条件的 baseline 不统一**：ReMoDiffuse 的无检索 baseline FID = 0.245，加检索后 0.155（改善 36.7%）；ReMoMask 消融中去掉 BMM 后 Top1 R-Precision 降 16.2%，FID 降 50.18%。但二者的「去掉检索」含义不同（一个去掉整个检索分支，一个只去掉 BMM 换回 baseline 检索器）。

## 共存的挑战

1. **检索噪声的放大**：低质量检索结果通过 cross-attention 注入后可能误导生成。LongLive-RAG 的消融表明 $K=8$（检索过多）时 background consistency 和 imaging quality 反而下降（93.07/60.02 vs $K=6$ 的最优值），说明过度融合有害。
2. **计算开销**：ReMoDiffuse 将检索动作下采样到 1/4 FPS 以降低 cross-attention 开销；ReMoMask 的 SSTA 2D attention 使参数量达 238M；LongLive-RAG 限制总注意力窗口为 12（1 sink + 6 retrieved + 5 local）。融合机制的效率是实际部署的瓶颈。
3. **CFG 与检索条件的交互**：ReMoDiffuse 需要 4 种条件组合（$w_1 \sim w_4$）的 grid search 来平衡检索条件和文本条件；ReMoMask 定义了 RAG-CFG（guidance scale $s=4$，10% 无条件采样）。多条件 CFG 的设计空间快速膨胀。

## 主要解决方法族（按方法分类,不按论文分类）

### 方法族 A：Cross-Attention 融合

**核心思路**：生成状态做 Query，检索结果做 Key/Value（或仅做 Key），通过 attention 机制选择性吸收检索信息。

- **SMA / SSTA**（ReMoDiffuse、ReMoMask、MoRAG）：将噪声动作/motion token 作为 Q，文本条件、自身特征和检索特征拼接为 K/V。关键设计是 $R_t$（检索文本特征）仅在 K 中出现，V 中只有 $R_m$（检索动作特征）。ReMoDiffuse 使用 Linear Attention 提高效率。ReMoMask 将 1D token map 扩展为 2D（$T \times J$），在时间和空间两个维度上建模。
- **EPT cross-attention**（RADiAnce）：E(3)-等变 Transformer 中的 cross-attention 层，检索模板 $T_v$ 同时做 K 和 V。置于 self-attention 和 FFN 之间。

### 方法族 B：注意力上下文直接拼接

- **KV 序列扩展**（LongLive-RAG）：不引入新的注意力层，直接将检索到的历史 latent 拼接到注意力的 KV 序列中：$A_t = [C_{\text{sink}} \| M_t \| C_{\text{loc}}]$。优势是不修改 base generator 架构（冻结权重），劣势是缺乏选择性——所有 token 平等参与注意力。

### 方法族 C：空间组合后融合

- **部位级拼接 + SMA**（MoRAG）：先将三个部位的检索结果按 SMPL 关节分割重组为全身动作，再送入 SMA 做条件融合。本质是两阶段融合——先在关节空间组合，再在注意力空间融合。组合阶段裁剪至最短序列长度 $f_{\min}$，可能引入时序不一致。

### 方法族 D：Prompt Context 注入

检索结果以 token 序列直接拼入语言模型 prompt，无需修改模型架构。

- **Motion-caption prompt**（ReMoGPT）：检索到的 motion VQ-VAE token 序列和配对 caption 作为 `### Context` 拼入 T5 输入。$k=1+1$ 最优，$k=2+2$ 反而略降——T5 对超长 motion token prompt 处理能力有限。
- **视频 + 文本 prompt + McDPO**（VimoRAG）：检索 rank-1 视频与文本拼接输入 Phi3-3.8B。McDPO 通过 dual-alignment 奖励模型（$w_\ell=0.9$ motion feature + $w_d=0.1$ semantic）构建偏好数据做 DPO 训练，教 LLM 区分有效和无效检索。

### 方法族 E：SDEdit 精炼

冻结预训练 diffusion 模型，将检索组合结果直接作为 SDEdit 初始化。

- **层级分解 + SDEdit**（RMD）：检索到的部位动作经 SLERP 组合后加噪到中间步 $t_0=0.96$，DDIM 去噪精炼。$t_0$ 控制检索 vs 生成的权衡——$t_0$ 小保留更多检索信息，$t_0$ 大让 diffusion prior 主导。无需训练但 $t_0$ 需手动调节。

### 方法族 F：动态多条件融合

检索/粗生成条件与文本条件的权重随生成阶段自适应变化。

- **Channel-wise + cross-modal attention**（GUESS）：在级联 diffusion 的每级 denoiser 中，文本条件 $c$ 和粗动作条件 $z_{i+1}$ 的融合权重 $[w_z, w_c]$ 由 Softmax 门控网络推断，且与去噪时间步相关。实验发现粗阶段动作权重高、细阶段文本权重回升。

### 方法族 G：多条件 Guidance

- **Condition Mixture**（ReMoDiffuse）：四种条件组合的加权混合 $\hat{S} = w_1 S(\text{retr},\text{text}) + w_2 S(\text{text}) + w_3 S(\text{retr}) + w_4 S(\emptyset)$，$w_4=0$。前 40 步 grid search + 后 10 步 Adam 优化。
- **RAG-CFG**（ReMoMask）：$(1+s) \cdot \text{logits}_{\text{con}} - s \cdot \text{logits}_{\text{un}}$，$s=4$，训练时 10% 无条件采样。比 Condition Mixture 更简洁但灵活性较低。

## 开放问题 / Gap

1. **检索信息的渐进注入 vs 一次性注入**：所有动作生成方法都在去噪/解码初始就注入全部检索信息。LongLive-RAG 在每个新 block 按需检索，但仍是一次性注入到该 block 的注意力中。是否应在不同去噪步/解码层注入不同粒度的检索信息？
2. **Fusion 机制与检索空间的解耦**：如果检索从独立空间迁移到生成 latent 空间（参见 [[latent-space-retrieval-alignment]]），融合机制是否可以简化？RADiAnce 的实践表明，检索和生成共享空间后，简单的 cross-attention 即可有效融合。
3. **检索置信度感知的融合**：当前融合机制不区分检索结果的可靠程度。RADiAnce 发现检索质量低时性能下降（其局限 1），但融合层并未根据检索相似度调整注意力权重。VimoRAG 的 McDPO 是事后补救而非事前门控，GUESS 的动态融合权重只区分文本 vs 动作条件而非检索质量好坏。参见 [[gap-retrieval-quality-aware-fusion]]。
4. **SDEdit 精炼 vs 端到端融合的权衡**：RMD 证明冻结模型 + SDEdit 可以免训练地利用检索结果，但 $t_0$ 超参数敏感（Table 7 检索-训练分布不匹配时最优 $t_0$ 从 0.96 降到 0.8）。端到端方法（SMA/SSTA）更灵活但需要训练。两类方法的系统比较缺失。
5. **GUESS 动态融合权重的可迁移性**：GUESS 发现文本 vs 粗动作条件的最优权重比随生成阶段变化（Figure 11）。如果将"粗动作条件"替换为"检索动作条件"，这种动态权重策略是否适用于 RAG 融合——早期检索权重高、后期文本权重高？
