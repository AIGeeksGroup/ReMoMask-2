---
type: concept
related_papers: [[2508.02605]], [[2304.01116]], [[2409.12140]], [[2604.17866]], [[2605.06285]], [[2606.02553]], [[2606.06474]], [[2601.11342]], [[2511.09803]]
---

## 领域现状

检索增强生成中，「何时检索、检索几次」是一个核心但常被忽略的设计决策。当前方法可分为三大类：

### 一次性检索（One-Shot Retrieval）

在生成开始前做一次检索，生成全程使用固定的检索结果。

- **ReMoMask** ([[2508.02605]])：BMM 检索器在生成前取回 top-K 动作和文本特征，整个 masked transformer 解码过程中检索条件不变。
- **ReMoDiffuse** ([[2304.01116]])：混合检索（CLIP 语义 + 动作长度运动学相似度）在扩散去噪前完成，预处理所有检索特征以避免重复计算。1000 步扩散、50 步推理中检索条件固定。
- **MoRAG** ([[2409.12140]])：LLM 分解文本 → 部位级检索 → 空间组合 → 送入扩散模型。检索在扩散开始前一次性完成。
- **RADiAnce** ([[2510.10480]])：编码结合位点 → 内积检索 top-$K$ 模板 → 100 步条件反向扩散。检索在扩散前完成。

### 多轮迭代检索（Multi-Turn Iterative Retrieval）

在生成过程中多次检索，每轮检索利用前一轮的生成结果更新查询。

- **LAnR** ([[2604.17866]])：每轮 $r$ 用 `[PRED]` token 的 hidden state 构造新查询 $q^{(r)}$，上下文包含前面所有已检索文档。Retrieval Control Head（仅 $d+1$ 个参数）自适应决定是否继续检索。消融表明去掉 control head 后 Avg EM 从 0.418 降到 0.350（$\Delta = -0.068$）。平均检索调用次数 1.54--2.48 次。
- **LatentRAG** ([[2605.06285]])：latent token 序列替代显式 subquery，LLM 根据最后一个 latent thought token 解码 action token $\alpha_t \in \{\langle\text{query}\rangle, \langle\text{answer}\rangle\}$ 决定继续检索还是生成答案。与 LAnR 类似的自适应终止，但用 latent token 而非文本 query。

### 去噪过程中的动态检索（Denoising-Time Dynamic Retrieval）

在 DLM（离散扩散语言模型）的迭代去噪过程中，逐步刷新检索。

- **SARDI** ([[2606.06474]])：每步去噪时，用低置信度预测（$\tau_q = 0$，所有位置参与）构造 proxy 序列拼接问题做检索 query，完全替换上一步的 context。双阈值设计——$\tau_q \ll \tau_c$——使投机性 token 可在远未确定前就引导检索。2WikiMQA EM 从 43.7 提升到 59.1（+15.4）。每 2 步刷新仅损失 1-2 EM，83-90% 文档跨步保持不变。
- **SPREAD** ([[2601.11342]])：不刷新检索，而是改进去噪策略——用 query-token 语义相关度（hidden state cosine）取代 confidence 来排序 unmask 优先级，将去噪轨迹锚定在 query 语义上，抑制 Response Semantic Drift。Dream 上 Precision 平均绝对提升 +30.90%。开销极小（复用前向 hidden state）。

### 自适应检索门控（Adaptive Retrieval Gating）

在生成前或生成中判断"是否值得检索"。

- **TARG** ([[2511.09803]])：生成前贪心解码 $k=20$ 个 token，用 logit 不确定性（entropy/margin/variance）做二值门控。Margin gate 检索率降至 5-30%，EM 持平或提升。CDF-based 阈值校准。无需训练。

### 逐步渐进检索（Progressive Retrieval）

在自回归生成的每个新步/block 中按需检索。

- **LongLive-RAG** ([[2606.02553]])：每生成一个新视频 block，用最近完成 block 的 embedding $v_{t-1}$ 作为 query 检索历史 latent。设有 recency guard $R=5$，跳过最近 $R$ 个 block。检索预算 $K=6$，120s rollout 全程额外开销仅 490 ms。

## 前人忽略的问题

1. **动作生成领域完全停留在一次性检索阶段**：ReMoMask、ReMoDiffuse、MoRAG 均在生成开始前做一次检索。扩散去噪过程中，随着噪声逐步减少，生成结果的语义逐渐清晰——此时理应能构造更精确的检索查询，但无任何工作利用这一信号。
2. **检索预算的固定 vs 自适应**：所有动作生成方法使用固定的检索数量（ReMoDiffuse $k=2$，ReMoMask 未明确报告但消融中对比了全局 vs 局部）。LAnR 和 LatentRAG 的 control head / action token 提供了自适应终止的先例，但未被引入动作生成。LongLive-RAG 同样使用固定 $K=6$，其消融表明 $K=8$ 时质量下降——自适应 $K$ 可能更优。
3. **检索与去噪步的耦合**：在 1000 步扩散过程中，早期步（高噪声）和晚期步（低噪声）对检索信息的需求可能不同——早期需要粗粒度语义指导，晚期需要细粒度结构参考。当前方法在所有步使用相同的检索条件。

## 共存的挑战

1. **延迟 vs 质量的权衡**：多轮检索增加延迟。LatentRAG 通过将检索查询从显式文本（Search-R1 平均 163 个 output token）压缩为 latent token（仅 5 个 token），将延迟从 5372 ms 降到 593 ms（-89%），同时性能略有提升。但动作生成中是否存在类似的 latent 压缩路径尚不确定。
2. **查询漂移**：多轮检索中，每轮查询基于前一轮结果更新，错误可能累积。LAnR 通过 Adaptive Contrastive Target（每轮只用剩余未检索的 gold 文档作正例）缓解此问题，但需要训练标签支持。
3. **历史库管理**：LongLive-RAG 逐步将已生成 block 加入历史库，需要管理库的增长和检索效率。动作生成中如果在去噪过程中检索，检索库内容不变（是训练集而非自身历史），问题结构不同。

## 主要解决方法族（按方法分类,不按论文分类）

### 方法族 A：固定预检索（Pre-Generation Retrieval）

所有动作 RAG 方法（ReMoMask、ReMoDiffuse、MoRAG）和 RADiAnce 属于此类。优点是简单高效，检索成本不随生成步数增长。缺点是无法利用生成过程中逐步清晰的信息来改进检索。

### 方法族 B：可学习的自适应终止

- **Control Head**（LAnR）：极轻量 MLP（$d+1$ 参数），对 `[PRED]` token 做 sigmoid 判断是否继续检索。训练标签由 gold 文档覆盖率自动生成。
- **Action Token**（LatentRAG）：LLM 解码一个 $\alpha_t \in \{\langle\text{query}\rangle, \langle\text{answer}\rangle\}$，由 SFT 轨迹监督。

### 方法族 C：按生成步渐进检索

- **Block 级检索**（LongLive-RAG）：每个新 block 用前一 block 的 embedding 检索。引入 recency guard 避免冗余。设计简洁但无自适应机制。

## 开放问题 / Gap

1. **扩散去噪过程中的迭代检索**：能否在去噪的不同阶段（如 $t=1000$ 时粗检索、$t=100$ 时细检索）动态更新检索条件？这要求检索空间能接受不同噪声水平的查询——如果使用 latent-aligned retrieval（参见 [[latent-space-retrieval-alignment]]），去噪中间态本身就在生成 latent 空间中，可以直接作为检索查询。参见 [[gap-iterative-retrieval-in-motion-generation]]。
2. **Masked modeling 中的渐进检索**：ReMoMask 使用 masked transformer 而非扩散——mask ratio 逐步降低的过程类似于去噪。是否可以在每轮 unmask 后用当前已解码的 token 作为新查询检索？
3. **检索与 CFG 的联合动态调整**：ReMoDiffuse 的 Condition Mixture 权重 $w_1 \sim w_3$ 是全程固定的。如果检索在不同步动态更新，guidance scale 是否也应随之调整？
4. **DLM 去噪中的动态检索已被验证，masked generation 尚未引入**：SARDI 和 SPREAD 在 DLM 上证明了逐步检索/锚定的有效性。ReMoMask 的 10 步迭代去掩码与 DLM 去噪结构高度同构——mask ratio 逐步降低类比噪声逐步减少——但无人在 masked motion generation 中做类似尝试。参见 [[gap-iterative-retrieval-in-motion-generation]]。
5. **检索门控在 motion 中的缺失**：TARG 证明不加区分的 Always-RAG 可能劣于 Never-RAG。ReMoMask、ReMoDiffuse、MoRAG 对每条 query 都无条件检索。motion 领域是否也存在"检索有害"的 query？参见 [[adaptive-retrieval-gating]]。
