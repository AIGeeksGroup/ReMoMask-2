---
type: gap
seeded_from: [[2508.02605]], [[2304.01116]], [[retrieval-timing-and-dynamics]], [[2606.06474]], [[2601.11342]], [[2511.09803]]
novelty_verified: false
---

## 问题陈述

所有现有 text-to-motion RAG 方法（ReMoMask、ReMoDiffuse、MoRAG）都采用一次性检索：在生成开始前检索一次，整个生成过程使用固定的检索条件。然而生成过程（扩散去噪或 masked token 解码）是渐进细化的——噪声逐步减少、mask 逐步消除——生成中间态包含越来越丰富的语义信息，但这些信息从未被用于更新检索查询。

在其他领域，迭代/自适应检索已被证明有效：
- LAnR ([[2604.17866]])：multi-turn 检索，control head 自适应终止。去掉 control head 后 Avg EM 从 0.418 降到 0.350（-16.3%）。
- LatentRAG ([[2605.06285]])：multi-step latent 检索，action token 决定继续还是生成。平均检索调用次数远少于 baseline（LatentRAG 约 2 次 vs Search-R1 约 3 次），延迟降 89%。
- LongLive-RAG ([[2606.02553]])：逐 block 检索，每个新 block 用前一 block 的 embedding 做 query。120s rollout 额外开销仅 490 ms。

**核心问题**：能否在动作生成的去噪/解码过程中动态更新检索条件，使检索与生成互相强化？

### 新增支撑证据（2025-2026 新论文）

DLM（离散扩散语言模型）领域已出现三项与此 gap 直接相关的工作，证明迭代去噪过程中的动态检索是可行且有效的：

- **SARDI** ([[2606.06474]])：在 DLM 去噪过程中每步刷新检索。核心创新是双阈值分离（$\tau_q \ll \tau_c$）——低置信度预测可在远未确定前就引导检索，不污染最终输出。2WikiMQA EM 从 43.7 提升到 59.1（+15.4），增益集中在 multi-hop 题型（compositional +28.7, inference +23.5, 1-hop -1.0）。每 2 步刷新仅损失 1-2 EM。RAG 提供的 context 大幅降低相邻 token 间条件互信息（有 gold 文档 0.060 vs 无 gold 0.588），证明 RAG 天然促进并行解码——这一发现可类比到 motion token 的条件独立性。
- **SPREAD** ([[2601.11342]])：不刷新检索但改进去噪策略——用 query-token 语义相关度排序 unmask 优先级，抑制 Response Semantic Drift。Dream 上 Precision 平均绝对提升 +30.90%，RSD 降低 61.18%。Motion 版本：在每步去掩码时用 text query embedding 与 masked motion token 的 hidden state 计算 cosine similarity，优先 unmask 与文本最相关的 motion token。
- **TARG** ([[2511.09803]])：提出"检索是否值得"的形式化框架——前缀 logit 不确定性做门控，检索率降至 5-30% 同时 EM 持平或提升。其 re-check 机制（每 $m$ token 重新评估）在短文 QA 退化为 Always-RAG，但 motion 生成的 10 步迭代信噪比变化大，re-check 可能有效。CDF-based 阈值校准和 $\Delta$-latency 报告方法可直接用于 Plan B 效率评估。

## 为什么前人没解决

1. **检索空间与生成空间分离**：当前动作 RAG 的检索在 CLIP/TMR 空间完成，去噪中间态在 diffusion/VQ-VAE 空间，二者无法直接对应——去噪中间态无法直接作为 CLIP/TMR 空间的查询。如果先解决 latent-aligned retrieval（参见 [[gap-latent-aligned-retrieval-for-motion]]），则去噪中间态天然可作为检索 query。
2. **计算开销顾虑**：ReMoDiffuse 1000 步扩散中每步都检索的开销巨大。但 LongLive-RAG 证明检索开销可以极低（10 ms 的 top-K search），且不必每步都检索——可以每 N 步检索一次。
3. **动作序列短**：HumanML3D 的动作序列大多 <100 帧，生成时间短，一次性检索的信息足够。但 ReMoMask 局限 2 指出未验证长序列（如舞蹈），长序列中迭代检索的收益可能更明显。
4. **缺少自适应终止的先例**：LAnR 的 control head 和 LatentRAG 的 action token 都是 NLP 领域的创新，动作生成社区尚未引入类似机制。

## 可能的切入点

1. **去噪阶段分段检索**：将扩散去噪过程分为 2--3 个阶段（如 $t \in [T, T/2]$ 粗阶段、$t \in [T/2, 0]$ 细阶段），每个阶段用当前去噪中间态做一次检索更新。这要求检索空间能接受不同噪声水平的输入——latent-aligned retrieval 是前提。
2. **Masked modeling 的渐进检索**：ReMoMask 的 masked transformer 逐步 unmask token。每轮 unmask 后，已解码的 token 携带了更多局部语义信息，可构造更精确的检索 query。类比 LAnR 的 multi-turn 框架：第一轮用文本 query 粗检索，第二轮用已解码 token + 文本联合查询精检索。
3. **轻量 control head 决定是否更新检索**：仿照 LAnR 的 retrieval control head（仅 $d+1$ 参数），在生成网络中加一个极小的判断模块，根据当前生成状态决定是否值得重新检索。这避免了不必要的检索开销。
4. **检索结果的渐进融合权重**：即使不重新检索，也可以在去噪过程中动态调整检索条件的融合权重——早期步给检索条件高权重（粗指导），晚期步降低权重（让生成模型自主细化）。

## 相关文献

- [[2508.02605]] — ReMoMask，一次性检索 + masked transformer
- [[2304.01116]] — ReMoDiffuse，一次性混合检索 + 1000 步扩散
- [[2409.12140]] — MoRAG，一次性部位级检索 + 扩散
- [[2604.17866]] — LAnR，multi-turn 检索 + control head 自适应终止
- [[2605.06285]] — LatentRAG，multi-step latent 检索 + action token
- [[2606.02553]] — LongLive-RAG，逐 block 渐进检索
- [[retrieval-timing-and-dynamics]] — 检索时机综述
- [[gap-latent-aligned-retrieval-for-motion]] — latent-aligned retrieval 是迭代检索的技术前提
- [[2606.06474]] — SARDI，DLM 去噪过程中每步刷新检索（双阈值设计）
- [[2601.11342]] — SPREAD，query-token 语义相关度引导 unmask 顺序
- [[2511.09803]] — TARG，前缀不确定性门控 + re-check 机制
- [[adaptive-retrieval-gating]] — 自适应检索门控综述
