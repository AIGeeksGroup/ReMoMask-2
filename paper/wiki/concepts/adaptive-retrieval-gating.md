---
type: concept
related_papers: [[2511.09803]], [[2606.06474]], [[2601.11342]], [[2604.17866]], [[2605.06285]]
---

## 领域现状

标准 RAG 对每条查询都做检索，但不加区分的检索并非总是有益——噪声检索结果反而拉低生成质量。"何时检索"和"是否值得检索"已成为 RAG 系统的核心设计决策。近期 5 篇工作从不同角度提出自适应检索门控机制：

| 方法 | 域 | 门控时机 | 门控信号 | 是否需要训练 |
|---|---|---|---|---|
| **TARG** ([[2511.09803]]) | 文本 QA | 生成前一次 | 前缀 logit 不确定性（entropy/margin/variance） | 否 |
| **SARDI** ([[2606.06474]]) | 文本 QA (DLM) | 每步去噪时 | 双阈值：query $\tau_q$ + commit $\tau_c$ | 否 |
| **SPREAD** ([[2601.11342]]) | 文本 QA (DLM) | 每步去噪时 | query-token 语义相关度（hidden state cosine） | 否 |
| **LAnR** ([[2604.17866]]) | 文本 QA | 每轮生成后 | Retrieval Control Head（$d+1$ 参数 MLP） | 是 |
| **LatentRAG** ([[2605.06285]]) | 文本 QA | 每步 latent 后 | Action token $\alpha_t \in \{\langle\text{query}\rangle, \langle\text{answer}\rangle\}$ | 是 |

## 前人忽略的问题

1. **动作生成领域完全没有检索门控**：ReMoMask、ReMoDiffuse、MoRAG、ReMoGPT、VimoRAG 对每条查询都无条件检索。但 TARG 的实验表明，即使在文本 QA 领域，Always-RAG 的准确率也可能低于 Never-RAG（Qwen2.5-7B 在 NQ-Open 上 Always 37.4 vs Never 38.8），检索反而有害的情况并非罕见。

2. **迭代生成中门控与去噪步耦合的研究空白**：SARDI 和 SPREAD 都在 DLM（离散扩散语言模型）的去噪过程中做逐步门控，这与 ReMoMask 的 10 步迭代去掩码结构高度同构，但目前无人在 masked motion generation 中做类似尝试。

3. **门控信号的跨模态迁移性**：所有 5 篇工作的门控信号都基于 token-level 的 logit/hidden state 不确定性。motion generation 中的不确定性信号形式不同（VQ-VAE codebook 距离、mask ratio 等），是否能沿用同样的门控逻辑需要重新设计。

## 共存的挑战

1. **训练 vs 免训练的权衡**：LAnR 的 Control Head 和 LatentRAG 的 Action Token 需要监督训练（gold 文档覆盖率标签 / SFT 轨迹），但更精准。TARG 和 SARDI 免训练但依赖阈值校准，跨域迁移时阈值需重设（TARG Table S8 跨数据集 EM 下降 5-15 pp）。

2. **单次门控 vs 逐步门控**：TARG 证明在短文 QA 场景中，单次前缀门控优于动态 re-check（re-check 退化为 Always-RAG）。但 SARDI 在 DLM 去噪中逐步门控效果显著（2WikiMQA EM +15.4）。差异在于生成过程的信噪比变化：AR 模型的 logit 分布随生成推进变化不大，DLM 的去噪过程信噪比变化剧烈，更适合逐步重估。

3. **前瞻信号的利用**：SARDI 发现"被丢弃的低置信度预测"反而是最有用的检索信号（$\tau_q=0$ 始终优于 $\tau_q=0.9$，EM 差 4-6 点）。这挑战了"只用高置信预测做检索 query"的直觉。

## 主要解决方法族（按方法分类,不按论文分类）

### 方法族 A：前缀不确定性一次判定

在生成开始前，用少量试探性 token 的 logit 不确定性做"要不要检索"的二值决策。

- **TARG Margin Gate**：贪心解码 $k=20$ 个 token，计算 top-1 与 top-2 logit 差的指数衰减均值 $U_\text{mar}(k;\tau) = \frac{1}{k}\sum \exp(-g_t/\tau)$。阈值 $\theta$ 由开发集 CDF 校准。检索率降至 5-30%，EM 持平或提升。
- **TARG Entropy Gate**：$U_\text{ent}(k) = \frac{1}{k}\sum H_t$，与 Margin Gate 互补但 Margin 一致更优。
- **TARG Variance Gate**：$N=5$ 条随机前缀采样，$d_t = 1 - \max_j \hat{p}_t(j)$。需要多次前向传播，开销更高。

### 方法族 B：去噪过程中逐步门控

在迭代去噪/去掩码的每一步（或每几步）重新评估检索需求。

- **SARDI 双阈值**：query 阈值 $\tau_q$ 控制哪些位置的预测参与检索 query（$\tau_q = 0$ 即全部参与）；commit 阈值 $\tau_c$ 控制哪些 token 提交到最终输出（$\tau_c \in \{0.9, 0.95\}$）。因为 $\tau_q \ll \tau_c$，投机性 token 可在远未确定前就引导检索，不污染输出。
- **SPREAD 语义相关度排序**：用模型前向的 hidden state 计算每个 masked 位置与 query 的余弦相似度 $\text{Rel}(i,q) = \sigma(h_i \cdot h_q / \|h_i\|\|h_q\|)$，优先 unmask 与 query 最相关的位置，抑制语义漂移。开销极小（复用前向 hidden state）。

### 方法族 C：可学习的自适应终止

训练一个轻量模块判断是否继续检索。

- **Control Head**（LAnR）：$d+1$ 参数的 sigmoid 分类器，对 `[PRED]` token 的 hidden state 做二值判断。训练标签由 gold 文档覆盖率自动生成。去掉后 Avg EM 从 0.418 降到 0.350（-16.3%）。
- **Action Token**（LatentRAG）：LLM 解码 $\alpha_t \in \{\langle\text{query}\rangle, \langle\text{answer}\rangle\}$，由 SFT 轨迹监督。本质上让 LLM 自身决定何时停止检索。

## 开放问题 / Gap

1. **motion generation 中门控信号的定义**：文本 QA 的门控信号（logit entropy、hidden state 余弦）不能直接迁移到 motion token。VQ-VAE codebook 的 top-1 与 top-2 距离差（类比 TARG Margin Gate）或 mask ratio 随步数的变化率，可能是 motion 领域的自然门控信号，但未被验证。参见 [[gap-retrieval-quality-aware-fusion]]。

2. **迭代检索门控在 masked generation 中的适配**：SARDI/SPREAD 验证了 DLM 去噪过程中逐步门控的有效性，ReMoMask 的 10 步去掩码过程结构类似，但 motion token 的条件独立性假设（SARDI 用 CMI 验证文本 token 在 RAG 下近似独立）是否成立尚不确定。参见 [[gap-iterative-retrieval-in-motion-generation]]。

3. **门控 + 质量感知融合的联合优化**：当前门控只做"要不要检索"的二值决策，不涉及"检索结果好不好"的质量评估。TARG 的 usefulness calibration 理论（Proposition 1）假设高不确定区域检索必然有益，但实际上即使触发检索，检索结果也可能不相关。将门控与融合权重联合优化是更完整的方案。
