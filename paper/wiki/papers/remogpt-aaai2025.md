---
title: "ReMoGPT: Part-Level Retrieval-Augmented Motion-Language Models"
authors: Qing Yu et al.
venue: AAAI 2025
tags: [RAG, text-to-motion, part-level-retrieval, motion-language-model, instruction-tuning, VQ-VAE, contrastive-learning]
status: compiled
compiled_at: 2026-06-27
source_path: raw/tpami/remogpt-aaai2025.pdf
---

## 一句话总结

把人体分成 6 个部位分别编码做细粒度 text-motion 对比检索 (PL-TMR)，取回的 motion-caption 对以 prompt context 形式喂给基于 T5 的 motion-language 模型做 instruction tuning，统一解决动作生成、描述和检索三类任务。

## 解决什么问题

1. 已有 motion-language 模型（MotionGPT）在罕见或非典型文本描述上生成质量显著下降。
2. ReMoDiffuse 的检索只靠 CLIP text-to-text 相似度，因为同一动作可对应差异很大的文本描述（如 "walking forward briskly" 和 "plants four steps leading with its left foot"），text-to-text 检索经常找错。
3. ReMoDiffuse 是纯扩散模型，依赖 classifier-free guidance 尺度且不能做 motion captioning，应用范围有限。

## 核心方法

### 1. 部位级文本-动作检索 (PL-TMR)

将全身动作按 SMPL 运动学树分为 6 个部位：Right Arm、Left Arm、Right Leg、Left Leg、Backbone、Root（轨迹）。

- **Part-Level Motion Encoder**：每个部位 $p$ 各用一个轻量 4 层 Transformer $F^M_p$ 编码部位动作 $m_p$，再拼接全部部位 embedding：
  $$F^M(m) = \text{Concat}[F^M_1(m_1), \ldots, F^M_P(m_P)]$$
- **Text Encoder** $F^T$：编码 caption $t$。
- 两路 embedding 经各自 projection head 投影到共享 latent space（$\hat{F}^M(m)$, $\hat{F}^T(t)$），用 cosine 相似度：
  $$s_{m\text{-}t} = \frac{\hat{F}^M(m) \cdot \hat{F}^T(t)}{\|\hat{F}^M(m)\| \|\hat{F}^T(t)\|}$$
- 训练用 $s_{m\text{-}t}$ 和 $s_{t\text{-}m}$ 的对比损失。
- 参数量 118M，小于 MotionPatches (152M)，因为各部位编码器更轻量。

### 2. 多模态检索策略

同时用两种相似度排序检索数据库（= 训练集全体）：
- **Motion generation**：text-to-motion ($s_{t\text{-}m}$) + text-to-text ($s_{t\text{-}t}$)，各取 top-$k$。
- **Motion captioning**：motion-to-text ($s_{m\text{-}t}$) + motion-to-motion ($s_{m\text{-}m}$)，各取 top-$k$。

实验表明 $k=1$（总共 2 条检索样本）已足够，$k=2$ 增益有限，$k=2+2$ 反而略降——作者推测 T5 在有限数据量下难以有效处理过长的 motion token 序列。

### 3. Instruction Tuning

- 基础模型：T5（encoder-decoder 各 12 层），motion VQ-VAE（codebook size 512，下采样率 $l=4$）。
- 将检索到的 motion-caption 对作为 `### Context` 拼入 prompt（见 Fig. 5），motion 以 VQ-VAE token 序列表示。
- 训练目标：
  $$\mathcal{L}_{\text{LM}} = -\sum_{i=0}^{L_t-1} \log p_\theta\!\left(x^i_{\text{out}} \mid x^{<i}_{\text{out}},\, x_{\text{in}},\, x_{\text{rag}}\right)$$
  其中 $x_{\text{rag}}$ 为检索样本 token。
- 训练超参：AdamW，lr $10^{-4}$，batch 16，200 epochs，8×A100。

## 关键实验结果

### Text-Motion Retrieval (HumanML3D, Table 1)

| Method | #Params | T2M R@1 | T2M R@10 | T2M MedR |
|---|---|---|---|---|
| TMR | 82M | 8.92 | 33.37 | 25.00 |
| MotionPatches | 152M | 10.80 | 38.02 | 19.00 |
| **PL-TMR** | **118M** | **11.00** | **43.43** | **14.00** |

### Text-to-Motion Generation (HumanML3D, Table 2)

| Method | R-Top1 | FID | MMDist | Diversity | MModality |
|---|---|---|---|---|---|
| ReMoDiffuse | 0.492 | 0.137 | 3.091 | 9.208 | 1.755 |
| MotionGPT (retrained) | 0.431 | 0.361 | 3.613 | 9.410 | 2.601 |
| **ReMoGPT** | **0.501** | **0.205** | **2.929** | **9.763** | **2.816** |

### Motion-to-Text Captioning (HumanML3D, Table 4)

| Method | R-Top1 | Bleu@4 | Rouge | Cider | BertScore |
|---|---|---|---|---|---|
| MotionGPT (retrained) | 0.521 | 12.4 | 38.5 | 29.3 | 31.2 |
| **ReMoGPT** | **0.534** | **13.4** | **39.6** | **31.5** | **33.9** |

### Rare Motion Generation (HumanML3D Top-5%, Table 8)

| Method | MMDist | Top-5% MMDist | Balanced MMDist | Diversity | Top-5% Diversity |
|---|---|---|---|---|---|
| ReMoDiffuse | 3.091 | 4.317 | 3.936 | 9.208 | 8.969 |
| MotionGPT | 3.613 | 4.421 | 4.161 | 9.354 | 8.889 |
| **ReMoGPT** | **3.001** | **3.563** | **3.056** | **9.701** | **9.117** |

### Ablation: 检索模态 (Table 6, HumanML3D)

- 无检索 baseline → R-Top1 0.431 / FID 0.361
- 仅 T2T or M2M (k=1) → 0.472 / 0.232
- 仅 T2M or M2T (k=1) → 0.479 / 0.231
- **Both (k=1+1, PL-TMR)** → **0.501 / 0.205**（最优）
- Both 替换为 TMR → 0.493 / 0.218；替换为 MotionPatches → 0.497 / 0.210

### External Database (Table 7)

训练 HumanML3D、数据库换 Motion-X（HumanML3D 的超集）→ FID 进一步降至 0.189。

## 局限与 Gap

### 作者自承

1. **全新动作域无法泛化**：如 yoga 数据集 (Tripathi et al. 2023)，即使把该数据集用作外部数据库（不参与训练），模型仍无法生成合理 yoga 动作——根本瓶颈在 VQ-VAE motion tokenizer 只见过训练分布的动作，decoder 无法重建分布外动作。
2. **长序列输入受限**：T5 对拼接了多条 motion token 序列的超长 prompt 处理能力有限，4 条检索样本已不如 2 条。

### 观察到的 Gap

3. **检索与生成空间分离**：PL-TMR 在独立的对比 embedding 空间检索，检索结果再转为 VQ-VAE token 喂给 T5——两个空间没有对齐，检索质量上界受限于对比学习而非生成目标。
4. **部位划分硬编码为 6 部位 (SMPL kinematic tree)**，不像层级化方案可在粗 / 细粒度间调节；对面部表情 / 手指等精细部位没有覆盖。
5. **检索数据库 = 训练集**，没有探索外挂更大动作库或在线扩展的可能。
6. **单次检索**：对于组合型描述（"先走再蹲再转"），只做一次全局检索，无法为各段子动作分别找参考。

## 与 ReMoMask V2 的关联

### 检索粒度对比

ReMoGPT 用 6 个固定部位 (Right/Left Arm/Leg + Backbone + Root) 做 part-level retrieval，ReMoMask 用 HBM（层级化 body model，5 个部位组，附带全身表征）。两者都走"部位级检索"路线，但 HBM 提供层级结构（全身 → 部位组）而非扁平拼接，理论上能在粗细粒度间灵活切换。

### 检索 / 生成空间关系——核心差异

ReMoGPT 的架构是**两套独立空间**：
- 检索端：PL-TMR 对比 embedding space（cosine 相似度）
- 生成端：VQ-VAE discrete token space + T5 autoregressive LM

检索结果必须经过 VQ-VAE 量化、拼入 T5 prompt，检索目标和生成目标之间没有梯度回传。

**ReMoMask V2 Plan A（latent-aligned retrieval）的核心主张正是消除这一分离**：让检索 query/key 和生成条件处于同一个 VQ-VAE latent space，检索和生成共享同一表征，避免跨空间转换带来的信息损失。ReMoGPT Table 6 ablation 的结果（PL-TMR > TMR > MotionPatches）间接说明检索质量对下游生成至关重要，但检索-生成对齐的问题它完全没有触及。

### 可借鉴的发现

- **多模态检索（T2M + T2T）优于单模态**：Table 6 证实，这与 ReMoMask V2 "latent + text 双路检索"的设计方向一致。
- **k=1+1 最优、k=2+2 不涨反跌**：提示 prompt 拼接 motion token 的信噪比问题；ReMoMask 若走 latent 条件注入而非 prompt 拼接，可能绕过这一瓶颈。
- **Rare motion 大幅提升**：RAG 对长尾 / OOD 文本最有价值，这一趋势与 ReMoMask V1 观察一致，为 V2 强化 RAG 提供了更多量化证据。

## 相关工作反链

- [[rag-for-motion-generation]] — PL-TMR + T5 instruction tuning 属于 motion RAG 的"检索 + prompt context 注入"范式
- [[contrastive-cross-modal-retrieval]] — PL-TMR 6 部位独立 Transformer 对比学习
- [[retrieval-generation-fusion]] — 检索 motion-caption 对作为 prompt context 拼入 T5
- [[gap-motion-contrastive-data-scarcity]] — 23K 数据量限制 PL-TMR 泛化，yoga 等 OOD 域完全失败
- [[2304.01116]] — ReMoDiffuse：ReMoGPT 直接对标的 RAG baseline，指出其 text-to-text 检索和 classifier-free guidance 的局限
- [[2305.00976]] — TMR：PL-TMR 的 whole-body 对比检索 baseline
- [[2409.12140]] — MoRAG：同期 part-level retrieval 工作，用 LLM 拆文本 + 3 个独立 TMR，ReMoGPT 在 motion encoder 端分部位而非 text 端
- [[2508.02605]] — ReMoMask V1：本方法前版，HBM 检索 + mask-based 生成
