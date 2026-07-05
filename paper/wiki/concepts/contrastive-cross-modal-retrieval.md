---
type: concept
related_papers: [[2305.00976]], [[2508.02605]], [[2409.12140]], [[2510.10480]], [[2604.17866]], [[remogpt-aaai2025]], [[2508.12081]], [[2601.07632]]
---

## 领域现状

跨模态检索是 retrieval-augmented generation 的基础环节：给定一个模态的查询（文本、结合位点、用户问题），在另一个模态的数据库中找到语义最相关的条目（动作序列、蛋白界面、文档）。当前主流做法是双编码器 + 对比损失：两个编码器分别将不同模态映射到共享 embedding 空间，通过 InfoNCE 或其变体拉近正对、推远负对，推理时用余弦相似度检索。

五篇论文均采用此范式，但在负样本策略、编码器架构和训练目标上各有侧重：

- **TMR** ([[2305.00976]])：TEMOS VAE 编码器 + InfoNCE（$\tau=0.1$），首次在 text-motion 检索中引入错误负样本过滤——用 MPNet 计算批内文本相似度，过滤阈值 > 0.8 的 pair（KIT 训练集约 17.29% 被过滤）。
- **ReMoMask** ([[2508.02605]])：BMM 双向动量对比学习，引入两条 65536 大小的负样本队列，将负样本数量与 mini-batch 解耦，动量系数 $\tilde{m}=0.99$、$\tau=0.07$。Part-level encoder 将全身运动分为 6 个部位编码。
- **MoRAG** ([[2409.12140]])：三个独立 TMR 模型分别对应躯干/手/腿，沿用 TMR 的 InfoNCE + 错误负样本过滤（MPNet 相似度阈值 0.8），$\tau=0.1$，latent 维度 256。
- **RADiAnce** ([[2510.10480]])：在全原子 VAE 隐空间上施加双向对比损失，将结合位点图级 embedding $k$ 与 binder 图级 embedding $v$ 对齐，跨越肽/抗体/蛋白片段三个域。
- **LAnR** ([[2604.17866]])：用同一 LLM 的 `[PRED]` token hidden state 作为检索向量，InfoNCE + ANCE hard negative 挖掘。
- **ReMoGPT** ([[remogpt-aaai2025]])：PL-TMR 将全身动作按 SMPL 运动学树分为 6 个部位（Right/Left Arm/Leg + Backbone + Root），各部位独立 Transformer 编码后拼接做对比学习，参数量 118M。同时用 T2M + T2T 双模态检索。
- **VimoRAG** ([[2508.12081]])：Gemini-MVR 双通道检索器——动作级（AlphaPose 2D 关键点 + MotionBERT 编码）和对象级（InternVideo）——通过关键点感知路由器自适应融合。用 425k 视频替代 3D 动作库，但面临跨模态鸿沟。
- **GeoMotionGPT** ([[2601.07632]])：在 codebook 上施加正交正则 $\|\hat{C}\hat{C}^\top - I_K\|_F^2$，使 codebook 向量近似正交，间接改善基于 codebook 的跨模态对齐。不直接做检索，但其几何约束对 latent-aligned retrieval 有启示。

## 前人忽略的问题

1. **负样本质量 vs 数量的权衡**：TMR 和 MoRAG 侧重过滤假负样本（语义相近但被当作负对），ReMoMask 侧重扩大负样本池。两个方向互相独立，尚无工作同时做——ReMoMask 的 65536 动量队列未过滤假负样本，TMR 过滤后 batch 内有效负样本减少。
2. **对比空间与生成空间的割裂**：TMR、ReMoMask、MoRAG、ReMoDiffuse 的对比检索空间与下游生成模型的 latent 空间完全独立（TMR 在 256 维 VAE 空间，ReMoMask 的 VQ-VAE 在 1024 维 codebook 空间）。RADiAnce 是唯一在同一 VAE 隐空间内同时完成对比对齐和条件扩散的工作。
3. **对比温度 $\tau$ 的敏感性**：TMR 实验表明 $\tau$ 敏感度极高（$\tau=0.001$ 时 R@3 仅 27.23，$\tau=0.1$ 时 41.93，$\tau=1.0$ 时 3.61），但多数后续工作（MoRAG、ReMoMask）直接沿用经验值而未重新调参。

## 共存的挑战

1. **动作描述语义高度重叠**：TMR 指出 KIT 训练集文本间平均 cosine 相似度达 0.71（远高于图文数据集 LAION 的 0.56），朴素对比学习容易误推语义相近 pair。
2. **数据规模远不及视觉-语言领域**：HumanML3D 仅 23K 条动作（TMR 局限 1），与 CLIP 训练的 400M 图文 pair 差数个数量级，限制了对比学习的泛化能力。
3. **部位级语义的歧义性**：MoRAG 用 LLM 分解文本为部位描述，但 LLM 生成的描述可能语义偏差（MoRAG 局限 1），ReMoMask 的 part-level encoder 对 "jumping joyfully" 等抽象描述效果不佳（ReMoMask 局限 3）。

## 主要解决方法族（按方法分类,不按论文分类）

### 方法族 A：扩大负样本池

- **动量队列**（ReMoMask BMM）：双动量编码器 + 65536 大小队列，负样本数不受 batch size 约束。参数量 238M，是 TMR（82M）的近 3 倍。
- **全库编码**（LAnR）：用 LLM 对整个语料做 last-token pooling 编码，推理时直接全库检索。开销大——每次换 backbone 需重新编码（LAnR 局限 1）。

### 方法族 B：过滤假负样本

- **文本相似度阈值**（TMR、MoRAG）：MPNet 计算批内文本 pair 相似度 > 0.8 则剔除。简单有效但阈值固定。
- **ANCE hard negative**（LAnR）：从模型自身检索错误中挖掘 hard negative，随训练迭代更新。

### 方法族 C：联合训练对比 + 生成

- **对比 + VAE 重建**（TMR、MoRAG、RADiAnce）：InfoNCE + 重建损失联合训练。TMR 消融证明去掉生成分支后 R@3 从 41.93 降到 36.87——生成分支迫使 latent 保留完整语义而非走 bag-of-words 捷径。
- **对比 + NTP**（LAnR）：$\mathcal{L}_{NCE} + \lambda \mathcal{L}_{NTP}$，消融证明 $\lambda_{NTP}=0$ 时性能崩溃至约 0.003 EM。

### 方法族 D：隐空间统一（对比 + 生成共享空间）

- **VAE latent 对比 + latent 扩散**（RADiAnce）：在同一个 VAE 隐空间上同时做对比对齐和条件扩散。检索和生成共享 latent 表征，无需额外对齐步骤。
- **LLM hidden state 作 query**（LAnR）：`[PRED]` token 的 hidden state 既是检索 embedding 又是生成上下文的一部分。

## 开放问题 / Gap

1. **动量队列 + 假负样本过滤的结合**：ReMoMask 的 65536 队列中是否存在大量假负样本？在动作描述语义高度重叠的场景下，队列越大假负样本越多，两种策略的结合尚未探索。
2. **对比空间 → 生成空间的迁移损失**：当检索空间（如 TMR 256 维）与生成空间（如 VQ-VAE 1024 维 codebook）不同时，检索结果到底在多大程度上「对」了——即检索空间的 top-K 是否也是生成空间的最优条件？目前没有量化分析（参见 [[gap-latent-aligned-retrieval-for-motion]]）。
3. **小数据下对比学习的上限**：23K 样本训练对比模型的性能天花板在哪里？是否可以通过数据增强、合成数据或跨域预训练突破？参见 [[gap-motion-contrastive-data-scarcity]]。
4. **部位级检索的统一框架缺失**：MoRAG 用 3 部位 + LLM 文本分解，ReMoMask 用 5 部位组 + HBM，ReMoGPT 用 6 部位 + 独立 Transformer。部位划分方案、编码器架构和融合策略各不相同，缺少系统比较。RMD ([[2412.04343]]) 的层级级联（全身 → 半身 → 6 部位，按检索信心自适应选择粒度）提供了一种灵活框架但不学习对比空间。
5. **codebook 几何对对比学习的影响**：GeoMotionGPT 证明正交 codebook 改善了 motion understanding，但 codebook 几何是否也影响对比检索质量（如 latent-aligned retrieval 场景），尚未探索。参见 [[vqvae-codebook-quality]]。
