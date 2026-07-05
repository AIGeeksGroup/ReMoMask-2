---
type: gap
seeded_from: [[2305.00976]], [[contrastive-cross-modal-retrieval]]
novelty_verified: false
---

## 问题陈述

Text-to-motion 对比检索模型（TMR、Part_TMR/BMM、PL-TMR）的训练数据极度匮乏，且存在有效负样本不足的结构性问题：

1. **规模差距**：HumanML3D 仅 23K 条动作-文本配对（TMR [[2305.00976]] 局限 1），与视觉-语言对比学习的标准训练规模（CLIP 400M 图文对、SigLIP 10B+）差 4-5 个数量级。

2. **语义高度重叠**：TMR 测量 KIT 训练集文本间平均 cosine 相似度达 0.71（远高于图文数据集 LAION 的 0.56）。HumanML3D 虽未报告具体数字，但动作描述的语义同质性类似——"walk forward slowly" 和 "a person walks at a slow pace" 本质描述同一动作，作为负样本对毫无区分度。

3. **有效负样本枯竭**：TMR 用 MPNet 相似度阈值 > 0.8 过滤假负样本后，KIT 训练集约 17.29% 的 batch 内 pair 被剔除。ReMoMask 的 65536 动量队列扩大了负样本池，但未过滤假负样本——队列越大，假负样本比例可能越高。两种策略（扩大 vs 过滤）互相矛盾，尚无工作同时解决。

## 为什么前人没解决

1. **3D 动作标注成本极高**：每条 HumanML3D 数据需要专业动作捕捉设备 + 人工文本标注，扩展数据集的人力和设备门槛远高于图像-文本数据（可从互联网爬取）。

2. **跨数据集标准不统一**：HumanML3D（AMASS 子集）、KIT-ML、BABEL、Motion-X 使用不同的骨架定义、采样率和标注风格，简单合并会引入分布偏移。ReMoGPT Table 7 尝试用 Motion-X 作外部数据库时 FID 进一步降至 0.189，但只是换了数据库而非扩大训练集。

3. **视频→3D 动作转换有信息损失**：VimoRAG ([[2508.12081]]) 用 425k 视频替代 3D 动作库，规模提升 30 倍，但 2D 视频→3D 动作的跨模态鸿沟客观存在。McDPO 只是教 LLM "何时忽略"，未从表征层面解决对齐。

4. **合成数据质量不确定**：LLM 生成的动作文本描述（如 VimoRAG 用 Qwen2-VL 为视频合成描述）可能引入语义偏差。ReMoGPT 发现 LLM 分解的部位描述质量不稳定，直接影响检索质量。

## 可能的切入点

1. **对比学习的数据增强**：text 侧用 LLM paraphrase 扩展描述多样性（减少文本重叠）；motion 侧用随机裁剪、时序扰动、关节 dropout 等增强。需验证增强后的对比学习是否能突破 23K 规模的性能天花板。

2. **Hard negative mining 迁移**：LAnR ([[2604.17866]]) 的 ANCE hard negative 策略从模型自身检索错误中挖掘 hard negative，随训练迭代更新。可直接迁移到 motion 对比学习——用当前检索器的 failure cases 作为下一轮训练的 hard negative。

3. **跨数据集联合训练**：RADiAnce ([[2510.10480]]) 在蛋白设计中实现了肽/抗体/蛋白片段三域联合训练，ITO 从 28.21% 提升到 43.93%。动作领域可尝试 HumanML3D + KIT-ML + BABEL 联合训练对比模型，需统一骨架表示。

4. **视频预训练 + 3D 动作微调**：先在大规模视频-文本对上预训练 motion-text 对比模型（利用 VimoRAG 的 HcVD 425k 视频），再在小规模 3D 动作数据上微调。类似 CLIP 先 web-scale 预训练再 downstream fine-tune 的范式。

5. **合成负样本**：利用 VQ-VAE codebook 的组合性质合成"伪动作"——随机替换部分 body part 的 token 序列，生成语义上不可能的动作（如"上身走路 + 下身坐着"），作为 hard negative。

## 相关文献

- [[2305.00976]] — TMR，23K 数据量限制 + 17.29% 假负样本过滤
- [[2508.02605]] — ReMoMask BMM，65536 动量队列扩大负样本池
- [[2409.12140]] — MoRAG，沿用 TMR 假负样本过滤
- [[remogpt-aaai2025]] — ReMoGPT PL-TMR，参数量 118M 但数据量受限
- [[2508.12081]] — VimoRAG，425k 视频扩展但跨模态鸿沟
- [[2510.10480]] — RADiAnce，跨域联合训练的成功先例
- [[2604.17866]] — LAnR ANCE hard negative mining
- [[contrastive-cross-modal-retrieval]] — 对比学习方法族综述
