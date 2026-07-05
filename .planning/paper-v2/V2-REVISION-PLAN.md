# ReMoMask-2 期刊版逐节修改方案(REVISION PLAN)

> 2026-07-05 起草。原则:**ECCV 内容不质疑、整体保留**,期刊拓展 = 在既有骨架上加 ReMoMask-2
> 层。所有数值先占位,训练/评估完成后回填。配套事实见 PLAN-A-FACTS.md。

## 总叙事(大方向,已与用户确认过的 Plan A 定位)

ECCV 版确立了两个结构设计轴:**对齐粒度**(HBM)与**融合兼容性**(SSTA/TSM,2D latent)。
期刊版加入第三轴——**检索空间与生成表征的一致性**(representation consistency):
V1 的检索证据活在独立的对比语义空间(Part_TMR/HBM 512d),而生成器活在 RVQ-VAE latent
空间;这道"表示鸿沟"意味着检索到的 R_m 要跨域才能被生成器消化。ReMoMask-2 把检索库
直接建在生成器自己的预量化 latent z_e 上,用一个 ~1.57M 的 query projector(KL 蒸馏自
HBM teacher)把文本 query 投进同一空间——检索证据与生成 substrate 同源,SSTA 无需跨域
翻译。附带贡献:latent 对齐后,R_t 已是 motion-domain 信号,重新打开了 ECCV 版因
"domain mismatch" 而关闭的 Value 路由(rt_in_value),作为独立正交的 minor contribution。

三轴叙事让 ReMoMask-2 不是"换个检索器",而是把 ECCV 版 "结构一致性" 的世界观推到第三
个维度,与题目 "Latent Retrieval-Augmented" 完全咬合。

## 逐节修改

### Abstract
- 保留前半(HBM/TSM/SSTA 一句话不动或微调)。
- 追加 2-3 句:指出 conference 版遗留的 representation gap;introduce ReMoMask-2 =
  latent-aligned retrieval(z_e 库 + query projector KL 蒸馏);一句结果 claim(占位:
  consistently improves FID over ReMoMask,e.g. [TBD]% on HumanML3D)。
- 首句可把 "we present ReMoMask" 改为 "we present ReMoMask and its journal extension
  ReMoMask-2" 风格的表述(具体措辞由 draft 定)。

### 1 Introduction
- 前 4 段(领域背景、两条线、fig:teaser、tab:delta)保留;tab:delta 填 ReMoMask-2 行
  (+新增 "Retrieval Space" 列区分 semantic vs latent-aligned,微调,可回退)。
- "两个设计轴" 段落后追加 1-2 段:第三轴 = 检索空间一致性;point out V1 检索空间与生成
  空间异构 → 引出 ReMoMask-2。
- 贡献列表:保留原 4 条(措辞可微调),追加 2-3 条 V2 贡献:
  1. 识别 retrieval-generation representation gap(第三设计轴);
  2. latent-aligned retrieval:冻结 VQ-VAE 的 z_e 检索库 + 轻量 query projector
     (HBM teacher KL 蒸馏),生成质量显著提升(数值占位);
  3. value-pathway routing 重审(rt_in_value):latent 对齐使 R_t 成为 motion-domain
     信号,K+V 路由成立(独立 minor contribution,正交消融验证);
  4. (可并入 3)更全面的期刊版实验:z_e 几何分析、正交消融、20-repeat 完整 pipeline。
- 段末加标准期刊拓展声明段("This paper extends our conference version [cite] in the
  following aspects: ...")。⚠️ 需在 reference.bib 加 ReMoMask ECCV 自引(arXiv:2508.02605)。

### 2 Related Work
- 两小节保留;RAG-T2M 小节末补一小段:现有 RAG-T2M(含 conference 版 ReMoMask)检索
  空间均独立于生成表征;可补 1-2 句 latent-space RAG 在其他模态(图像/视频/LLM)的相关
  工作对照(引文占位,后续核实补引)。不展开大改。

### 3 Design Principles(sec:preliminary)
- 原二轴实验(tab:prelimi_experiment)保留不动。
- 末尾加一段(或小节)"Representation consistency of the retrieval space":用 z_e 几何
  分析(cosine mean 0.62 无塌缩、各向异性)论证 latent 检索可行性,自然引出第三轴。
  z_e 分析图(cosine hist + eigenspectrum)作为新图占位(fig/ze_geometry.pdf,待重绘矢量版)。

### 4 Methodology
- 4.1-4.5(Overview/HBM/Part Encoder/TSM/SSTA)整体保留;Overview 末尾加一句过渡
  (conference 版检索发生在 HBM 语义空间,4.6 将其迁移到生成 latent 空间)。
- **新增 4.6 Latent-Aligned Retrieval(ReMoMask-2 核心)**:
  a) Motivation:representation gap 一段;
  b) z_e Retrieval Database:冻结 RVQ-VAE encoder2d,z_e (B,1024,T/4,6) → 时空 mean
     pool → L2 norm → 1024-d 检索键;66,912 (motion, caption) 对;
  c) Query Projector:φ = Linear(512→1024)+GELU+Linear(1024→1024),L2 归一;
     KL 蒸馏:teacher = HBM 检索器 top-K(256) soft ranking,τ=0.07,
     L_align = KL(p_HBM ‖ p_φ);
  d) Integration with SSTA:R_m, R_t 均来自 z_e 空间(1024d),线性投影到 latent_dim;
     SSTA 结构不变(呼应 LOCKED-05:即插即用,凸显贡献干净)。
- **新增 4.7 Revisiting Value-Pathway Routing(rt_in_value)**:
  叙事:ECCV 版将 R_t 排除出 Value 是因为 CLIP 语义空间的 R_t 与 motion latent 跨域;
  latent 对齐后 R_t ∈ z_e 空间,domain mismatch 前提消失 → K+V 路由重新成立。
  定位为独立、正交的 minor contribution;结论以正交消融表(§5)为准。
  ⚠️ 写作红线:不否定 ECCV 表 tab:ablation 的结论,而是"前提变了,结论随之更新"。
- 符号表(tab:notation)补新符号:z_e、φ、q、p_HBM/p_φ、retrieval_dim 等。

### 5 Experiment
- 数据集/指标/实现细节:保留;Implementation Details 追加 ReMoMask-2 段(projector
  训练超参、z_e 库构建、V2 训练配置——严格单变量对照,数值按事实清单写)。
- 主表 tab:t2m_experiment:Ours→ReMoMask,加 ReMoMask-2 黄色高亮占位行 ×3 数据集。
- 检索表 tab:rag_experiment:HBM (Ours)→HBM (ReMoMask),加 ReMoMask-2 占位行
  (z_e-space 检索的 R@k;若最终数字不佳,fallback 叙事 = 检索精度 trade-off 但生成
  质量为王,由 dummy 分析预铺一句)。
- Main Results 文字:各加一段 ReMoMask-2 的对比描述(数值占位)。
- **新增消融小节(V2)**,设计见 ABLATION-DESIGN.md,核心:
  1. 正交消融 {Part_TMR, z_e} × {rt_in_value on/off}(主消融,2×2);
  2. 对齐目标:KL vs InfoNCE(有 LatentRAG 数据支撑 43.46 vs 41.86,占位重验);
  3. teacher 选择:BMM vs TMR;
  4. 检索键聚合方式 / top-K / projector 容量(酌情,占位);
  5. z_e 几何分析(定性图 + 讨论,呼应 §3)。
- 原有消融(hierarchical/momentum/info source/masking/fusion/temporal/coverage)全保留。

### 6 Limitations / 7 Conclusion
- Limitations:保留数据库覆盖度一条;加一句 z_e 各向异性(有效维度 11/1024)与
  projector 蒸馏上限依赖 teacher 质量;Plan B(iterative/dynamic retrieval)作为
  future work 一句带过。
- Conclusion:补 ReMoMask-2 一段(三轴叙事收束)。

### 图
- fig:teaser 重做(ReMoMask vs ReMoMask-2 overall 对比图)——设计留档 FIG1-DESIGN.md,
  等外部绘图工具;tex 端 caption 先占位扩写。
- fig:framework 后续可能加 (d) latent-aligned retrieval 子图或独立新图(留待 fig1 一起做)。
- 新增 z_e 几何分析图占位(fig/ze_geometry.pdf)。

## 风险与开放问题(写作时显式留给用户拍板)

1. **tab:delta 加 "Retrieval Space" 列**:属微调,若用户不喜欢可只填行不加列。
2. **检索表的 ReMoMask-2 行**:z_e 检索的 R@k 可能低于 HBM(z_e 是运动几何空间),
   届时可改为不在检索表加行、只在消融中报告——先占位,回填时再定。
3. **rt_in_value 与 ECCV tab:ablation 的张力**:采用"前提变化"叙事化解;若正交消融
   显示 Part_TMR+rt 也涨,则改为"更长训练/完整 pipeline 下的重新评估"叙事。
4. ECCV 自引条目需要正式出处(ECCV 2025 proceedings 信息)——bib 先用 arXiv 占位。
5. 正文声称 codebook "512 codes of 512-dims" 与 ckpt 实际 nb_code2d=256/code_dim2d=1024
   不一致——ECCV 原文如此,**不动正文**,但 V2 新增段落写 z_e 维度时用 1024(事实),
   两处并存可能被审稿人问,标记待用户定夺。

## 执行顺序

1. 表格三处改名/加行(机械,先做)
2. 方法 4.6/4.7 + 摘要/引言/结论追加(draft → 评审 → 融入)
3. 消融设计 + dummy 分析(新表占位 + 文字)
4. fig1 设计留档
5. memory/wiki 对齐
