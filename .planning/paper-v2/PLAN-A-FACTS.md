# ReMoMask-2 论文写作权威事实清单(Plan A)

> 用途:论文写作(方法/消融/图表)的唯一事实来源。所有写作断言必须可追溯到此文件;
> 此文件的每一条都来自 .planning/phases/02-*/ SUMMARY 或 pami.tex 原文,禁止推断。
> 更新:2026-07-05。

## 0. 版本约定

- **最新 LaTeX 工程**:`D:\tpami\currentversion\v2working\`(解压自 `_TPAMI_2026__ReMoMask_2 (1).zip`)——唯一可信版本,其他位置(如 `D:/tpami/_TPAMI_2026__ReMoMask_2/`)一律作废。
- 主文件 `pami.tex`(单文件,~1690 行);标题已定:**ReMoMask-2: Latent Retrieval-Augmented Masked Motion Generation**。
- 当前内容 = ECCV 中稿版 + rebuttal 融合;**ECCV 内容不质疑、不大改**,只做期刊拓展与小调整。

## 1. V1(ECCV 版)核心事实(以 pami.tex 为准)

### 方法
- 三个组件:**HBM**(Hierarchical Bidirectional Momentum,instance+part 双层双向动量对比,K=6 parts,queue 65,536,μ=0.999,λ_P=1)、**TSM**(Topology Structured Masking,π_{i,k}=π_base·(1−α_{i,k}),π_base=0.5)、**SSTA**(Semantic Spatial-Temporal Attention,非对称 QKV)。
- SSTA 发表版信息路由:**Q = W_q z;K = W_k concat(z, h_sem),h_sem=MLP(concat[t; R_t; R_m]);V = W_v concat(z, R_m)** —— 正文明确声称 "textual semantics (t, R_t) are excluded from Value to prevent domain mismatch"。
- tab:ablation(information source)最优行:K={t,R_t,R_m}, V={R_m},FID 0.027;V={R_t,R_m} 行 FID 0.104(差)。⚠️ 与 rt_in_value 叙事有张力,见 §3。
- 2D RVQ-VAE(来自 MoGenTS 线):6 个量化层,codebook 512×512(正文如此写;实际 ckpt 见 §2)。
- Design Principles 章(sec:preliminary):1D/2D latent × concat/crossAttn 二轴初步实验,2D+crossAttn 最优(FID 0.036)。

### 实验(HumanML3D 主数字,20 repeats,完整 pipeline)
- ReMoMask FID **0.026**,Top1 0.566(全场最佳);检索 HBM R@1 t2m **18.49**。
- 数据集:HumanML3D / KIT-ML / SnapMoGen。KIT FID 0.131;SnapMoGen FID 13.509。
- 训练:masked models 8×A800、2000 epochs、batch 64;检索模型 200 epochs、batch 128。

### 表格清单(需要 Ours→ReMoMask + 加 ReMoMask-2 高亮行的)
| 表 | label | 现名 | 处理 |
|---|---|---|---|
| Table 1 架构对比 | tab:delta | 已有 ReMoMask 行 + 空的黄色高亮 ReMoMask-2 行(示例样式:`\rowcolor{yellow!20}`) | 填充 ReMoMask-2 行(可微调列设计) |
| 检索 benchmark | tab:rag_experiment | "HBM (Ours)" ×3(三数据集) | → "HBM (ReMoMask)",下加 ReMoMask-2 占位行,multirow 5→6 |
| T2M 主表 | tab:t2m_experiment | "ReMoMask(Ours)" ×3 | → "ReMoMask",下加 ReMoMask-2 高亮占位行,multirow 12→13 |
| 其余表(prelim/消融/cross) | - | 无 "Ours" | 不动(V2 新消融另加新表) |

### 图清单
- fig:teaser = fig/teaser.pdf,caption 已改 "Overview of ReMoMask vs ReMoMask-2.",**PDF 本身还是旧图,需重做**(设计留档见 FIG1-DESIGN.md)。
- fig:framework(方法框架 a/b/c)、rag_t2m_visual、heatmap、hyperparameter、db_coverage、scatter_rebuttal、part-level-motion-encoder、demo_00、compare_00、userStudy、question_1/2、rebuttal_video。

## 2. ReMoMask-2(Plan A)实现事实(以 02-0x SUMMARY 为准)

### 核心思想
把检索空间从独立的 Part_TMR/HBM 对比语义空间迁移到**生成器自身的 RVQ-VAE 预量化连续 latent z_e 空间**,使检索证据与生成 substrate 共享同一表征;冻结 VQ-VAE,只训一个轻量 query projector(~1.57M 参数)。

### 锁定决策(不可违背)
- LOCKED-01 操作 z_e(预量化连续),不操作 z_q(离散 token)
- LOCKED-02 冻结 VQ-VAE,仅训 projector
- LOCKED-03 对齐目标用 KL 散度(优于 InfoNCE:43.46 vs 41.86)
- LOCKED-04 teacher 用 BMM(= V1 的 HBM 检索器,代码名 BMM;R@1 13.76 ≫ TMR 5.68)
- LOCKED-05 SSTA 架构不换,只加维度适配

### 跨空间秩相关实测(E03,2026-07-05,可入文的分析事实)
- 协议:N=2,000 unique motions(22,418 可用,seed 3407),BMM 512d vs z_e 1024d,cosine 排名。
- **Spearman ρ:mean 0.58(std 0.16,median 0.63)**;**top-k 近邻重叠:@1 18.0%、@5 18.1%、@10 18.5%、@50 24.1%**。
- 论文用法:全局相似结构中度相关,但检索决定性的排名头部 >80% 不一致 → representation gap 实测成立。
- 产出:results/phase4/e03_space_gap/(远程)+ .planning/.../results/e03/(本地,含 overlap_curve.png、rho_hist.png,可做图)。

### z_e 提取与几何(02-01)
- ckpt 实际参数:**code_dim2d=1024**(非默认 512)、code_dim1d=512、nb_code2d=256、num_quantizers=6、down_t=2。
- 提取:encoder2d 输出 (B, 1024, T/4, 6) → 对 (T/4, 6) mean pooling → L2 norm → (B, 1024)。
- 几何:cosine mean 0.62、std 0.25、range [−0.55, 0.998],**无塌缩,cosine 检索可行**;但严重各向异性:isotropy≈0,有效维度 11/1024,7 维解释 95% 方差。
- 分析产物:cosine_sim_hist.png、eigenvalue_spectrum.png(可做论文分析图,需转矢量图)。

### 检索数据库(02-02)
- `database_ze/`:23,384 个训练 motion 按 caption 展开为 **66,912** 对;encoded_motions.npy (66912,1,1024) L2 归一;encoded_texts.npy = projector 输出 (N,1,1024)(训练后重投影);构建 ~5 分钟。
- ZeRetriever 与 MocoTmrRetriever 接口完全兼容(tokenize/encode_text/forward),含按 motion_id 去重。

### Query Projector + KL 对齐(02-03)
- 结构:**Linear(512→1024) + GELU + Linear(1024→1024)**,输出 L2 归一;输入 CLIP ViT-B/32 text embedding(与 V1 共享)。
- 训练:BMM teacher 的 top-K(256)soft-target 排名;loss = KL(p_teacher ‖ p_student)(PyTorch kl_div(student_log_prob, teacher_prob));temperature 0.07(与 Part_TMR 一致);Adam + CosineAnnealingLR,200 epochs,batch 128,单卡;best ckpt 按 R@1 选。

### SSTA 维度适配(02-04, 02-05)
- SSTA 新增 retrieval_dim 参数;retrieval_dim=1024 时创建 re_motion_proj/re_text_proj = Linear(1024→512)(latent_dim=512);None 时 nn.Identity(V1 行为不变)。
- info_mlp 保留 3 路(t + R_m + R_t),不合并——R_m 与 R_t 承载不同信号。
- V2 全 pipeline 由 `--use_ze_retrieval` 门控,V1 路径不动。

### rt_in_value(独立 minor contribution)
- 来源:Phase 1 ABL-02(R_t 进 SSTA Value 分支,K-only → K+V),zero-cost 结果 FID 0.102→0.089(−13%)。⚠️ 未在完整重训 pipeline 复现,**论文里当占位/待验证处理,不当确定筹码**。
- **原始发表 V1 是 K-only(V 只有 R_m)**;我们的 V1 retrain 加了 rt_in_value = "增强版 baseline",保证 V1/V2 对照的唯一变量 = 检索模块。
- 与 ECCV tab:ablation 的张力及化解叙事:ECCV 表中 V+={R_t}(CLIP 语义空间的 R_t)FID 变差 0.104,支撑 "domain mismatch" 论断;**V2 中 R_t 经 projector 投入 z_e 空间(motion-aligned),不再是跨域信号**——因此 value 路由的重新审视在 latent 对齐后成立。消融设计 {Part_TMR, z_e} × {rt on/off} 正好检验此叙事(若 Part_TMR+rt 也涨,则叙事调整为"更长训练下的重新评估")。

### 训练对照(进行中,2026-07-05)
- V1 `v1_retrain_rtval`(job 13846)vs V2 `v2_ze_rtval`(job 13845),persephone 各 1×L40,配置与 V1 opt.txt 完全一致(latent_dim=512, n_heads=8, n_layers=8, ff_size=1024, batch 64, lr 2e-4 恒定, seed 3407),两边都开 rt_in_value,唯一变量 = 检索模块。
- 中间结果(MaskTransformer 单独 + 单次 eval,**不是论文数字**):V2 best FID **0.0991@ep409** vs V1 **0.1273@ep316**(V2 −22%)。
- 论文数字必须来自 `eval_res.py`(Mask+Residual 完整 pipeline)× 20 repeats,checkpoint 显式用 `net_best_fid_ep0409.tar` / `net_best_fid_ep0316.tar`(无后缀文件已被 resume 污染)。
- V2 eval 命令模板:`python eval_res.py ... --use_ze_retrieval --ze_database_path database_ze --projector_path logs/query_projector/best_projector.pt --rt_in_value --retrieval_dim 1024`。

## 3. 写作占位符约定

- 表格数值占位:`$x.xxx^{\pm.xxx}$`(检索表用 `xx.xx`);正文数值占位:`\textbf{[TBD]}`。
- dummy 分析按"一切如预期"假设写:V2 全面优于 V1、rt_in_value 两种检索器下均正贡献、z_e 检索 R@k 与 HBM 可比或有 trade-off(以生成质量为主叙事)。后续真实数据不符再改。

## 4. 明确不做 / 延后

- Plan B(iterative dynamic retrieval)暂时搁置,论文里最多在 future work 提一句。
- 实验数字回填、fig1 实际绘制(需外部工具)、消融实际跑数,均在训练/评估完成后。
