# Phase 4 实验执行规格(E01-E11)——可直接上手版

> 2026-07-05 制定。每条实验:论文落点 → 前置 → 命令 → 代码 gap → 产出 → 成本。
> 远程环境:diana:~/ReMoMask-2,conda env `remomask`;提交模板见 memory
> `reference_usyd-slurm-cluster`。所有 eval 结果落 `results/phase4/<exp_id>/`,
> 并同步一份 json 摘要回本地 `.planning/phases/04-comprehensive-evaluation/results/`。
> 通用红线:ckpt 显式 ep 后缀;sbatch `set -e`;数字回填后在 PLACEHOLDERS.md 划账。

## 通用资产清单(跑任何实验前先核对)

| 资产 | 位置(远程) | 状态 |
|---|---|---|
| V1 best ckpt | `checkpoints/humanml3d/v1_retrain_rtval/model/net_best_fid_ep0316.tar` | ✅(路径按实验名核实) |
| V2 best ckpt | `checkpoints/humanml3d/v2_ze_rtval/model/net_best_fid_ep0409.tar` | ✅ |
| V1 官方 ckpt(ep616) | `checkpoints/humanml3d/pretrain_mtrans/...` | ✅(Phase 1 用过) |
| BMM/Part_TMR 检索库 | `database/`(66,912 条,全量重建版) | ✅ |
| z_e 检索库 | `database_ze/`(encoded_motions (66912,1,1024)) | ✅ |
| query projector | `logs/query_projector/best_projector.pt` | ✅ |
| RVQ-VAE ckpt(冻结) | Phase 2 所用 `--vq_name` 对应目录 | ✅ |
| ResidualTransformer ckpt | `logs/humanml3d/pretrain_rtrans/model/net_best_fid.tar`,V1/V2 共享 | ✅ 已核实(2026-07-05):**rtrans 不吃检索条件**(transformer_aux.py 零 re_dict 引用,检索只进 mtrans 的 SSTA)→ 共享冻结 rtrans 对受控对照是最干净协议;另 eval_res.py 内 repeat_time=20 硬编码,与论文口径一致 |

## E01 正式对照评估(主数字)★P0

- **论文落点**:tab:t2m_experiment HumanML3D 的 ReMoMask-2 行(及 retrain 口径决策后的表述);abstract/intro/conclusion/teaser 的 [TBD]%;comparison.json/tex。
- **前置**:训练停止(D-P4-01);资产表核实。
- **命令**(V2;V1 去掉三个 ze flags 与 --rt_in_value 保留——注意 V1 retrain 也开了 rtval):
  ```bash
  python eval_res.py --dataset_name humanml3d --name <rtrans_name> \
    --mtrans_name v2_ze_rtval --ckpt net_best_fid_ep0409.tar \
    --use_ze_retrieval --ze_database_path database_ze \
    --projector_path logs/query_projector/best_projector.pt \
    --rt_in_value --retrieval_dim 1024 \
    --repeat_times 20 --ext phase4_e01_v2
  # V1: --mtrans_name v1_retrain_rtval --ckpt net_best_fid_ep0316.tar --rt_in_value(无 ze flags)
  ```
- **代码 gap**:确认 eval_res.py 接受 --ckpt 指定文件名(若硬编码 latest/best 需加 ~5 LOC);其余 02-05 已支持。
- **产出**:两组 mean±std 全指标;`compare_v1_v2.py` 输出 comparison.json/tex。
- **成本**:2 × (20 repeats ≈ 数小时) 单卡。

## E02 mask-only 诊断 ★P1

- **落点**:tab:ablation_v2_maskonly(4 格)。
- **命令**:同 E01 但 `eval_mask.py`,V1/V2 各一次 ×20 repeats;full-pipeline 两格直接复用 E01 数字。
- **gap**:无(--repeat_times 现成)。
- **成本**:2 次 eval。

## E03 跨空间秩相关(representation gap 量化)★P0,无前置可立即跑

- **落点**:Design Principles「Representation Consistency」段新增一句量化结论 + 可选小图;
  是全文前提的实测证据。
- **方法**:采样 N=2,000 motions(train split,按 motion_id 去重);分别在 BMM 空间(database/encoded_motions 512d)与 z_e 空间(database_ze 1024d)算全对 cosine 排名;报 (a) Spearman ρ(对每个 query 的邻居排名相关,取均值±std);(b) top-k 邻居重叠率 Jaccard@{1,5,10,50}。低相关 ⇒ gap 实测成立。
- **gap**:**新脚本 `scripts/analyze_space_gap.py`(~150 LOC)**——输入两个 database 目录,输出 summary.json + overlap 曲线 png。纯 numpy,可本地跑(两库共 ~650MB,本地已有副本则不占集群)。
- **成本**:CPU 分钟级。

## E04 z_e vs z_q 检索键 ★P0

- **落点**:新表(建议 tab:ablation_v2_keyspace,或并入 alignment 表加一节)——检索层 R@k 必做;下游 FID 用 zero-shot 换库口径(SSTA 权重不动,caption 注明),完整重训口径可选。
- **方法**:z_q = 6 层残差量化后的重建 latent(与 z_e 同形 1024×T/4×6),同 pooling+L2;库重建后:(a) projector 不变做检索 R@k;(b) eval_mask.py 换 `--ze_database_path database_zq` 出 zero-shot FID。
- **gap**:**build_rag_database_ze.py 加 `--use_quantized` flag(~30 LOC)**:encoder2d 后过 quantizer 取重建 latent 而非原始 z_e。
- **成本**:库重建 ~5min + 2 次 eval。

## E05 推理期检索条数 k ★P1

- **落点**:新小表或曲线(tab:ablation_v2_topk);当前默认 top_k=2、候选池 num_retrieval=10。
- **方法**:V2 权重不动,k ∈ {1, 2, 4, 8} eval_mask ×20(k=2 即主配置复用 E02);k=8 时 num_retrieval 提到 ≥16。
- **gap**:**eval 入口加 `--retrieval_topk`/`--retrieval_pool` 透传到 ZeRetriever ctor(~10 LOC)**(t2m_retriever 走 cfg.rag.top_k,同样透传)。V1 对照可选(同 sweep 一遍)。
- **成本**:3-7 次 eval。

## E06 database coverage × V2 ★P1

- **落点**:fig:db_coverage 加 V2 曲线(或新图);叙事=对齐放大检索效用(V2 斜率更陡则成立)。
- **方法**:{10,25,50,75,100}% 子采样 database_ze(按 motion_id 分层采样,固定 seed),各 eval_mask ×20 或 ×5(曲线用途,可降 repeats,caption 注明)。
- **gap**:**子采样脚本 `scripts/subsample_database.py`(~40 LOC)**,输出 database_ze_p10 等目录;查 V1 当年 coverage 实验脚本是否在 repo(scripts/ 里没看到,大概率一次性脚本,重写)。
- **成本**:5 × eval(×5 repeats 则轻)。

## E07 效率图(L40 全图重测)★P2

> **已拍板(2026-07-05):方案 A——整图在 L40 上重测**,caption 改 "a single NVIDIA L40"。

- **落点**:fig:scatter 整图换血 + ReMoMask-2 新点(含 projector 开销)。
- **步骤 ①**:L40 上测 V1(v1_retrain_rtval)与 V2(v2_ze_rtval)各 100 样本平均推理时延
  (momask 协议)。
- **步骤 ②**:盘点集群上基线推理资产(MoMask/ReMoDiffuse/MDM/T2M-GPT/MoGenTS/MoRAG/
  ReMoGPT);现成的逐个 L40 计时;缺的列清单向用户要 ECCV † 复现资产;个别拉不起来的
  基线沿用旧值 + caption 脚注(部分降级,不阻塞出图)。
- **gap**:无新代码,计时脚本沿用 V1 做法(scripts/flops.py 可辅助)。
- **成本**:每方法分钟级计时;变数 = 基线环境搭建。

## E08 projector 变体消融(objective / teacher / κ / 容量)★P1

- **落点**:tab:ablation_v2_alignment(KL vs InfoNCE **vs MSE regression**——按 2026-07-05 决定补第三行,tex 侧回填时加行)、tab:ablation_v2_teacher、tab:ablation_v2_sensitivity。
- **方法**:每个变体重训 projector(200ep 单卡)→ 报 projector R@1/5/10(脚本内置 recall_at_k)+ 下游 zero-shot FID(换 projector 不动 SSTA,eval_mask ×20;caption 注明 shared-backbone 口径;若审稿要求再补重训口径)。
- **变体矩阵**(默认=KL/BMM/κ256/MLP-1024,已有):
  - objective:InfoNCE、MSE → **gap:train_query_projector.py 加 `--objective {kl,infonce,mse}`(~40 LOC)**
  - teacher:TMR → 走 `--database_bmm_path database_tmr` 换库思路;**gap:先产 TMR 特征库**——用 Part_TMR/ 代码加载 TMR 官方权重(HumanML3D)对 66,912 条编码,输出与 database/ 同格式(**gap 脚本 ~80 LOC;TMR ckpt 来源:官方 release,下载列入前置**)
  - κ ∈ {64,256,1024}:`--teacher_topk` **现成,零 gap**
  - 容量 {Linear, MLP-1024, MLP-2048}:**gap:query_projector.py + 训练脚本加 `--hidden {0,1024,2048}`(~20 LOC)**
- **成本**:~7 次 projector 训练(<1h each,单卡)+ 对应 eval。

## E09 正交 2×2 补两格 ★P0(唯一需要新训 MaskTransformer 的)

> **已拍板(2026-07-05):800ep 截断;排在全部 eval 批次(E01-E08/E10)之后最后跑。**

- **落点**:tab:ablation_v2_orthogonal 第 1、3 行(Semantic✗ / z_e✗);已有第 2 行(v1_retrain_rtval)、第 4 行(v2_ze_rtval)。
- **方法**:严格复用两条主线的 sbatch(同超参同 seed),仅去掉 `--rt_in_value` 并设
  `--max_epoch 800`;实验名 `v1_retrain_nortval` / `v2_ze_nortval`;persephone 2×L40 并行;
  各自 eval_res ×20(显式 ep 后缀 best ckpt)。论文 caption 注明:补格采用 800-epoch
  截断协议,best checkpoint 窗口(两主线 ep316/ep409)完整覆盖。
- **gap**:sbatch 以 `job_v1_retrain.sh` / `train_v2.sh` 为模板复制改名去 flag(分钟级);
  **注意 watchdog 要一并覆盖新 job**。
- **成本**:2 格并行 ≈ 2.7 天 + eval。

## E10 定性近邻可视化 + 图资产 ★P2

- **落点**:新定性图(同一 prompt 在 S 与 Z 空间的 top-3 检索对比渲染)、fig1 性能面板数据(E01 出)、fig/ze_geometry.pdf 矢量重绘(数据现成:results/ze_analysis/)。
- **方法**:选 4-6 条代表 prompt(含一条长尾),两空间各取 top-3,render.py/plot 渲染网格图;体现"语义相近但运动学不同 vs 运动学贴近"。
- **gap**:小脚本拼装(检索调用现成接口);渲染管线沿用 repo(render.py)。
- **成本**:小时级,人工挑图。

## E11 长尾/罕见 prompt 子集(可选,协议草案)

- **落点**:若做,新表一张(全集 vs 尾部子集的 FID/R-Prec,V1/V2 对比),支撑 RAG 鲁棒性卖点。
- **协议草案**:test set caption 按 (a) 训练集 caption 词频倒序 或 (b) 与训练集最近邻相似度倒序,取尾部 20% 为 rare 子集;两法各报或选其一(写作时定)。评估管线复用 E01,只换 split 文件。
- **gap**:子集构造脚本(~60 LOC)+ eval 数据加载器接受自定义 split 列表(查 dataset 类,预计小改)。
- **状态**:等用户点头再排。

## E12 Residual Transformer 重训(条件性,随大服务器阶段)

> 2026-07-05 用户提出、经代码核实后定位:**rtrans 与检索无关**(不吃 re_dict),
> 因此 (a) 所有受控对照/消融(E01/E02/E09)共享冻结 pretrain_rtrans 是**最干净**协议,
> 不需要为 V2 单独训 rtrans;(b) 仅当最终"发布版"主数字要走完整重训口径
> (对齐 ECCV 的自训 mtrans+rtrans 协议,预计随换大服务器的长训批一起做)时,
> 重训 **一个** rtrans 即可——因为它与检索空间无关,新 rtrans 可被 V1/V2/全部消融
> 配置共享,只训一次。

- **触发条件**:大服务器完整重训批启动(与 04-07 同期),或审稿人质疑 rtrans 口径。
- **方法**:`run_rtrans.sh` 协议照搬 V1(超参以其 opt.txt 为准),训一次共享;之后全部
  eval_res 口径统一换新 rtrans 重跑(20 repeats 便宜)。
- **成本**:与 mtrans 同量级(2000ep);大服务器上一次性。
- **论文口径注意**:换 rtrans 前后的数字不可混表;当前所有 E01/E09 数字的 caption
  口径 = "shared pretrained residual transformer"。

## 04-07 数据集扩展(KIT-ML / SnapMoGen)——留白但定死用法

> **已拍板(2026-07-05):做,但延后到 HumanML3D 全线(含 E09)之后;长时间训练可能
> 切到卡更多的新服务器**——届时先跑下方第 1 步资产盘点(在新机器上同样适用),数据
> 下载/预处理排期到时再定。主表占位行保留。

**用什么**:
- KIT-ML:HumanML3D repo(EricGuo5513/HumanML3D)流程产出的标准处理版;feature dim **251**,21 joints;splits 沿用 T2M 标准。
- SnapMoGen:MoMask++(momask2)发布版;20,450 motions / 122,565 texts;feature dim **296**。

**怎么用(每个数据集重走一遍 HumanML3D 全流程,协议不变)**:
1. 盘点 V1 侧资产:该数据集的 RVQ-VAE ckpt、BMM/Part_TMR retriever ckpt、database/ 是否随 V1 发布现成(**先盘点 checkpoints/,缺哪个训哪个——这是最大成本变数**);
2. `build_rag_database.py`(BMM 库,teacher 用)+ `build_rag_database_ze.py`(z_e 库;code_dim 以该数据集 VQ ckpt opt.txt 为准,**不得沿用 1024 假设**);
3. `train_query_projector.py` 蒸馏(同超参);
4. V1/V2 对照训练(同 seed 同协议,双方开 rt_in_value)+ E01 式评估 ×20;
5. 回填主表对应 ReMoMask-2 行。

**留白**:数据下载与预处理排期;若 V1 侧 VQ/retriever ckpt 缺失,追加其训练成本估算后再拍板是否砍数据集。

## 汇总:代码 gap 清单(全部可分发 Sonnet xhigh 实现)

| # | 文件 | 改动 | LOC | 服务实验 |
|---|---|---|---|---|
| G1 | eval_res.py | --ckpt 指定文件名(若缺) | ~5 | E01 |
| G2 | scripts/analyze_space_gap.py | 新脚本:双空间秩相关 | ~150 | E03 |
| G3 | build_rag_database_ze.py | --use_quantized | ~30 | E04 |
| G4 | eval 入口 + 两个 retriever | --retrieval_topk/--retrieval_pool 透传 | ~10 | E05 |
| G5 | scripts/subsample_database.py | 新脚本:分层子采样 | ~40 | E06 |
| G6 | train_query_projector.py | --objective {kl,infonce,mse} | ~40 | E08 |
| G7 | query_projector.py + 训练脚本 | --hidden {0,1024,2048} | ~20 | E08 |
| G8 | scripts/build_tmr_teacher_db.py | 新脚本:TMR 特征库 | ~80 | E08 |
| G9 | scripts/(可选) | 长尾子集构造 | ~60 | E11 |

全部实现后逐个冒烟测试(mock 数据/小样本),再排 SLURM。
