---
type: note
source: D:\tpami\V2-PLAN-A-PROGRESS.md
date: 2026-06-26
status: 已实现，V1/V2 训练对照进行中
---

## V2 Plan A 实施进度

状态：**已实现，V1/V2 训练对照进行中**（updated 2026-07-05；权威事实见 `.planning/paper-v2/PLAN-A-FACTS.md`）

### 实际实现（updated 2026-07-05，与下方原计划的差异以本节为准）

- 检索空间 = 冻结 RVQ-VAE 的**预量化连续 latent z_e**（不是离散 z_q）；ckpt 实际 **code_dim2d=1024**（非默认 512）；encoder2d 输出 (B,1024,T/4,6) → 时空 mean pool → L2 norm → 1024 维检索键。
- 检索数据库 `database_ze/`：23,384 个训练 motion 按 caption 展开为 **66,912** 对。
- QueryProjector：Linear(512→1024) + GELU + Linear(1024→1024)，输出 L2 归一；训练目标不是原计划的 cosine 对齐，而是 **KL 蒸馏**自 BMM teacher 的 top-256 soft ranking（τ=0.07；KL 优于 InfoNCE）；~1.57M 参数。
- SSTA 架构不变，仅加 Linear(1024→512) 适配层（re_motion_proj/re_text_proj）；V2 全 pipeline 由 `--use_ze_retrieval` 门控，V1 路径不动。
- 附带独立 minor contribution：**rt_in_value**（R_t 进 SSTA Value）。发表版 V1 是 K-only（V 只含 z、R_m）；latent 对齐后 R_t 已是 motion-domain 信号，"domain mismatch" 前提消失，Value 路由重新成立（「前提变化」叙事，不否定 ECCV 消融结论）。

### 训练对照状态（2026-07-05）

- V1 retrain（job 13846）vs V2（job 13845），严格单变量对照：配置一致、两边都开 rt_in_value，唯一变量 = 检索模块。
- 中间结果：V2 best FID **0.0991@ep409** vs V1 **0.1273@ep316**（MaskTransformer 单独 + 单次 eval，**不是论文数字**）。
- 论文数字必须来自 `eval_res.py`（Mask+Residual 完整 pipeline）× 20 repeats，checkpoint 显式用 ep 后缀文件（`net_best_fid_ep0409.tar` / `net_best_fid_ep0316.tar`）。

### 7 步实施计划（原计划，2026-06-26 快照）

1. **VQ-VAE 特征提取验证** — 确认 RVQVAE encoder 输出的 latent 维度/格式，决定聚合方式
   - 涉及：`models/vq/model.py`（只读）
2. **重建检索数据库** — `build_rag_database.py` 中用 RVQVAE encoder 替换 Part_TMR 编码
3. **实现 query_projector** — Linear(512, code_dim) 或小 MLP，CLIP → VQ latent space
   - 涉及：`models/rag/t2m_retriever.py`
4. **对齐训练** — 训练 query_projector，目标 cosine_distance(projector(clip), vq_encoder(motion))
   - 新文件：`train_query_projector.py`
5. **调整 SSTA 融合** — info_mlp 维度可能变化，评估是否还需要 re_text
   - 涉及：`models/transformer/semantics_modulated.py`
6. **端到端训练验证** — 新检索库 + query_projector 重训 MaskTransformer，对比 V1 指标
7. **评估与文档** — 完整评估 + 记录结果

### 决策记录
| 日期 | 决策 |
|---|---|
| 2026-06-26 | 确认 A+B 组合方案，先做 A（Zeyu 建议参考 LongLive-RAG / LatentRAG / LAnR / RADiAnce）|
| 2026-06-26 | Plan A 分 7 步实施，总改动量 ~200 行，4-5 个文件 |
| 2026-07-05 | Plan A 已实现并进入 V1/V2 对照训练；对齐目标定为 KL 蒸馏（BMM teacher，top-256，τ=0.07）；rt_in_value 定位为独立 minor contribution；Plan B 搁置（仅 future work 一句） |
