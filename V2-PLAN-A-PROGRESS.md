# ReMoMask V2 — Plan A 实施进度

**方案**: Latent-Aligned Retrieval（检索空间迁移到 VQ-VAE latent space）
**确认日期**: 2026-06-26
**状态**: Wave 1-3 完成, 训练中

---

## 实施步骤

### Step 1: VQ-VAE 特征提取验证 ✅
- [x] 确认 RVQVAE encoder 输出的 latent 维度和格式 — **code_dim2d=1024**
- [x] 验证用 `rvqvae.encode()` 提取的 1D/2D latent 是否适合做检索 key
- [x] 决定 latent 的聚合方式：时间维 mean pool？还是保留序列？
- **涉及文件**: `models/vq/model.py` (只读，不改)
- **产出**: latent 格式确认文档

### Step 2: 重建检索数据库 ✅
- [x] 修改 `build_rag_database.py`：用 RVQVAE encoder 替换 Part_TMR 编码
- [x] 生成新的 `encoded_motions.npy`（VQ latent space） — **database_ze/ 66912 samples**
- [x] 保留 `encoded_texts.npy`（CLIP 编码，后续 Step 3 需要）
- [x] 验证新数据库的覆盖率和维度正确性
- **涉及文件**: `build_rag_database.py`
- **产出**: 新的 RAG 数据库文件

### Step 3: 实现 query_projector ✅
- [x] 新增 `query_projector`：Linear(512, code_dim) 或小 MLP — **架构 512→1024, retraining on SLURM**
- [x] 把 CLIP text embedding 投影到 VQ latent space
- [x] 替换 `MocoTmrRetriever` 中的 `encode_text` 调用
- **涉及文件**: `models/rag/t2m_retriever.py`
- **产出**: 新的检索 query 路径

### Step 4: 对齐训练 ✅
- [x] 编写 query_projector 的对齐训练脚本 — **SLURM job 13739**
- [x] 训练目标：cosine_distance(projector(clip_text), vq_encoder(motion).mean())
- [x] 用训练集的 (text, motion) 对训练
- [ ] 验证对齐质量：检索 recall@K
- **涉及文件**: 新文件 `train_query_projector.py`
- **产出**: 训练好的 query_projector checkpoint

### Step 5: 调整 SSTA 融合 ✅
- [x] 评估 `info_mlp` 是否需要调整维度（3D→D 可能变化） — **retrieval_dim=1024**
- [x] 决定是否还需要 `re_text`（同空间下可能只需 `re_motion`）
- [x] 如果维度变化，更新 `SemanticsModulatedAttention.__init__`
- **涉及文件**: `models/transformer/semantics_modulated.py`
- **产出**: 更新后的 SSTA 模块

### Step 6: 端到端训练验证 🔄
- [x] 用新的检索库 + query_projector 重新训练 MaskTransformer — **SLURM job 13740**
- [ ] 对比 V1 的 FID / R-Precision 指标
- [ ] 消融实验：V1 检索 vs Plan A 检索（其他不变）
- **涉及文件**: `train_mask_transformer_ddp.py`, `transformer_trainer_ddp.py`
- **产出**: 对比实验结果

### Step 7: 评估与文档
- [ ] 完整评估 (eval_mask.py / eval_res.py)
- [ ] 记录实验结果
- [ ] 更新 README

---

## Plan B（后续）
待 Plan A 验证效果后启动。核心改动在 `transformer_ts.py:generate()` 循环体内增加动态检索。

---

## 决策记录

| 日期 | 决策 | 备注 |
|------|------|------|
| 2026-06-26 | 确认 A+B 组合方案，先做 A | Zeyu 建议参考 LongLive-RAG / LatentRAG / LAnR / RADiAnce |
| 2026-06-26 | Plan A 分 7 步实施 | 总改动量 ~200 行，4-5 个文件 |
| 2026-06-28 | ABL-02 证实 rt_in_value=True (FID -13%), Phase 2 默认采用 | Phase 1 消融结果 |
| 2026-06-30 | code_dim2d=1024 确认, projector 架构 512→1024 | LA-01 z_e 分析 |
| 2026-06-30 | 远程 SLURM 集群 (diana) 配置完成, projector+V2 训练提交 | Job 13739, 13740 |
