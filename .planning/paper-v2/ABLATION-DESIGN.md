# ReMoMask-2 消融设计(已入文,2026-07-05)

> 状态:全部四张消融表 + dummy 分析已融入 pami.tex(subsec:ablation_v2,位于原有
> Ablation Study 内容之后、Qualitative Results 之前);实现细节段落已追加到
> Implementation Details。数值全为占位符,登记于 PLACEHOLDERS.md。

## 已落地的表

| 表 | label | 内容 | 优先级 |
|---|---|---|---|
| 正交主消融 | tab:ablation_v2_orthogonal | {Semantic(HBM), Latent-aligned(z_e)} × {R_t in Value: ✗/✓},FID/Top1/MMDIST,末行黄色高亮=完整 ReMoMask-2 | P0(对应集群 G3 消融) |
| 对齐目标 | tab:ablation_v2_alignment | KL distillation vs InfoNCE,projector R@k + 下游 FID | P1 |
| 蒸馏 teacher | tab:ablation_v2_teacher | HBM vs TMR teacher,R@k + FID | P1 |
| 敏感性 | tab:ablation_v2_sensitivity | 蒸馏 top-κ {64,256,1024} + projector 容量 {Linear, MLP-1024, MLP-2048},仅 R@k | P2(便宜,只需重训 projector) |
| mask-only 诊断 | tab:ablation_v2_maskonly | {ReMoMask, ReMoMask-2} × {mask-only, full pipeline},FID/Top1(2026-07-05 应用户要求新增,eval_mask.py × 20 repeats 即可,无需新训练) | P1(源于用户 mask-only 之问;论证增益出现在 coarse token 阶段) |

## dummy 分析的核心假设(结果不符时的改写预案)

1. 检索空间是主导因素,rt_in_value 是次级增益,二者近似可加 → 若 rt 在 semantic 行为负,
   叙事退回「conference 结论在原 setting 下成立,latent 对齐后前提变化」,表格保持,分析段改写第二段。
2. KL > InfoNCE(有 LatentRAG 旁证 43.46 vs 41.86)→ 若反转,删「label noise」论证段,改为经验性陈述。
3. HBM teacher > TMR teacher(teacher 检索差距 18.49 vs 5.68 支撑)→ 风险最低。
4. top-κ/容量不敏感 → 若敏感,把「stems from where, not capacity」结论软化。
5. mask-only 诊断:V2 优势在 mask-only 阶段已出现、且相对幅度与 full pipeline 相当 →
   若不符(优势主要在 residual 之后才出现),分析段改写为「增益由检索条件与精修阶段共同实现」,
   删去「improves the coarse token prediction itself」的强 claim。

## 候选扩充(2026-07-05 提议,未入文,等用户挑选)

| # | 实验 | 防御点 | 成本 | 建议 |
|---|---|---|---|---|
| 1 | z_e vs z_q 检索键 | LOCKED-01 的实验支撑,"为什么不用离散 token" | z_q 库重建 ~5min + eval | 强烈加 |
| 2 | MSE regression 基线进对齐表 | 方法正文已 claim regression 不行,需证据 | projector 重训 ×1 | 强烈加 |
| 3 | 跨空间秩相关(Spearman/top-k overlap, S vs Z) | 把 representation gap 从修辞变测量;价值最高 | 纯离线,零训练 | 强烈加 |
| 4 | 推理期检索条数 k 消融 | RAG 经典旋钮,ECCV 版缺失 | 纯推理 | 强烈加 |
| 5 | db coverage 曲线补 V2 | 对齐放大检索效用的叙事 | 纯推理 | 强烈加 |
| 6 | 效率散点补 ReMoMask-2 点 | 不补显得心虚 | 一次计时 | 强烈加 |
| 7 | 检索近邻可视化(同 prompt 两空间对比) | gap 最直观定性证据,期刊版新图 | 离线检索+渲染 | 考虑 |
| 8 | 罕见/长尾 prompt 子集评估 | RAG 鲁棒性卖点 | 协议设计+eval,中等 | 考虑 |

⚠️ 隐含承诺提醒:主表 KIT-ML/SnapMoGen 的 ReMoMask-2 占位行意味着两数据集各需完整训一遍 V2
(+检索库+projector);不跑则主表只留 HumanML3D 行 → 进拍板清单。

明确不加:VQ-VAE 联合微调、ArcVQ(future work);TSM/HBM 重消融(2×2 已覆盖);CFG schedule(已弃)。

## 跑数需求(对应 GOAL.md G3)

- P0 四格:V1_rtval(已在训=Semantic✓行)、V2_rtval(已在训=末行)、还差 Semantic✗(=原版 V1 retrain,
  可考虑用原 ckpt 或补跑)与 z_e✗ 两格 → 每格 eval_res.py × 20 repeats。
- P1 两表:各需重训 projector(200ep 单卡,便宜)+ 重训或复用 MaskTransformer?
  ——注意:严格口径应重训下游,预算不够时先报 projector R@k、下游 FID 标注 shared-backbone 口径,写作时再定。
- P2:仅 projector 重训 + R@k,最便宜。

## 附:起草 agent 的原始锚点与风险说明
INSERTION ANCHORS (D:/tpami/currentversion/v2working/pami.tex, current line numbers):
- Blocks 0-5 (new subsection + 4 tables + prose): insert after the "Complete Retrieval Ablation" table (ends `\end{table}` at ~line 1780, label tab:rag_ablation) and before `\subsection{Qualitative Results}` at line 1783. Tables float; keep them in source order near the subsection so first references resolve nearby.
- Block 6: append inside `\subsection{Implementation Details}` (line 1530), immediately after "All models are implemented in PyTorch." (line 1536). It is written to be hardware-silent for the masked-model retrain, so it cannot contradict the existing "8 Tesla A800" sentence.

SUGGESTED NEW ROWS FOR tab:notation (line ~1947; add after the $R_t$ row or at the end, same two-column format):
  $z_e$ & Pre-quantization motion latent from the frozen 2D-RVQ-VAE encoder; pooled and $\ell_2$-normalized as retrieval key \\
  $\phi$ & Query projector mapping the CLIP text embedding into the $z_e$ space \\
  $q$ & Projected text query, $q = \phi(t)$ \\
  $p_{\mathrm{HBM}}, p_{\phi}$ & Teacher and student retrieval distributions over top-ranked candidates in KL distillation \\

DESIGN RATIONALE:
- 2x2 orthogonal table is the load-bearing evidence for BOTH the latent-alignment claim and the "premise changed" rt_in_value narrative: main effect (rows) = retrieval space, second factor (checkmark column) = routing; near-additive composition = independence claim. First row doubles as the conference configuration retrained under the journal protocol, which pre-empts "is the gain just retraining?" while the caption wording ("retrained under the journal protocol") blocks direct comparison against the published 0.026 in tab:t2m_experiment.
- Alignment-objective and teacher tables carry the two LOCKED design decisions (KL over InfoNCE; HBM over TMR). Per the placeholder policy I did NOT put the preliminary 43.46 vs 41.86 numbers in the table — V2-REVISION-PLAN marks them "占位重验" (to re-verify); the table stays xx.xx.
- Sensitivity table included deliberately (TPAMI reviewers expect it, and the paper already has fig:hyperparameter for HBM, so it matches house convention). Kept retrieval-only (no FID column) so the sweep needs only cheap projector retrains, not masked-model retrains.
- Prose uses only FACTS-listed measured numbers: cosine mean 0.62 / std 0.25 / range [-0.55, 0.998], 11/1024 effective dims, 7 dims -> 95% variance, 66,912 pairs / 23,384 motions, top-256, tau 0.07; plus numbers already printed in pami.tex (0.104 vs 0.027 from tab:ablation; 18.49 vs 5.68 from tab:rag_experiment). Everything else is \textbf{[TBD]}.
- Highlighting convention: yellow!20 reserved for the single full ReMoMask-2 row (matches tab:delta line 655); \rowcolor[gray]{0.90} for adopted-default rows (matches existing ablation tables).

RISKS / OPEN QUESTIONS:
1. Naming: I wrote "Semantic (HBM)" rather than "Part_TMR" — the paper never uses the code name Part_TMR; HBM is the published name of the V1 retriever/space. Flip if you prefer.
2. Deferred cross-references: two TODO-REF comments — (a) refs to Sec. 4.6/4.7 (suggested labels subsec:latent_retrieval / subsec:rt_value, to be created by the methodology draft), (b) Fig.~\ref{fig:ze_geometry} in the geometry paragraph (label to be created by the Design Principles addition). Both are commented out, so the file compiles before those drafts land.
3. Downstream-FID protocol for tab:ablation_v2_alignment / tab:ablation_v2_teacher: decide before filling whether each variant projector implies (a) a full masked-model retrain (clean but expensive) or (b) plug-in evaluation with the frozen ReMoMask-2 masked model (cheap; note that swapping phi also changes the stored R_t database embeddings, since encoded_texts.npy is a projector re-projection). If (b), add one protocol sentence to both captions.
4. Semantic+routing row direction: prose currently assumes a small positive gain (per the dummy policy) and attributes it to the longer journal protocol without disputing the ECCV tab:ablation finding. If real numbers show semantic+routing neutral/negative, only the last sentence of the second routing paragraph needs replacing (it then STRENGTHENS the premise-changed narrative — even easier).
5. Param counts in the sensitivity table: 0.53M (Linear 512->1024) and 3.15M (hidden 2048) are computed from the architecture incl. biases; 1.57M is from FACTS. Verify 0.53M/3.15M when the variants are actually instantiated.
6. Symbol clash: Top-$K$ vs body-part count $K$ (tab:notation) — disambiguated in the sensitivity caption; if you prefer zero ambiguity, rename to $K_{\mathrm{KD}}$ in table + prose.
7. TODO-CITE placeholders for Hinton-KD and InfoNCE (neither key exists in reference.bib); both are near-mandatory citations for the Alignment Objective paragraph.
8. tab:ablation_v2_orthogonal reports FID/Top1/MMDIST only (per task spec), matching tab:ablation_ssta's column set; if reviewers ask for Diversity/MModality, columns can be appended without redesign.
