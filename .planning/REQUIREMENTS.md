# Requirements: ReMoMask V2

**Defined:** 2026-06-27
**Core Value:** 检索空间与生成空间统一后，FID 必须优于 V1 独立空间方案

## v1 Requirements

### 零成本消融（前置验证）

- [ ] **ABL-01**: Step-dependent RAG-CFG schedule 消融：10 步中 s 值从高到低递减 vs 固定 s=4，对比 FID/R-Precision
- [ ] **ABL-02**: R_t in K vs K+V 消融：SSTA 的 Value 分支加入 R_t 对比现有 K-only 设计

### Plan A — Latent-Aligned Retrieval

- [ ] **LA-01**: 验证 VQ-VAE encoder 输出 z_e 的几何分布（cosine similarity 分布、各向异性程度、dimensional collapse 检查）
- [ ] **LA-02**: 用 RVQVAE encoder 替换 Part_TMR 编码，重建检索数据库（encoded_motions.npy 从 Part_TMR 512d → z_e 空间）
- [ ] **LA-03**: 实现 query_projector（CLIP 512d → z_e 空间），用 KL 散度对齐训练，BMM 当 teacher
- [ ] **LA-04**: 调整 SSTA info_mlp 维度适配新检索空间（可能从 3D→D 变为 2D→D）
- [ ] **LA-05**: 端到端训练：新检索库 + query_projector + MaskTransformer，三数据集评估
- [ ] **LA-06**: Plan A 消融：V1 Part_TMR 检索 vs V2 z_e 检索（其他不变），对比 FID/R-Prec/MM-Dist

### Plan B — Iterative Dynamic Retrieval

- [ ] **IR-01**: 在 transformer_ts.py:generate() 中实现 50% 时定点精检索（用已确定 token 构造新 query）
- [ ] **IR-02**: 训练时模拟动态检索避免 train-test mismatch
- [ ] **IR-03**: Plan B 消融：静态 1 次 vs 动态 1 次（50%）vs 动态 2 次（33%+67%）

### 综合消融

- [ ] **COMB-01**: 2×2 消融矩阵完成（Part_TMR/z_e × 静态/动态），四组配置全部评估
- [ ] **COMB-02**: 三数据集完整评估（HumanML3D + KIT-ML + SnapMoGen），20 次重复

## v2 Requirements

### 可选增强（视 v1 结果决定）

- **OPT-01**: Retrieval confidence-gated SSTA（log(s_i) bias）— 若 Plan A 检索质量波动大
- **OPT-02**: ArcVQ-VAE codebook 正则化 — 若 z_e 各向异性严重
- **OPT-03**: 联合训练 VQ-VAE + 对比损失 — 若冻结 VQ-VAE 的检索 ceiling 不够

## Out of Scope

| Feature | Reason |
|---------|--------|
| VQ-VAE 架构替换（SoftVQ-VAE 等） | 改动太大，偏离期刊拓展版范围 |
| 新数据集采集或合成 | 用现有三数据集，与 V1 保持可比性 |
| 每步检索（Plan B 原始版） | 收窄为 50% 时一次定点精检索 |
| LAnR 式 entropy control head | 过于复杂，收益不确定 |
| 论文写作 | 代码实现完成后再处理 |

## Traceability

| REQ-ID | Phase | Status |
|--------|-------|--------|
| ABL-01 | Phase 1: Zero-Cost Ablations | Pending |
| ABL-02 | Phase 1: Zero-Cost Ablations | Pending |
| LA-01 | Phase 2: Latent-Aligned Retrieval | Pending |
| LA-02 | Phase 2: Latent-Aligned Retrieval | Pending |
| LA-03 | Phase 2: Latent-Aligned Retrieval | Pending |
| LA-04 | Phase 2: Latent-Aligned Retrieval | Pending |
| LA-05 | Phase 2: Latent-Aligned Retrieval | Pending |
| LA-06 | Phase 2: Latent-Aligned Retrieval | Pending |
| IR-01 | Phase 3: Iterative Dynamic Retrieval | Pending |
| IR-02 | Phase 3: Iterative Dynamic Retrieval | Pending |
| IR-03 | Phase 3: Iterative Dynamic Retrieval | Pending |
| COMB-01 | Phase 4: Comprehensive Evaluation | Pending |
| COMB-02 | Phase 4: Comprehensive Evaluation | Pending |

---
*Defined: 2026-06-27*
