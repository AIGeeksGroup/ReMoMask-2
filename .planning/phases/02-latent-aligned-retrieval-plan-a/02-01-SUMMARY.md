---
phase: 02-latent-aligned-retrieval-plan-a
plan: 01
subsystem: analysis
tags: [vq-vae, z_e, cosine-similarity, anisotropy, geometry]

requires:
  - phase: 01-ablation-studies
    provides: pretrained RVQVAE checkpoint (net_best_fid.tar)
provides:
  - z_e dimension confirmed: code_dim2d = 1024 (not default 512)
  - z_e geometric distribution characterised (anisotropy, effective dims)
  - cosine retrieval feasibility validated (mean cos = 0.62, no collapse)
  - D-08 dimension mismatch confirmed (1024 vs CLIP 512)
affects: [02-02, 02-03, 02-04, 02-05, 02-06]

tech-stack:
  added: []
  patterns: [z_e extraction via encoder2d + mean pooling + L2 norm]

key-files:
  created:
    - ReMoMask/scripts/analyze_ze.py
    - ReMoMask/results/ze_analysis/summary.json
    - ReMoMask/results/ze_analysis/cosine_sim_hist.png
    - ReMoMask/results/ze_analysis/eigenvalue_spectrum.png
  modified: []

key-decisions:
  - "code_dim2d = 1024 confirmed from checkpoint opt.txt; query projector must output 1024-d, not 512-d"
  - "D-08 resolution: code_dim != 512 (CLIP dim), projector needs Linear(512->1024) output head"
  - "ArcVQ regularization deferred but flagged: z_e has only 11/1024 effective dims, isotropy ~0"
  - "Cosine retrieval feasible despite anisotropy: mean cos = 0.62 with good spread [-0.55, 0.998]"

patterns-established:
  - "z_e extraction: encoder2d(rearrange(pad(x2d), 'b t j d -> b d t j')) -> mean(dim=[2,3]) -> L2 norm"

requirements-completed: [LA-01]

coverage:
  - id: D1
    description: "z_e code_dim2d value confirmed from checkpoint (1024, not default 512)"
    requirement: "LA-01"
    verification:
      - kind: automated_ui
        ref: "results/ze_analysis/summary.json#code_dim2d == 1024"
        status: pass
    human_judgment: false
  - id: D2
    description: "Cosine similarity distribution visualised and analysed (mean=0.62, no collapse)"
    requirement: "LA-01"
    verification:
      - kind: other
        ref: "results/ze_analysis/cosine_sim_hist.png"
        status: pass
    human_judgment: false
  - id: D3
    description: "Isotropy score quantified (0.0) with eigenvalue spectrum, effective dims = 11/1024"
    requirement: "LA-01"
    verification:
      - kind: other
        ref: "results/ze_analysis/eigenvalue_spectrum.png + summary.json#isotropy_score"
        status: pass
    human_judgment: false

duration: 5min
completed: 2026-06-28
status: complete
---

# Phase 2 Plan 01: z_e Geometry Verification Summary

**VQ-VAE encoder z_e 维度确认 1024-d (非默认 512)，cosine 检索可行 (mean=0.62)，但存在严重各向异性 (有效维度仅 11/1024)**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-06-28T00:50:00Z
- **Completed:** 2026-06-28T00:55:00Z
- **Tasks:** 1
- **Files created:** 4

## Accomplishments

- 从 checkpoint opt.txt 确认 **code_dim2d = 1024**，推翻了代码默认值 512 的假设
- z_e 原始形状为 **(B, 1024, T/4, 6)**，mean pooling 后 **(B, 1024)**
- Cosine similarity 分布健康：mean=0.62, std=0.25, range [-0.55, 0.998]，**无 collapse**
- 特征值谱分析揭示严重各向异性：isotropy score ≈ 0，仅 7 维解释 95% 方差
- 所有分析结果保存至 `results/ze_analysis/`，含直方图和特征值谱图

## Key Findings

### 1. VQ-VAE 参数 (从 checkpoint opt.txt 读取)

| Parameter | Value | Code Default |
|-----------|-------|-------------|
| code_dim2d | **1024** | 512 |
| code_dim1d | 512 | 512 |
| nb_code2d | 256 | 512 |
| nb_code1d | 512 | 512 |
| num_quantizers | 6 | 3 |
| down_t | 2 | 2 |
| stride_t | 2 | 2 |

### 2. z_e Shape

- **编码器输出**: `(B, 1024, T/4, 6)` — 例如 196 帧输入 -> `(B, 1024, 49, 6)`
- **Mean pooling 后**: `(B, 1024)` — 这是检索向量的维度
- L2 归一化后范数 = 1.0；原始范数 mean=20.01, std=8.04

### 3. Cosine Similarity 分布

- **均值**: 0.6198
- **标准差**: 0.2547
- **中位数**: 0.6774
- **范围**: [-0.5538, 0.9983]
- **判定**: mean < 0.95 → **无严重塌缩，cosine 检索可行**

### 4. 各向异性分析

- **Isotropy score**: ≈ 0.0 (min_eig / max_eig)
- **有效维度**: 11/1024 (1.1%)，1% 阈值
- **Participation ratio**: 4.92
- **95% 方差** 仅需 **7 个维度**
- Top-3 特征值: 0.122, 0.086, 0.067
- Bottom 特征值: 0.0 (数值精度下限)

### 5. 对后续任务的影响

1. **D-08 已明确**: code_dim=1024 ≠ clip_dim=512，query projector 必须包含 `Linear(512 -> 1024)` 输出头
2. **检索可行性**: cosine similarity 有足够区分度，latent-aligned retrieval 方案成立
3. **ArcVQ 作为后备**: 各向异性严重，但当前不阻塞 — 如果 projector 对齐效果不佳，再考虑 ArcVQ 正则化
4. **有效维度集中**: 实际信息高度集中在少数维度，projector 的训练损失应能有效利用这一结构

## Files Created

- `ReMoMask/scripts/analyze_ze.py` — z_e 几何分析脚本 (只读，不修改现有代码)
- `ReMoMask/results/ze_analysis/summary.json` — 所有数值结果
- `ReMoMask/results/ze_analysis/cosine_sim_hist.png` — cosine similarity 分布直方图
- `ReMoMask/results/ze_analysis/eigenvalue_spectrum.png` — 特征值谱 + 累积方差图

## Decisions Made

1. **code_dim2d = 1024 确认** — 后续所有 LA-02~LA-06 的维度参数以此为准
2. **D-08 判定: 需要维度适配** — projector 输出 1024-d 而非直接用 512-d
3. **ArcVQ 暂缓** — 各向异性虽然严重，但 cosine retrieval 区分度足够，先跑 baseline 再看是否需要正则化

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- z_e 维度参数已完全明确，LA-02 (Query Projector) 可以开始
- D-04 中的 `Linear(512 -> D)` 现在确定 D = 1024
- 检索数据库 (LA-05) 的向量维度确定为 1024

---
*Phase: 02-latent-aligned-retrieval-plan-a*
*Completed: 2026-06-28*
