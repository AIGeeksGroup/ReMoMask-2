---
phase: 02-latent-aligned-retrieval-plan-a
plan: 04
subsystem: transformer/SSTA
tags: [dimension-adaptation, retrieval, projection]
dependency_graph:
  requires: [02-01]
  provides: [ssta-retrieval-dim-support]
  affects: [02-05, 02-06]
tech_stack:
  added: []
  patterns: [conditional-projection-layer, identity-fallback]
key_files:
  created: []
  modified:
    - ReMoMask/models/transformer/semantics_modulated.py
    - ReMoMask/models/transformer/transformer_ts.py
decisions:
  - "保留 info_mlp 3-way 结构（text + R_m + R_t），不合并为 2-way，因为 R_m 和 R_t 承载不同信号"
  - "retrieval_dim=None 时用 nn.Identity 而非跳过投影，保持 forward 逻辑统一"
metrics:
  duration: ~5min
  completed: 2026-06-28
status: complete
---

# Phase 02 Plan 04: SSTA Dimension Adaptation Summary

SSTA 新增 retrieval_dim 投影层，将 z_e 空间的 1024 维检索特征映射到 latent_dim=512，兼容 V1 checkpoint。

## What Was Done

### Task 1: SemanticsModulatedAttention 投影层 (465c66c)

- `__init__` 新增 `retrieval_dim=None` 参数
- `retrieval_dim != latent_dim` 时创建 `re_motion_proj` 和 `re_text_proj`（`nn.Linear(retrieval_dim, latent_dim)`）
- `retrieval_dim` 为 None 或等于 `latent_dim` 时使用 `nn.Identity()`（V1 行为不变）
- `forward()` 在 squeeze 后、pooling 前应用投影，后续 info_mlp / attention 逻辑无需修改

### Task 2: MaskTransformer2D 参数传递 (cd229c2)

- `__init__` 新增 `retrieval_dim=None` 参数，存入 `self.retrieval_dim`
- SSTA 构造 cfg dict 中加入 `'retrieval_dim': retrieval_dim`
- 8 层 SSTA 全部接收到 retrieval_dim 配置

## Verification Results

| Case | retrieval_dim | Input re_dict D | Output Shape | Status |
|------|--------------|-----------------|-------------|--------|
| V1 compat | None | 512 | (2, 294, 512) | PASS |
| V2 1024d | 1024 | 1024 | (2, 294, 512) | PASS |
| V2+rt_in_value | 1024 | 1024 | (2, 294, 512) | PASS |
| MaskTransformer2D V1 | None | - | Identity proj | PASS |
| MaskTransformer2D V2 | 1024 | - | Linear(1024,512) | PASS |
| End-to-end user test | 1024 | 1024 | (2, 48, 512) | PASS |

## Backward Compatibility

- `retrieval_dim=None`（默认值）→ 所有投影层为 `nn.Identity` → V1 行为完全不变
- V1 checkpoint 用 `strict=False` 加载时，`re_motion_proj` 和 `re_text_proj` 为 missing keys（可接受，V2 需重训）

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED
