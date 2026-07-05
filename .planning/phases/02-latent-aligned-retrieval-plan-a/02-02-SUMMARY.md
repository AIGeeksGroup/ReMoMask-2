---
phase: 02-latent-aligned-retrieval-plan-a
plan: 02
subsystem: retrieval-database
tags: [z_e, rvqvae, retrieval, database, clip]

requires:
  - phase: 02-latent-aligned-retrieval-plan-a
    plan: 01
    provides: code_dim2d = 1024 confirmed
provides:
  - z_e retrieval database (66912 samples, 1024-d motion vectors)
  - ZeRetriever class with MocoTmrRetriever-compatible interface
  - CLIP text features as placeholder for query projector (LA-03)
affects: [02-03, 02-04, 02-05, 02-06]

tech-stack:
  added: []
  patterns: [streaming batch VQ encode, CLIP batch text encode, database_ze format]

key-files:
  created:
    - ReMoMask/build_rag_database_ze.py
    - ReMoMask/models/rag/ze_retriever.py
    - ReMoMask/database_ze/encoded_motions.npy
    - ReMoMask/database_ze/encoded_texts_clip.npy
    - ReMoMask/database_ze/all_captions.npy
    - ReMoMask/database_ze/motion_ids.npy
    - ReMoMask/database_ze/tag_lists.npy
    - ReMoMask/database_ze/motion_tokens.npy
    - ReMoMask/database_ze/metadata.json
  modified: []

key-decisions:
  - "Streaming batch encoding: accumulate x2d in batches of 64, encode, discard - avoids 8GB memory for full dataset"
  - "CLIP ViT-B/32 text features stored as placeholder; to be replaced after LA-03 projector training"
  - "Cross-space fallback in ZeRetriever: when projector absent, truncates database to match query dim for cosine sim"
  - "motion_tokens.npy copied from existing database/ (same tokenization, different retrieval features)"

requirements-completed: [LA-02]

coverage:
  - id: DB1
    description: "database_ze/encoded_motions.npy shape (66912, 1, 1024) matches code_dim2d"
    requirement: "LA-02"
    verification:
      - kind: automated_ui
        ref: "python -c 'import numpy as np; m=np.load(\"database_ze/encoded_motions.npy\"); print(m.shape)'"
        status: pass
    human_judgment: false
  - id: DB2
    description: "ZeRetriever.forward() returns re_dict with correct shapes"
    requirement: "LA-02"
    verification:
      - kind: automated_ui
        ref: "ZeRetriever test: re_motion=(2,2,1,1024), re_text=(2,2,1,512)"
        status: pass
    human_judgment: false
  - id: DB3
    description: "metadata.json records code_dim2d=1024"
    requirement: "LA-02"
    verification:
      - kind: automated_ui
        ref: "database_ze/metadata.json#code_dim2d == 1024"
        status: pass
    human_judgment: false

duration: 7min
completed: 2026-06-28
status: complete
---

# Phase 2 Plan 02: Rebuild Retrieval Database with z_e Summary

**用 RVQVAE encoder2d 的 z_e 向量重建检索数据库，66912 样本，1024-d motion + 512-d CLIP text，ZeRetriever 通过验证**

## Performance

- **Duration:** ~7 min (5 min database build + 2 min code/test)
- **Started:** 2026-06-28T06:58:00Z
- **Completed:** 2026-06-28T07:05:00Z
- **Tasks:** 2/2
- **Files created:** 9 (2 source + 7 database)

## Accomplishments

### Task 1: z_e 数据库构建脚本

- 新建 `build_rag_database_ze.py`，三阶段流水线：
  1. **收集阶段**: 遍历 23384 个训练 motion，按 caption 展开为 66912 对 (motion, caption)
  2. **VQ 编码**: 流式批量 (batch=64) 调用 encoder2d -> mean pool (T/4, 6) -> L2 normalize，输出 (N, 1, 1024)
  3. **CLIP 编码**: 批量 (batch=256) encode 所有 caption，输出 (N, 1, 512)
- 从已有 `database/motion_tokens.npy` 复制 token 数据
- 数据库总大小 ~650MB，构建耗时 5.16 分钟

### Task 2: ZeRetriever

- 新建 `models/rag/ze_retriever.py`，接口与 MocoTmrRetriever 完全兼容
- `forward(captions)` 返回 `re_dict = {"re_motion": (B,K,1,1024), "re_text": (B,K,1,512)}`
- 支持 query_projector 注入：有 projector 时走 CLIP -> projector -> z_e 空间检索；无 projector 时走跨空间 fallback
- 保留了 de-duplication (按 motion_id 去重) 和 shuffle 逻辑

## Database Statistics

| File | Shape | Description |
|------|-------|-------------|
| encoded_motions.npy | (66912, 1, 1024) | z_e vectors, L2-normalized |
| encoded_texts_clip.npy | (66912, 1, 512) | CLIP ViT-B/32 text features |
| all_captions.npy | (66912,) | caption strings |
| motion_ids.npy | (66912,) | motion IDs (format: name_idx) |
| tag_lists.npy | (66912, 2) | f_tag, to_tag pairs |
| motion_tokens.npy | dict | copied from database/ |
| metadata.json | - | code_dim2d=1024, encoder_type=rvqvae_2d |

## Verification Results

1. `encoded_motions.npy` shape = (66912, 1, 1024) — matches code_dim2d
2. L2 norm of all motion vectors = 1.000000 (std=0.000000) — correctly normalized
3. ZeRetriever.forward(['walk', 'jump']) returns re_motion=(2,2,1,1024), re_text=(2,2,1,512)
4. metadata.json reports code_dim2d=1024, encoder_type=rvqvae_2d

## Decisions Made

1. **流式批量编码** — 不一次性加载所有 x2d 到内存 (~8GB)，而是攒够 batch_size=64 就 encode 并释放
2. **CLIP text 作为占位** — LA-03 训完 projector 后替换；当前跨空间检索能跑通但精度有限
3. **跨空间 fallback 策略** — 无 projector 时截断 database 维度匹配 query dim，粗糙但可用

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Unicode encoding error on Windows**
- **Found during:** Task 1 initial run
- **Issue:** `codecs.open()` defaults to GBK on Windows, some text files contain non-GBK bytes
- **Fix:** Added `encoding='utf-8', errors='replace'` to text file reads
- **Files modified:** build_rag_database_ze.py

**2. [Rule 3 - Blocking] Memory explosion with full dataset x2d collection**
- **Found during:** Task 1 second run
- **Issue:** Collecting all 66912 x2d tensors (196x22x12 each) in memory before encoding caused ~8GB allocation, swap thrashing, and extreme slowdown (from 1100 it/s to 50 it/s)
- **Fix:** Switched to streaming batch pattern: accumulate batch_size x2d, encode, clear buffer. Memory stays constant at ~batch_size * 200KB
- **Files modified:** build_rag_database_ze.py

## Files Created

- `ReMoMask/build_rag_database_ze.py` — database builder (324 LOC)
- `ReMoMask/models/rag/ze_retriever.py` — retriever class (239 LOC)
- `ReMoMask/database_ze/` — 7 database files (~650MB total)

---
*Phase: 02-latent-aligned-retrieval-plan-a*
*Completed: 2026-06-28*
