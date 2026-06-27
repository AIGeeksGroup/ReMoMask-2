<!-- refreshed: 2026-06-27 -->
# Architecture

**Analysis Date:** 2026-06-27

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT: Text Prompt                                 │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Text Encoding & Retrieval                                          │
│  • CLIP tokenization                                                         │
│  • Part_TMR text encoder  `Part_TMR/models/builder_bimoco.py`               │
│  • Database retrieval  `models/rag/t2m_retriever.py`                        │
│  ├─ Re-Dict: {re_motion: (B, K, 1, D), re_text: (B, K, 1, D)}              │
│  └─ Caption embedding: (B, D)                                               │
└────────────────────────────┬───────────────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌──────────────────────┐  ┌─────────────────┐  ┌────────────────────┐
│  Stage 2: 1D Mask    │  │  Stage 2: 2D    │  │  Length Estimator  │
│  Transformer (Aux)   │  │  Mask Transformer  │  `models/vq/model.py`│
│  `transformer_aux.py`│  │  `transformer_ts.py` │  (B, D) → (B, 50) │
│  • Pure CLIP text    │  │  ← Uses Retrieval  │                    │
│  • Output: 1D codes  │  │  • SemanticsModAtten│                    │
│  (B, seq_len)        │  │  • Output: 2D codes│ → (B, seq_len)    │
└──────────┬───────────┘  │  (B, seq_len, J) │                    │
           │              └─────────┬────────┘  └────────────────────┘
           │                        │
           │                        │ J=6 joint dimension
           │                        │ re_dict passed to forward()
           └────────────┬───────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 3: Residual Transformers (Refinement)                                 │
│  • ResidualTransformer (1D aux)  `transformer_aux.py`                       │
│  • ResidualTransformer2D (2D)    `transformer_ts.py`                        │
│  ├─ Input: masked codes from Stage 2                                        │
│  ├─ Refine through multiple quantizers                                      │
│  └─ Output: refined codes (B, seq_len) & (B, seq_len, J)                  │
└────────────────────────────┬───────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 4: RVQVAE Decoder                                                     │
│  `models/vq/model.py` :: RVQVAE.forward_decoder()                           │
│  ├─ Input: (mids_aux, mids_ts) — refined codes                             │
│  ├─ Decodes through residual VQ structure                                   │
│  └─ Output: motion sequences (B, seq_len, 263)                             │
└────────────────────────────┬───────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 5: Post-Processing & Rendering                                       │
│  • Inverse normalization (mean/std)                                         │
│  • Motion recovery from RIC format `utils/motion_process.py`                │
│  • BVH export `visualization/joints2bvh.py`                                 │
│  • Animation/video rendering  `render.py`                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Text Encoder (CLIP) | Tokenize and embed text prompts to 512-d vectors | `models/transformer/transformer_ts.py` (load_and_freeze_clip) |
| Part_TMR | Encode database motions/captions to 512-d vectors; load pre-trained checkpoint | `Part_TMR/models/builder_bimoco.py` |
| RAG Retriever | Query database; return K nearest motions & captions by similarity | `models/rag/t2m_retriever.py` (MocoTmrRetriever) |
| MaskTransformer (1D) | Iterative decoding for 1D codes using only text conditioning | `models/transformer/transformer_aux.py` |
| MaskTransformer2D | Iterative decoding for 2D codes with Semantics Modulated Attention using retrieval | `models/transformer/transformer_ts.py` (MaskTransformer2D) |
| SemanticsModulatedAttention | Modulate query-key-value attention with retrieval features (text & motion) | `models/transformer/semantics_modulated.py` |
| ResidualTransformer (1D & 2D) | Refine coarse codes through multiple quantizer levels | `models/transformer/transformer_aux.py` / `transformer_ts.py` (ResidualTransformer classes) |
| RVQVAE | Hierarchical VQ-VAE with residual quantizers; decode codes to motion | `models/vq/model.py` (RVQVAE) |
| Database Builder | Encode full motion dataset into vectors; cache for fast retrieval | `build_rag_database.py` |

## Pattern Overview

**Overall:** Five-stage hierarchical text-to-motion generation with retrieval-augmented generation (RAG)

**Key Characteristics:**
- **Latent hierarchy:** 1D codes (temporal) + 2D codes (joint space); refined through quantizer levels
- **Retrieval-guided generation:** MaskTransformer2D fuses retrieved motion/text features into attention via SemanticsModulatedAttention
- **Masking-based iterative decoding:** BERT-style mask scheduling (cosine annealing) for both generation and refinement
- **Multi-scale conditioning:** Text prompt + retrieved neighbors + conditioning signals (presence of text/retrieval flags)

## Layers

**Text Encoding & Retrieval (Stage 1):**
- Purpose: Convert text prompts to embeddings; retrieve similar motions from database
- Location: `Part_TMR/models/builder_bimoco.py`, `models/rag/t2m_retriever.py`
- Contains: Text encoder (transformer-based), similarity scorer, motion/text database
- Depends on: Pre-trained MoCoTMR checkpoint
- Used by: MaskTransformer (auxiliary & 2D) for conditioning and guidance

**Mask Transformers (Stage 2):**
- Purpose: Iteratively decode VQ-VAE token sequences via masking strategy
- Location: `models/transformer/transformer_aux.py`, `models/transformer/transformer_ts.py`
- Contains: Token embedding, positional encoding, transformer encoder, output projection, generation loop
- Depends on: Text embeddings, retrieval dict (for 2D variant), VQ codebook size
- Used by: Main inference pipeline (`demo.py`, evaluation scripts)

**Semantics Modulated Attention Layer (Stage 2b):**
- Purpose: Inject retrieved motion/text features into self-attention for 2D decoding
- Location: `models/transformer/semantics_modulated.py` (SemanticsModulatedAttention)
- Contains: Info fusion MLP, query/key/value projections, conditioning flags
- Depends on: Motion & text features from retriever (re_dict)
- Used by: MaskTransformer2D encoder stack

**Residual Transformers (Stage 3):**
- Purpose: Refine coarse codes through hierarchical quantizer levels
- Location: `models/transformer/transformer_aux.py`, `models/transformer/transformer_ts.py`
- Contains: Similar architecture to mask transformers; processes partial codes iteratively
- Depends on: Coarse codes from prior stage, quantizer count from VQ config
- Used by: Main inference pipeline after stage 2

**RVQVAE (Stage 4):**
- Purpose: Decode token sequences to continuous motion representations
- Location: `models/vq/model.py` (RVQVAE class)
- Contains: Encoder, decoder, residual quantizers, codebooks
- Depends on: Pre-trained checkpoint with codebook embeddings
- Used by: Final decoding step in inference

## Data Flow

### Primary Request Path (demo.py)

1. Parse options and load configs  (`demo.py:166-170`)
2. Load all models: VQ → M-Transformer → R-Transformer → Length Estimator → RAG (`demo.py:191-243`)
3. Parse text prompts; estimate or use specified motion length  (`demo.py:256-288`)
4. **RAG retrieval** → `retriever(captions)` returns `re_dict`  (`demo.py:300-307`)
5. Encode text prompts via retriever  (`demo.py:306-307`)
6. **For each repeat:**
   1. Generate 1D codes: `mask_transformer_aux.generate(...)`  (`demo.py:314-319`)
   2. **Generate 2D codes with retrieval:** `mask_transformer_ts.generate(..., re_dict=re_dict, n_j=J)`  (`demo.py:320-325`)
   3. Refine 1D: `res_transformer_aux.generate(mids_aux, ...)`  (`demo.py:326`)
   4. Refine 2D: `res_transformer_ts.generate(mids_ts, ...)`  (`demo.py:327`)
   5. Decode both: `vq_model.forward_decoder(mids_aux, mids_ts)` → motion tensors  (`demo.py:328`)
   6. Inverse normalize, save, render  (`demo.py:330-356`)

### Retrieval-Guided Decoding (MaskTransformer2D.generate with re_dict)

1. Initialize mask schedule (cosine annealing)  (`transformer_ts.py:560-562`)
2. **For each diffusion timestep t:**
   1. Mask positions with probability `noise_schedule(t)`
   2. Call `trans_forward()` passing `re_dict`  (`transformer_ts.py:582-586`)
   3. Collect logits from SemanticsModulatedAttention layers
   4. Sample new tokens using top-k + temperature
3. Return final token sequence

### Forward Pass in MaskTransformer2D.trans_forward (with retrieval)

1. Embed motion codes & apply 2D positional encoding  (`transformer_ts.py:341-353`)
2. **For each SemanticTransEncoderLayer in semanticTransEncoder:**
   - Pass `(x, xf, src_mask, cond_type, re_dict)` to module  (`transformer_ts.py:362-365`)
   - SemanticsModulatedAttention fuses retrieval into attention:
     - Extract `re_motion` (K retrieved motion features)
     - Extract `re_text` (K retrieved text features)
     - Pool K features; fuse with query text via MLP info fusion
     - Modulate key/value with fused info
3. Output processed tokens → logits  (`transformer_ts.py:395`)

**State Management:**
- **Conditioning flags:** `cond_type` (B, 1, 1) controls text/retrieval usage (01 = text only, 11 = both)
- **Motion masking:** Padding tokens (id=pad_id) excluded from loss/generation
- **Retrieval cache:** `re_dict` computed once per prompt; reused across all timesteps

## Key Abstractions

**Masking Strategy (Diffusion-Style):**
- Purpose: Iteratively replace tokens with high/low confidence predictions
- Examples: `demo.py:300-327`, `transformer_ts.py:560-620`
- Pattern: Noise schedule → mask subset → forward pass → sample → update scores

**Retrieval Database:**
- Purpose: Pre-computed vectors for fast nearest-neighbor search
- Examples: `build_rag_database.py`, `models/rag/t2m_retriever.py`
- Pattern: Encode dataset once; store as `.npy` files; load at inference; index by cosine similarity

**Conditioning Mechanism:**
- Purpose: Gate text/retrieval signals based on generation mode
- Examples: `semantics_modulated.py:69-91` (cond_type masking)
- Pattern: Flags control whether features are zeroed out (unconditional path for cfg)

## Entry Points

**demo.py:**
- Location: `D:\tpami\ReMoMask\demo.py`
- Triggers: `python demo.py --text_prompt "..." [--motion_length N]`
- Responsibilities: Load all models, run inference loop, render output

**train_mask_transformer_ddp.py:**
- Location: `D:\tpami\ReMoMask\train_mask_transformer_ddp.py`
- Triggers: `python train_mask_transformer_ddp.py ...`
- Responsibilities: Training loop for M-Transformer on masked token prediction

**train_res_transformer_ddp.py:**
- Location: `D:\tpami\ReMoMask\train_res_transformer_ddp.py`
- Triggers: `python train_res_transformer_ddp.py ...`
- Responsibilities: Training loop for R-Transformer refinement

**build_rag_database.py:**
- Location: `D:\tpami\ReMoMask\build_rag_database.py`
- Triggers: `python build_rag_database.py` (uses Hydra config)
- Responsibilities: Encode full dataset; save motion/text vectors & tokens to database

**eval_*.py (eval_mask.py, eval_res.py, eval_vq.py):**
- Location: `D:\tpami\ReMoMask\eval_*.py`
- Triggers: Evaluation metrics (FID, Matching Score, Diversity)
- Responsibilities: Benchmark generation quality

## Architectural Constraints

- **Threading:** Single-threaded PyTorch (models on GPU). Data loading uses workers, inference single-threaded.
- **Global state:** Pre-loaded models held in memory during inference (VQ, M-Trans×2, R-Trans×2, retriever, length estimator); no dynamic reload.
- **Circular imports:** None detected. Clear dependency hierarchy: data → encoders/retriever → transformers → VQ → rendering.
- **Fixed sequences:** All motion codes padded to `max_motion_length` (224 frames); padding handled via explicit mask tensors.
- **Retrieval immutable:** Database loaded once at startup; no online updates during inference.

## Anti-Patterns

### Duplicate Model Loading

**What happens:** Models loaded separately for 1D (aux) and 2D variants, each with independent checkpoints and state dicts.
**Why it's wrong:** Code duplication in `load_trans_aux()` and `load_trans_ts()` (lines 50-93 in demo.py); harder to swap backbones or share weights if needed.
**Do this instead:** Refactor into `load_mask_transformer(which_variant, ...)` factory in `models/transformer/__init__.py`.

### Re-Dict Creation Happens Once But Used Multiple Times

**What happens:** `retriever(captions)` called once at line 302 (demo.py); passed to both `mask_transformer_ts.generate()` and later accessed inside semantics layers.
**Why it's wrong:** No explicit connection between dict creation site and consumption sites; hard to trace data flow for Plan A when modifying retrieval strategy.
**Do this instead:** Document clear contract in `MocoTmrRetriever.forward()` return signature; assert schema in `semantics_modulated.py` entry.

### Hardcoded Dimension Constants

**What happens:** Latent dims (512), number of joints (6), token counts embedded in configs + model initialization.
**Why it's wrong:** Plan A may change retrieval feature dims; scattered config values make updates fragile.
**Do this instead:** Centralize retrieval feature dims in config; pass through constructor chain.

## Error Handling

**Strategy:** Assertions for critical invariants; let PyTorch raise on shape mismatches.

**Patterns:**
- Model checkpoint loading: Assert expected keys present (`demo.py:64-66`); log missing/unexpected keys
- Input validation: Shapes checked implicitly via tensor operations
- Database loading: File existence checked; NaN values skipped with warning (`build_rag_database.py:145-147`)

## Cross-Cutting Concerns

**Logging:** Print statements to stdout; checkpoint epochs/values logged at load time.
**Validation:** Motion length cutoffs (min 40 frames); padding applied automatically.
**Conditioning:** `cond_type` flags control text/retrieval presence; unconditional path used for classifier-free guidance.

---

*Architecture analysis: 2026-06-27*
