# Codebase Concerns

**Analysis Date:** 2026-06-27

## Tech Debt & Coupling Issues

### 1. Dimension Mismatch in Semantics Modulated Attention

**Issue:** The `SemanticsModulatedAttention` module assumes inconsistent dimensions across its fusion pipeline.

**Files:** `models/transformer/semantics_modulated.py` (lines 85-89)

**Problem:**
```python
# Line 85-89: Info fusion expects all inputs to have same dimension
fused_input = torch.cat([
    xf * text_cond,           # (B, 1, latent_dim) = 256/384
    R_m_pooled * retr_cond,   # (B, 1, 512) from database
    R_t_pooled * retr_cond    # (B, 1, 512) from database
], dim=-1)  # Fails: 256+512+512 ≠ 3*latent_dim
```

The `info_mlp` (line 36-39) expects input of shape `(B, 1, 3 * latent_dim)`, but:
- Text features (`xf`) are projected to `latent_dim` (256/384 in config)
- Retrieved motion/text features come from database at **512 dimensions** (not latent_dim)
- This causes concatenation dimension mismatch

**Impact:** Plan A will surface this bug. If re_dict features are not 512-dim, the forward pass will fail with shape mismatch during training.

**Affected:**
- Training: `transformer_trainer_ddp.py` line 220, 272 calls `self.retriever()` which returns 512-dim re_dict
- Inference: `demo.py` line 83, `eval_mask.py`, `eval_res.py` all pass re_dict to transformer

**Current Workaround:** The code appears to work because of undocumented behavior or the dimension mismatch is masked. This needs verification.

---

### 2. RAG Database Format & Shape Constraints

**Issue:** The database format hard-encodes dimension and shape assumptions that Plan A must respect.

**Files:** 
- `build_rag_database.py` (lines 224-263)
- `models/rag/t2m_retriever.py` (lines 221-227)

**Current Format:**
```python
# build_rag_database.py, line 240-242:
motion_embeddings = np.array(motion_embeddings)   # Shape: (N, 1, 512)
text_embeddings = np.array(text_embeddings)       # Shape: (N, 1, 512)

# Saved to:
# - database/encoded_motions.npy: (N, 1, 512)
# - database/encoded_texts.npy: (N, 1, 512)
```

**Loading in retriever:**
```python
# t2m_retriever.py, lines 221-222:
motion_features = np.load(f"{database_path}/encoded_motions.npy")[:, 0, :]  # (N, 512)
text_features = np.load(f"{database_path}/encoded_texts.npy")[:, 0, :]      # (N, 512)
```

**Constraints for Plan A:**
- Database is immutable until rebuild (23K samples × 2 files = 46K+ I/O operations)
- Any change to retrieval latent dimension requires:
  1. Rebuild database from scratch (112+ seconds per run mentioned in build_rag_database.py header)
  2. Re-encode all 23K motions through new VQ-VAE latent space
  3. Regenerate motion_tokens.npy with new token sequence format
- Shape `(N, 1, D)` is hardcoded in slicing logic; changing D will cascade to all downstream code

**Risk:** If Plan A changes latent_dim, you **must** rebuild the entire database. No partial updates possible.

---

### 3. VQ-VAE Token Sequence Format Dependency

**Issue:** Retrieved motion tokens are used elsewhere in the codebase, creating another dimension coupling point.

**Files:**
- `build_rag_database.py` (lines 268-289): Saves motion tokens
- `t2m_retriever.py` (implied usage in forward dict)

**Current Logic:**
```python
# build_rag_database.py, lines 276-286:
motion_token = np.load(motion_token_path)[0]  # Load from TOKENS/ directory
# ... time-based slicing with fps=20, unit_length=4
motion_token_dict[motion_name] = motion_token
np.save(f"{output_folder}/motion_tokens.npy", motion_token_dict)
```

**Constraints:**
- Token sequences are indexed by motion name (e.g., "001_0")
- Time slicing uses hardcoded `fps=20` and `unit_length=4` (line 49-50)
- Plan A must ensure token sequences remain valid if retrieving from latent space instead of Part_TMR space

**Risk:** If retrieval database is rebuilt with different latent encoding, the token sequences may become stale or inconsistent. This is especially risky during training with DDP when different workers might be out of sync.

---

### 4. SSTA Dimension Coupling in Semantics Modulated Attention

**Issue:** The `SemanticsModulatedAttention` has hardcoded assumptions about head dimension divisibility.

**Files:** `models/transformer/semantics_modulated.py` (line 24)

**Problem:**
```python
self.head_dim = latent_dim // num_heads  # Line 24
```

**Constraints:**
- `latent_dim` must be **exactly divisible by `num_heads`** (default 4)
- Config shows `latent_dim: [1, 256]`, which is divisible by 4 → head_dim = 64
- If Plan A changes latent_dim, ensure:
  - New dimension is divisible by 4 (or all num_heads values used)
  - This affects both motion tokens (line 106) and retrieval tokens

**Current Coupling:**
- Positional encoding in `PositionalEncoding2D` (line 52-78) has similar constraint: `d_model % 4 != 0` check
- This means latent_dim must be divisible by **both** 4 (heads) AND 2 (positional encoding pairs)
- Safest choice: use multiples of 4 or 8

**Risk:** Choosing odd dimensions or dimensions not divisible by 4 will cause silent failures in attention computation.

---

### 5. Database Rebuild Cascades Across Training Pipeline

**Issue:** The retriever module and training loop assume database is frozen during training.

**Files:**
- `transformer_trainer_ddp.py` (lines 220, 272): Calls `self.retriever()` dynamically
- `config.py`: Global config has hardcoded 512-dim assumptions
- All checkpoint files assume specific re_dict shape

**Current Flow:**
```python
# transformer_trainer_ddp.py, line 220:
re_dict = self.retriever(captions)  # Pulls from frozen database each iteration
```

**Constraints:**
- Retriever is set to `eval()` mode (line 33) and never re-initialized during training
- Database path is configured once at trainer init (line 24, via args)
- If you rebuild database mid-training, old checkpoints become incompatible with new re_dict format

**Risk:** Plan A must rebuild database **before** starting training. Rebuilding mid-training will cause:
- Shape mismatch errors in `semantics_modulated.py` forward pass
- DDP synchronization failures if different workers read different database versions
- Checkpoint loading failures (re_dict was saved with old dimensions)

---

### 6. Hardcoded Feature Dimensions in Global Config

**Issue:** The global configuration hard-encodes dimensions that Plan A will change.

**Files:** `config.py` (lines 2-6)

**Current:**
```python
global_bimoco_config = {
    "motion_embedding_dims":  512,
    "text_embedding_dims":  512,
    "projection_dims": 512
}
```

**Constraints:**
- These dimensions are used in `build_rag_database.py` (line 216-230) to encode motions and texts
- Changing to VQ-VAE latent space means these globals must be updated **in sync** with the new latent_dim
- If `motion_embedding_dims` ≠ `latent_dim`, the retriever will return mismatched shapes

**Risk:** Updating `global_bimoco_config` without updating transformer latent_dim (or vice versa) will cause silent shape mismatches.

---

### 7. Checkpoint Compatibility: Token Embedding & Codebook

**Issue:** Checkpoint loading assumes specific codebook dimension.

**Files:** `models/transformer/transformer_ts.py` (lines 224-235)

**Problem:**
```python
# Line 231:
self.token_emb.weight = nn.Parameter(
    torch.cat([codebook, torch.zeros(size=(2, d), device=codebook.device)], dim=0)
)
```

This assumes:
- Codebook shape is `(c, d)` where `d` is the VQ-VAE code dimension
- Two dummy tokens are added (mask token, pad token)
- Token embedding is frozen from codebook weights

**Current Assumptions:**
- Line 195: `self.token_emb = nn.Embedding(_num_tokens, self.code_dim)`
- `code_dim` is passed from VQ-VAE and is **immutable** once model is built
- All saved checkpoints encode the codebook dimension in their saved state

**Risk:** If Plan A changes the VQ-VAE latent space:
1. Old checkpoints with `code_dim=X` cannot load into model with `code_dim=Y`
2. Must retrain from scratch or carefully handle checkpoint conversion
3. Token embedding matrix size changes, breaking checkpoint compatibility

---

### 8. DDP Synchronization Risks During Database Access

**Issue:** Distributed training reads from a single shared database, creating potential race conditions.

**Files:** `transformer_trainer_ddp.py` (lines 23-34, 220, 272)

**Problem:**
```python
# Multiple DDP workers call retriever in parallel:
# Rank 0: reads database/encoded_motions.npy
# Rank 1: reads database/encoded_motions.npy
# Rank 2: reads database/encoded_motions.npy
re_dict = self.retriever(captions)  # Called by all workers
```

**Current Safeguards:**
- Retriever loads numpy arrays into memory once (lines 219-227 in t2m_retriever.py)
- Features are registered as buffers (lines 226-227), so they're part of model state
- But if database is being rebuilt or modified:
  - Worker 0 might read old database version
  - Worker 1 might read new database version
  - DDP synchronization will fail with shape mismatch

**Risk:** Plan A must ensure:
1. Database rebuild is **complete and verified** before resuming training
2. All workers see the same database version
3. No partial writes or in-place modifications to .npy files during training

---

### 9. FID Regression with RAG Integration

**Issue:** Plan A involves RAG (retrieval-augmented generation), which introduces known quality regression.

**Files:** All retrieval-related files

**Known Issue (from setup context):**
- V1 with RAG shows FID regression: 0.099 vs MoMask baseline 0.045 (2.2× worse)
- This suggests the retrieval mechanism is not yet properly aligned with the generative model

**Root Causes (Unknown):**
- Latent space geometry mismatch (z_e geometry not characterized)
- VQ-VAE codebook anisotropy (non-uniform token distribution)
- Data scarcity (23K samples may not be enough to learn robust retrieval)
- Attention fusion mechanism (semantics_modulated.py) may not be optimal

**Risk for Plan A:**
- Migrating to latent space retrieval will likely worsen FID further if the above root causes aren't addressed
- Must include ablation studies to identify which component degrades quality
- Consider implementing alternative fusion mechanisms (not just `info_mlp` concatenation)

---

### 10. Missing Projection Layer for Clip_Dim → Latent_Dim

**Issue:** Retrieved features are 512-dim (clip_dim) but transformer latent_dim may be 256/384, with no explicit projection.

**Files:** `models/transformer/transformer_ts.py` (lines 213-221, 214-215)

**Current Code:**
```python
cfg = {
    'latent_dim': latent_dim,           # 256 or 384
    'text_latent_dim': clip_dim,        # 512 (from re_dict)
    'num_heads': num_heads,
    'dropout': dropout,
}
self.semanticTransEncoder.append(SemanticsModulatedAttention(**cfg))
```

But `SemanticsModulatedAttention.__init__()` doesn't use `text_latent_dim` for any projection:
```python
# semantics_modulated.py lines 18-20:
def __init__(self, latent_dim, text_latent_dim, num_heads, dropout):
    # ... latent_dim used throughout, text_latent_dim only for LayerNorm (line 29)
    self.text_norm = nn.LayerNorm(text_latent_dim)
```

**The Bug:**
- `text_latent_dim` is documented but unused for re_dict features
- `re_motion` and `re_text` are 512-dim (clip_dim) but are concatenated as if they're `latent_dim`
- No projection layer converts 512 → latent_dim

**Risk:** Plan A must either:
1. Add explicit projection layers for re_motion and re_text in `SemanticsModulatedAttention`
2. Ensure VQ-VAE latent space is also 512-dim to match clip_dim
3. Or refactor semantics_modulated.py to handle dimension mismatch correctly

---

## Known Bugs & Workarounds

### Commented-Out Projection in Transformer

**Files:** `models/transformer/transformer_ts.py` (lines 347-349)

**Suspicious Code:**
```python
# text projection
# cond = self.cond_emb(cond).unsqueeze(0)  # (1, b, latent_dim)  # (1, 64, 512)
cond = cond.unsqueeze(0)  # COMMENTED OUT!
```

This suggests the text conditioning projection was disabled or removed. The comment says output is `(1, 64, 512)` but if `cond_emb` is a projection from clip_dim to latent_dim, this projection is now missing.

**Investigation Needed:** Check git history to see why this was commented out and if it affects Plan A.

---

## Missing Critical Features

### 1. Re_dict Dimension Projection Layer

**What's Missing:** 
- No projection from clip_dim (512) to latent_dim (256/384) for retrieved features
- Either the dimension mismatch is handled implicitly somewhere, or it's a latent bug

**Where to Add:**
- `SemanticsModulatedAttention.__init__()` should have:
  ```python
  if text_latent_dim != latent_dim:
      self.re_motion_proj = nn.Linear(text_latent_dim, latent_dim)
      self.re_text_proj = nn.Linear(text_latent_dim, latent_dim)
  else:
      self.re_motion_proj = nn.Identity()
      self.re_text_proj = nn.Identity()
  ```

**Risk if Missing:** Plan A will expose this bug when re_dict features don't match expected dimensions.

---

### 2. Database Rebuild Validation

**What's Missing:**
- No checksums or version tags on database files
- No validation that database and model configs are aligned
- No rollback mechanism if rebuild fails mid-process

**Where to Add:**
- Modify `build_rag_database.py` to save metadata: dimension, timestamp, model version
- Add validation in `t2m_retriever.py` to check database is compatible with model

**Risk if Missing:** Accidentally loading old database with new model config will cause silent shape mismatches.

---

### 3. DDP-Safe Database Locking

**What's Missing:**
- No file locking for database during training
- No check to ensure all workers see same database version
- No warning if database is being modified during training

**Where to Add:**
- `transformer_trainer_ddp.py` should verify database before each training run
- Optionally add `.lock` file to database directory during training

---

## Test Coverage Gaps

### 1. Retriever Shape Compatibility

**What's Not Tested:**
- Dimension mismatch between re_dict output and attention input
- Forward pass with mismatched dimensions

**Where to Add Tests:**
- `test/test_retriever.py`: Add test that verifies re_dict shape matches expected transformer input
- Include test with different latent_dim values

**Risk:** Shape bugs will only appear during training, wasting computational resources.

---

### 2. Database Integrity

**What's Not Tested:**
- Database rebuild produces valid output
- Database features are normalized correctly
- Motion IDs are unique and non-corrupted

**Where to Add Tests:**
- `test/test_rag_database.py`: Validate database after build

---

### 3. DDP Training with Retrieval

**What's Not Tested:**
- Multi-worker training with shared database access
- Race conditions or synchronization failures
- Checkpoint save/load with re_dict

---

## Performance Bottlenecks

### 1. Database Encoding is Slow

**Problem:** `build_rag_database.py` takes ~112 seconds to encode 23K motions (line 12).

**Cause:**
- Per-motion loop with individual model.encode_motion() calls (lines 215-230)
- No batching of encoding

**Improvement:**
- Batch multiple motions together to reduce overhead
- Parallelize encoding across multiple GPUs

**Impact on Plan A:** If you rebuild database multiple times during development, this becomes a bottleneck.

---

### 2. Retriever Uses Brute-Force Similarity Search

**Problem:** `t2m_retriever.py` computes cosine similarity to all database entries (lines 325-326).

**Files:** `models/rag/t2m_retriever.py` (line 325-326)

```python
score = self.cal_text_motion_sim(text_feature, self.motion_features)  # (N,)
indexes = torch.argsort(score, descending=True).cpu().numpy()  # (N,)
```

**Constraints:**
- O(N) similarity computation per query
- Suitable for 23K database, but will degrade with larger databases
- No spatial indexing (FAISS, VQ, etc.)

**Impact on Plan A:** Plan A doesn't change this, but it's a known limitation to be aware of.

---

## Scaling Limits

### 1. Database Size Cap (~23K)

**Current:** 23K motion samples in database

**Limit:** At current implementation:
- Database features loaded into GPU memory: 23K × 512 × 4 bytes ≈ 47 MB (negligible)
- But brute-force similarity search becomes slow > 100K entries

**Plan A Impact:** If you expand database later, must add indexing (FAISS) or approximate nearest neighbor search.

---

### 2. Data Scarcity for VQ-VAE Codebook

**Current:** 23K samples to train VQ-VAE codebook

**Limit:** VQ-VAE codebook learning may suffer from:
- Insufficient diversity to learn robust latent space
- Codebook collapse (only using subset of code symbols)
- Anisotropy in latent space (non-uniform token distribution)

**Plan A Impact:** Moving retrieval to latent space with only 23K samples is risky. Consider:
- Data augmentation during training
- Regularization to prevent codebook collapse
- Analyzing codebook utilization metrics

---

## Security & Data Integrity Concerns

### 1. Unencrypted Checkpoints & Databases

**Files:** 
- `database/encoded_motions.npy`
- `database/encoded_texts.npy`
- `database/motion_tokens.npy`
- Checkpoint files with state_dict

**Risk:** These files contain:
- Encoded motion representations (can be reverse-engineered)
- Text embeddings from CLIP (can leak private captions)
- VQ-VAE tokens (can be decoded to raw motion)

**Mitigation:** Ensure database and checkpoints are stored securely, especially if dataset is proprietary.

---

## Backward Compatibility Concerns

### 1. Checkpoint Format Changes

**Issue:** If Plan A changes latent_dim, old checkpoints become incompatible.

**Files:**
- Saved transformer states with `token_emb` weights
- Saved attention layer weights with `latent_dim` implicit in sizes

**Mitigation:**
- Add checkpoint version tag to detect incompatible formats
- Implement checkpoint migration script if changing latent_dim
- Document breaking changes in checkpoint format

---

### 2. Database Format Changes

**Issue:** If Plan A changes database shape from `(N, 1, D)` to something else, old databases break.

**Files:**
- All `.npy` files in `database/` directory

**Mitigation:**
- Add database version/metadata file
- Never in-place modify existing databases, always create versioned copies
- Document database format in build_rag_database.py

---

## Recommendations for Plan A

1. **Add Projection Layer:** Before implementing latent space retrieval, add explicit projection layers in `SemanticsModulatedAttention` to handle clip_dim → latent_dim conversion. This will expose the current dimension mismatch bug.

2. **Dimension Alignment:** Decide on final latent_dim:
   - If keeping 512: Update config and semantics_modulated.py to handle clip_dim == latent_dim
   - If using 256/384: Add projections and test thoroughly

3. **Database Rebuild Process:**
   - Implement metadata/versioning for database files
   - Add integrity checks before training
   - Consider implementing incremental rebuild (diff-based) for faster iteration

4. **Test Coverage:**
   - Add unit tests for re_dict shape compatibility
   - Add integration tests for DDP training with retrieval
   - Test checkpoint save/load cycle

5. **FID Regression Investigation:**
   - Implement codebook utilization metrics to detect collapse
   - Analyze latent space geometry (PCA, UMAP)
   - A/B test different fusion mechanisms in semantics_modulated.py

---

*Concerns audit: 2026-06-27*
