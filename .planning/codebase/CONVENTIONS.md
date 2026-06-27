# Coding Conventions

**Analysis Date:** 2026-06-27

## Naming Patterns

**Files:**
- `snake_case.py` for all Python files
- Example: `transformer_ts.py`, `t2m_dataset.py`, `train_mask_transformer_ddp.py`

**Classes:**
- `PascalCase` for all classes
- Example: `MaskTransformer2D`, `RVQVAE`, `InputProcess`, `OutputProcess_Bert`
- Abbreviations in caps (RVQ, VAE, DDP, MLP)

**Functions & Methods:**
- `snake_case` for methods and standalone functions
- Example: `load_and_freeze_clip()`, `cal_loss()`, `forward()`

**Variables:**
- Mostly `snake_case`
- Some properties use `camelCase`: `poseEmbedding`, `poseFinal`
- Acronym variables: `x1d`, `x2d`, `m_lens`, `re_dict`
- Private/internal variables: prefix with underscore (e.g., `_loss1`, `_pred_ids`, `_acc1`)
- Positional/shape abbreviations: `bs` (batch size), `seq_len`, `d_model`, `nhead`

**Module Names:**
- Append module type to distinguish variants: `transformer_aux.py` (auxiliary), `transformer_ts.py` (temporal-spatial)

## Code Style

**Formatting:**
- No explicit formatter config (no Black, autopep8 config found)
- 4-space indentation (standard Python)
- Line lengths vary (80-120+ characters observed)
- Mixed formatting across codebase

**Linting:**
- No linting config present (.pylintrc, .flake8, pyproject.toml not found)
- Code follows basic PEP 8 conventions informally

## Import Organization

**Order:**
1. Standard library (`os`, `sys`, `math`, `time`)
2. Third-party packages (`torch`, `numpy`, `clip`, `einops`)
3. Local imports (relative absolute paths from project root)

**Common Patterns:**
```python
# Path utilities
from os.path import join as pjoin

# Relative imports from project root (NOT relative dots)
from models.transformer.tools import *
from options.train_option import TrainT2MOptions
from utils.motion_process import recover_from_ric

# Wildcard imports for utility modules
from models.transformer.tools import *
from utils.utils import *
```

**Config Imports:**
- Hydra compose pattern for Part_TMR configs:
```python
from hydra import initialize, compose
with initialize(config_path="Part_TMR/conf", version_base=None):
    retriever_cfg = compose(config_name="config")
```

**Path Aliases:**
- `pjoin` is universal for `os.path.join()`
- Enables readable path operations: `pjoin(self.opt.checkpoints_dir, self.opt.dataset_name, model_name)`

## Configuration Patterns

**Global Configs:**
- Located in `config.py` as simple Python dictionaries
- Example (`config.py`):
```python
global_bimoco_config = {
    "motion_embedding_dims": 512,
    "text_embedding_dims": 512,
    "projection_dims": 512
}

global_momentum_config = {
    'embed_dim': 512,
    'queue_size': 65536,
    'momentum': 0.99,
}
```

**Options/Arguments:**
- Class-based options pattern in `options/` directory
- Inheritance: `BaseOptions` → `TrainT2MOptions`, `EvalT2MOptions`, `TrainLenEstOptions`
- Each option class extends argparse via `.add_argument()`
- Options parsed once via `.parse()`, returns `opt` namespace object
- Passed as `opt` parameter throughout training/eval pipeline

## Model Initialization

**Pattern:**
- Large parameter lists (8+ parameters common)
- Named parameters: `latent_dim`, `ff_size`, `num_layers`, `num_heads`, `dropout`, `clip_dim`, `cond_drop_prob`
- Extracted from `opt` object: `MaskTransformer2D(code_dim=..., latent_dim=opt.latent_dim, ...)`

**Example from `models/transformer/transformer_ts.py`:**
```python
class MaskTransformer2D(nn.Module):
    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1,
                 clip_version=None, opt=None, **kargs):
        super(MaskTransformer2D, self).__init__()
```

**Module Sections:**
- Code organized into logical blocks with comment headers:
  - `# Preparing Networks` - component initialization
  - `# Preparing frozen weights` - external model loading (CLIP)
  - `# SemanticsAtten` / `# 初始化 semanticAtten` - specialized module setup

## Layer Initialization

**Weight Initialization:**
- Custom `__init_weights()` method applied via `.apply()`
- Pattern in `transformer_ts.py`:
```python
def __init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
```

## Comments & Documentation

**Style:**
- Block comments use `'''...'''` for multi-line explanations
- Inline comments with `#` (variable spacing)
- Mixed Chinese/English comments throughout codebase
- Minimal docstrings (not systematic)

**Comment Placement:**
- Comments often appear BEFORE code blocks to explain what follows
- Commented-out code preserved (not deleted)

**Example from `semantics_modulated.py`:**
```python
# Normalization
self.norm = nn.LayerNorm(latent_dim)
self.text_norm = nn.LayerNorm(text_latent_dim)

# Query projection (只作用于motion tokens z)
self.query = nn.Linear(latent_dim, latent_dim)

# Info fusion MLP: [t; R_m; R_t] -> info
self.info_mlp = nn.Sequential(...)
```

## Utility Functions

**Location:** `utils/tools.py`, `models/transformer/tools.py`

**Common Patterns:**
- Functional utilities (no class wrapping): `exists()`, `default()`, `l2norm()`
- Decorators for common patterns: `@eval_decorator` for train/eval mode toggling
- Loss & metric calculation: `cal_loss()`, `cal_performance()` with optional label smoothing
- Sampling helpers: `gumbel_sample()`, `top_k()` for discrete sampling
- Scheduling: `cosine_schedule()`, `scale_cosine_schedule()`

**Example from `models/transformer/tools.py`:**
```python
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner
```

## Known Inconsistencies & Paper-Code Gaps

### Dimension Mismatch (Critical for Plan A)
**Files:** `options/base_option.py` (line 19), `models/transformer/transformer_ts.py` (line 125)

- **Default latent_dim:** 384 in argparse
- **SSTA constraint:** Requires latent_dim == clip_dim == 512
- **Current behavior:** Code runs with latent_dim=384, but SSTA documentation indicates coupling to 512
- **Implication:** Plan A latent alignment retrieval assumes latent_dim=512; misaligned if using defaults

### SSTA Layer Count
**File:** `options/base_option.py` (line 21)

- **Default n_layers:** 8
- **Paper:** Claims 6 SSTA layers
- **Code:** Creates `num_layers` instances of `SemanticsModulatedAttention` (line 220-221 in transformer_ts.py)

### Part Encoder Architecture
**File:** `models/transformer/transformer_ts.py` (lines 219-221)

- **Paper description:** "Shared part encoder"
- **Code reality:** `semanticTransEncoder = nn.ModuleList()` creates 6 separate instances
- **Behavior:** Each layer gets its own SemanticsModulatedAttention, not weight-sharing

## Type Hints

**Status:** Inconsistently used

- Some methods include type hints: `def forward(self, batch_data, re_dict=None):`
- Many lack them: most utility functions, legacy code
- No systematic typing across codebase
- Type annotation not enforced by project config

## Error Handling

**Strategy:** Minimal

- Assertions for critical conditions: `assert self.training, 'Only necessary in training mode'`
- Conditional logic for optional features: `if 'attnt' in opt and opt.attnt:`
- Try-except blocks in visualization only (e.g., `plot_3d_motion()` wraps in try-except)
- No custom exception classes observed

**Example from `transformer_ts.py`:**
```python
if self.cond_mode == 'text':
    assert 'num_actions' in kargs  # Explicit validation
```

## Tensor Operations & Shapes

**Convention:** Shapes documented in comments

Example from `transformer_trainer_ddp.py`:
```python
'''
conds:  prompt embedding # (b, 512)
motion: (b, t, raw_motion_dim)   # (64, 196, 263)
joints: (b, t, j, jd)  # (64, 196, 22, 12)
m_lens: (b,)  # 64
'''
```

**Pattern:** Always include batch dimension first, temporal second, spatial third

## Trainer Pattern

**File:** `models/transformer/transformer_trainer_ddp.py`

- Trainer class wraps model(s) and encapsulates training loop
- Methods: `forward()`, `update()`, `save()`, `validation()`
- Integrates distributed training (DDP) handling
- Manages optimizers and learning rate scheduling

---

*Convention analysis: 2026-06-27*
