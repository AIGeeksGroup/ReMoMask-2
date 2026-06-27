# Codebase Structure

**Analysis Date:** 2026-06-27

## Directory Layout

```
ReMoMask/
├── Part_TMR/                       # Text-Motion Retrieval: text encoding & database retrieval
│   ├── conf/                       # Hydra config files
│   ├── datasets/                   # Motion dataset loading utilities
│   │   ├── dataset.py              # HumanML3D/KIT dataset class
│   │   ├── utils.py                # Motion part decomposition (whole2parts)
│   │   └── __init__.py
│   ├── models/
│   │   ├── builder_bimoco.py       # MoCoTMR model (text/motion encoder with momentum contrast)
│   │   ├── encdoc.py               # Encoder/decoder components
│   │   ├── hbm_loss.py             # Head-based motion loss
│   │   ├── losses.py               # Contrastive losses
│   │   └── positional_encoding.py  # Position encodings for TMR
│   └── scripts/
│       ├── train.py                # Training script for TMR
│       ├── test.py                 # Test/eval script
│       └── utils.py                # Training utilities
│
├── models/                         # Core generative models
│   ├── rag/                        # Retrieval-Augmented Generation
│   │   └── t2m_retriever.py        # MocoTmrRetriever: database load + kNN search
│   ├── transformer/                # Main generative transformers
│   │   ├── transformer_aux.py      # MaskTransformer (1D) + ResidualTransformer (1D)
│   │   ├── transformer_ts.py       # MaskTransformer2D (2D) + ResidualTransformer2D (2D)
│   │   ├── semantics_modulated.py  # SemanticsModulatedAttention: fusion of retrieval features
│   │   ├── tools.py                # Helper functions (schedules, sampling, etc)
│   │   ├── transformer_trainer_ddp.py  # DDP training utilities (unused in inference)
│   │   └── __init__.py
│   ├── vq/                         # Vector Quantization VAE
│   │   ├── model.py                # RVQVAE: encoder/decoder with residual quantizers
│   │   ├── residual_vq.py          # Hierarchical VQ structure
│   │   ├── quantizer.py            # Single quantizer implementation
│   │   ├── encdec.py               # Encoder/decoder networks
│   │   ├── resnet.py               # ResNet backbone for VQ encoder
│   │   └── vq_trainer.py           # Training loop for VQ model
│   ├── t2m_eval_modules.py         # Evaluation metric implementations (FID, Matching, etc)
│   ├── t2m_eval_wrapper.py         # Wrapper for batch evaluation
│   └── __init__.py
│
├── data/                           # Dataset handling
│   ├── t2m_dataset.py              # Text-Motion pair dataset class
│   └── __init__.py
│
├── motion_loaders/                 # Motion file I/O
│   ├── dataset_motion_loader.py    # Load .npy motion files + metadata
│   └── __init__.py
│
├── options/                        # Configuration parsing
│   ├── base_option.py              # Base command-line options
│   ├── train_option.py             # Training-specific options
│   ├── eval_option.py              # Evaluation options (used by demo.py)
│   ├── vq_option.py                # VQ model options
│   └── __init__.py
│
├── configs/                        # Static YAML config files
│   ├── base.yaml                   # Base dataset/model config
│   ├── assets.yaml                 # Asset paths (checkpoints, data)
│   ├── config_*_humanml3d.yaml     # Dataset-specific configs
│   ├── modules/                    # Module-level configs (denoiser, VAE, text encoder)
│   └── render_*.yaml               # Rendering options
│
├── utils/                          # Utilities
│   ├── motion_process.py           # Motion <-> RIC format conversion
│   ├── plot_script.py              # 3D visualization
│   ├── paramUtil.py                # Kinematic chain, joint mapping
│   ├── fixseed.py                  # Seed setting
│   ├── get_opt.py                  # Load saved options from files
│   └── ...
│
├── helper/                         # Misc helpers
│   ├── read_npy.py                 # NPY file utilities
│   ├── rtrans_loader.py            # Load residual transformer checkpoints
│   └── ...
│
├── visualization/                  # Rendering pipeline
│   ├── joints2bvh.py               # Convert joint data to BVH animation format
│   └── ...
│
├── scripts/                        # Analysis/conversion scripts
│   ├── fbx_output.py               # FBX export helper
│   └── ...
│
├── common/                         # Shared utilities
│   ├── quaternion.py               # Quaternion math for skeleton
│   ├── skeleton.py                 # Skeleton structure definitions
│   └── __init__.py
│
├── assets/                         # Static assets (images, templates)
│
├── reference/                      # Reference materials
│
├── course/                         # Learning materials / documentation
│
├── logs/                           # Training checkpoints (generated at runtime)
│
├── checkpoints/                    # Inference checkpoints (symlink or copied models)
│
├── database/                       # RAG database (generated by build_rag_database.py)
│
├── outputs/                        # Generated motions (BVH, MP4, NPY) (generated at runtime)
│
# Entry point scripts
├── demo.py                         # Main inference script: text → motion generation
├── eval_mask.py                    # Evaluate M-Transformer generation quality
├── eval_res.py                     # Evaluate R-Transformer refinement
├── eval_vq.py                      # Evaluate VQ reconstruction
├── fit.py                          # Fit model on data (unused in Plan A)
│
# Training entry points
├── train_mask_transformer_ddp.py   # Train MaskTransformer (1D + 2D) with DDP
├── train_res_transformer_ddp.py    # Train ResidualTransformer (1D + 2D) with DDP
├── train_vq.py                     # Train RVQVAE
│
# Utilities
├── render.py                       # Render BVH to MP4 video
├── config.py                       # Global config loader (retriever_cfg, etc)
│
# Database builder
├── build_rag_database.py           # Build retrieval database from dataset
│
# Scripts
├── run_mtrans.sh                   # Shell script: train M-Transformer
├── run_rtrans.sh                   # Shell script: train R-Transformer
├── run_rvq.sh                      # Shell script: train RVQ
│
├── requirements.txt                # Python dependencies
├── README.md                        # Project overview
├── .gitignore                      # Git ignore rules
└── .git/                           # Git repository
```

## Directory Purposes

**Part_TMR/:**
- Purpose: Pre-trained text-motion encoder for encoding text and retrieving similar motions
- Contains: Text encoder (BERT-like), motion encoder (transformer), contrastive learning losses
- Key files: `models/builder_bimoco.py` (MoCoTMR), `datasets/utils.py` (motion part split)
- Generated: Pre-trained checkpoint at `checkpoints/exp1/HumanML3D/{best,last}_model.pt`

**models/rag/:**
- Purpose: Retrieval-augmented generation pipeline
- Contains: Database loader, kNN search, feature indexing
- Key files: `t2m_retriever.py` (MocoTmrRetriever)

**models/transformer/:**
- Purpose: Main generative models for iterative token prediction
- Contains: Mask transformer (2 variants: 1D, 2D), residual transformers (2 variants), semantics modulation layer
- Key files: `transformer_aux.py` (1D), `transformer_ts.py` (2D), `semantics_modulated.py` (attention fusion)

**models/vq/:**
- Purpose: Discrete motion representation (codebook) and decoding
- Contains: RVQVAE, residual quantizers, codebook embeddings
- Key files: `model.py` (RVQVAE), `residual_vq.py` (quantizer chain)

**configs/:**
- Purpose: Static YAML configuration defaults
- Contains: Dataset paths, model hyperparameters, training settings
- Key files: `base.yaml` (datasets), `modules/text_encoder.yaml` (TMR config)

**options/:**
- Purpose: Command-line argument parsing + option storage
- Contains: Base parser, dataset-specific parsers (train, eval, vq)
- Key files: `eval_option.py` (used by demo.py), `base_option.py` (common options)

**utils/:**
- Purpose: Shared helper functions
- Contains: Motion format conversion, visualization, kinematic chains, seed management
- Key files: `motion_process.py` (RIC ↔ joint conversion), `plot_script.py` (3D visualization)

## Key File Locations

**Entry Points:**
- `demo.py` - Primary inference entry point (text → motion)
- `train_mask_transformer_ddp.py` - Train M-Transformer
- `train_res_transformer_ddp.py` - Train R-Transformer
- `build_rag_database.py` - Build retrieval database

**Configuration:**
- `config.py` - Loads global retriever config via Hydra
- `options/eval_option.py` - Command-line options for demo.py
- `Part_TMR/conf/config.yaml` - Hydra config for retriever & database builder

**Core Logic:**
- `models/rag/t2m_retriever.py` - Retrieval logic (MocoTmrRetriever)
- `models/transformer/transformer_ts.py` - 2D mask transformer + generation loop
- `models/transformer/semantics_modulated.py` - Retrieval-aware attention
- `models/vq/model.py` - RVQVAE decoding (forward_decoder)

**Testing/Evaluation:**
- `eval_mask.py` - Evaluate generated motions from M-Transformer
- `eval_res.py` - Evaluate refined motions from R-Transformer
- `eval_vq.py` - Evaluate VQ reconstruction
- `models/t2m_eval_modules.py` - FID, Matching Score, Diversity metrics

## Naming Conventions

**Files:**
- `{component}_transformer_{variant}.py` - Transformer models (aux/ts for 1D/2D)
- `{component}_trainer{,_ddp}.py` - Training loops
- `eval_{component}.py` - Evaluation for specific component
- `build_{system}.py` - Data/resource builders
- `{adjective}_modulated.py` - Attention modules with additional conditioning

**Directories:**
- `Part_TMR/` - Acronym for "Text-Motion Retriever" (also known as MoCoTMR)
- `models/{rag,transformer,vq}/` - Component grouping by function
- `configs/` - Static YAML files
- `options/` - Option/argument classes

**Classes:**
- `MaskTransformer{,2D}` - Iterative decoding with masking
- `ResidualTransformer{,2D}` - Refinement over quantizer levels
- `RVQVAE` - Residual VQ-VAE
- `MocoTmrRetriever` - Retrieval interface
- `SemanticsModulatedAttention` - Attention with retrieval fusion

**Functions:**
- `forward()` - Main forward pass
- `generate()` - Autoregressive/iterative generation
- `forward_decoder()` - Decode tokens to motion
- `encode_{motion,text}()` - Feature encoding
- `forward_with_cond_scale()` - Classifier-free guidance

## Where to Add New Code

**New Feature (Retrieval-Augmented):**
- Primary code: `models/transformer/semantics_modulated.py` (modify SemanticsModulatedAttention.forward)
- Secondary: `models/rag/t2m_retriever.py` (modify re_dict structure if needed)
- Tests: `eval_mask.py` (test generation quality with new retrieval strategy)
- Config: `Part_TMR/conf/config.yaml` (add retrieval hyperparameters)

**New Component/Module:**
- Implementation: `models/{rag,transformer,vq}/new_module.py`
- Registration: `models/__init__.py`
- Tests: Create `test_new_module.py` in root or `tests/` (if created)

**Utilities:**
- Shared helpers: `utils/new_helper.py`
- Motion format: `utils/motion_process.py`
- Visualization: `visualization/new_visual.py`

**Training Scripts:**
- New trainer: `train_new_component_ddp.py` (follow DDP pattern from `train_mask_transformer_ddp.py`)
- Options: Add to `options/train_option.py`

**Database/Offline Processing:**
- Database builder: Extend `build_rag_database.py` (add new feature types to `encode_motion()`)
- Output: Save to `database/` directory (loaded by MocoTmrRetriever)

## Special Directories

**Part_TMR/conf/:**
- Purpose: Hydra configuration for retriever model training and database building
- Generated: No, static configs in repo
- Committed: Yes
- Key file: `config.yaml` (main Hydra config)

**checkpoints/:**
- Purpose: Pre-trained model checkpoints loaded at inference
- Generated: Yes, by training scripts or downloaded
- Committed: No, symlink to external storage
- Structure: `checkpoints/{dataset}/{model_name}/model/{best,last}_model.tar` or `.pt`

**database/:**
- Purpose: Pre-computed retrieval vectors and metadata
- Generated: Yes, by `build_rag_database.py`
- Committed: No (too large)
- Files: `motion_ids.npy`, `all_captions.npy`, `encoded_motions.npy`, `encoded_texts.npy`, `motion_tokens.npy`

**logs/:**
- Purpose: Training logs and checkpoints during training
- Generated: Yes, by training scripts with DDP
- Committed: No
- Structure: `logs/{dataset}/{experiment_name}/model/net_best_fid.tar`

**outputs/:**
- Purpose: Generated motion files (BVH, MP4, NPY)
- Generated: Yes, by demo.py and evaluation scripts
- Committed: No
- Structure: `outputs/{ext}/animations/`, `outputs/{ext}/joints/`

**assets/:**
- Purpose: Static project assets
- Generated: No, static
- Committed: Yes

---

*Structure analysis: 2026-06-27*
