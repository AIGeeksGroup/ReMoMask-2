# External Integrations

**Analysis Date:** 2026-06-27

## APIs & External Services

**OpenAI CLIP:**
- Service: OpenAI CLIP ViT-B/32 text encoder
- What it's used for: Frozen text feature extraction for text-motion alignment and contrastive learning
- SDK/Client: `clip` package (git install from openai/CLIP)
- Model: `ViT-B-32.pt` (must be manually downloaded and placed in project root)
- Training: Frozen during all training stages (not fine-tuned)
- Reference: `D:\tpami\ReMoMask\models\rag\t2m_retriever.py` (imports `clip`)

**HuggingFace Models:**
- Service: HuggingFace Model Hub integration
- What it's used for: Pre-trained text encoders, tokenization
- SDK/Client: `transformers` package (4.52.4), `huggingface-hub` (0.33.1)
- Models accessed:
  - ViT-B-32.pt (CLIP, stored locally)
  - distilbert-base-uncased (alternative text encoder, loaded via transformers)
- References: `D:\tpami\ReMoMask\Part_TMR\models\encdoc.py` (TextEncoder uses AutoModel.from_pretrained)

## Data Storage

**Databases:**
- Type/Provider: Local HDF5/NPY file system
- Connection: File paths configured in training options
  - HumanML3D: `./dataset/HumanML3D/`
  - KIT-ML: `./dataset/KIT-ML/`
  - SnapMoGen: `/home/weihao/Documents/github/human/weichao_motion/dataset/MotionX/`
- Client: numpy (npy files), h5py (HDF5)
- Normalization: Mean/Std statistics stored in `Mean.npy` and `Std.npy` per dataset

**File Storage:**
- Local filesystem only
- Motion data: `.npy` files containing joint vectors
- Text annotations: `.txt` files (one text per line)
- Preprocessed splits: `train.txt`, `val.txt`, `test.txt` (file lists)
- References: `D:\tpami\ReMoMask\data\t2m_dataset.py`

**RAG Database:**
- Type: FAISS-like motion retrieval database
- Location: `./database/` directory (built during Stage 1 of training)
- Built from: Part_TMR model embeddings
- Used for: Retrieval-augmented generation during mask transformer training
- Build script: `D:\tpami\ReMoMask\build_rag_database.py`
- Retrieval function: `D:\tpami\ReMoMask\models\rag\t2m_retriever.py` (MocoTmrRetriever class)

**Caching:**
- None (no caching middleware detected)
- TensorBoard logs written to: `./logs/` and checkpoint directories

## Authentication & Identity

**Auth Provider:**
- Custom model checkpoint loading
- No API authentication required (all models stored locally or downloaded)
- Manual download of pretrained models from HuggingFace:
  - `Part_TMR/checkpoints/` - RAG model checkpoints
  - `logs/` - T2M generator checkpoints
  - `ViT-B-32.pt` - CLIP weights
- Checkpoint structure:
  ```
  checkpoints/
  ├── {dataset_name}/
  │   ├── {vq_name}/model/net_best_fid.tar
  │   ├── {mtrans_name}/model/net_best_fid.tar
  │   └── {rtrans_name}/model/...
  ```

## Monitoring & Observability

**Error Tracking:**
- None (no external service)
- Local exception handling with try/catch blocks
- Reference: `D:\tpami\ReMoMask\train_vq.py` uses simple exception printing

**Logs:**
- TensorBoard logging to `./logs/{dataset_name}/{exp_name}/` directory
- Tensorboard-data-server (0.7.2) for serving log data
- Log configuration in base config: `D:\tpami\ReMoMask\configs\base.yaml` (LOGGER section)
- Per-step metrics logged at configurable intervals (LOG_EVERY_STEPS)

## CI/CD & Deployment

**Hosting:**
- Local machine or cluster deployment
- No cloud platform integration detected
- Execution via command-line scripts with Hydra/argparse

**CI Pipeline:**
- None (no CI service integration)
- Training initiated via bash scripts:
  - `run_rvq.sh` - VQ-VAE training
  - `run_mtrans.sh` - Mask transformer training with DDP
  - `run_rtrans.sh` - Residual transformer training

**Model Serving:**
- Inference via Python scripts:
  - `demo.py` - Interactive generation demo
  - `eval_mask.py` - Evaluation pipeline for mask transformer
  - `eval_res.py` - Evaluation pipeline for residual transformer
  - Video rendering: `render.py` (Blender batch mode)
  - SMPL mesh fitting: `fit.py` (using pyopengl/smplx)

## Environment Configuration

**Required env vars:**
- `TOKENIZERS_PARALLELISM=true` - Enable tokenizer parallelization (set in `train_mask_transformer_ddp.py`)
- `OMP_NUM_THREADS=1` - Limit OpenMP threads (set in `train_vq.py`)
- `WORLD_SIZE`, `RANK`, `LOCAL_RANK` - DDP distributed training variables (set by SLURM/torchrun)
- `CUDA_VISIBLE_DEVICES` - GPU selection (implicit via --gpu_id or --device args)

**Secrets location:**
- No secrets file detected (no .env analysis needed per forbidden_files rules)
- Model paths configured via command-line args and YAML configs

## Webhooks & Callbacks

**Incoming:**
- None (no webhook endpoints)

**Outgoing:**
- None (no external API callbacks)

## Pretrained Models

**Part_TMR (Text-Motion Retriever):**
- Purpose: Retrieval-augmented motion database builder and retriever
- Implementation: `D:\tpami\ReMoMask\Part_TMR\models\builder_bimoco.py` (MoCoTMR class)
- Checkpoint: `./Part_TMR/checkpoints/{exp_name}/{dataset_name}/`
- Architecture: MoCo contrastive learning with bi-directional momentum
- Configuration: `D:\tpami\ReMoMask\Part_TMR\conf\config.yaml`
  - embed_dim: 512
  - queue_size: 65536
  - momentum: 0.999
  - temperature: 0.07
  - HBM loss enabled
- Training: Stage 1 of pipeline (`Part_TMR/scripts/train.py`)
- Used by: RAG database builder and MaskTransformer during Stage 2

**RVQVAE (Residual Vector Quantized VAE):**
- Purpose: Motion tokenization with dual-branch architecture
- Implementation: `D:\tpami\ReMoMask\models\vq\model.py` (RVQVAE class)
- Checkpoint: `checkpoints/{dataset_name}/{vq_name}/model/net_best_fid.tar`
- Architecture:
  - Branch 1D: Global motion encoding (Encoder1d → ResidualVQ → Decoder1d)
  - Branch 2D: Joint-level detail encoding (Encoder2d → ResidualVQ → Decoder2d)
  - Quantization: 6 layers, 512 codebook size, 512 code dimension per branch
  - Input: 263D (HumanML3D) or 251D (KIT-ML) motion vectors
- Configuration: `D:\tpami\ReMoMask\configs\config_vae_humanml3d.yaml`
- Training: Stage 2a (`train_vq.py` via `run_rvq.sh`)
- Used by: MaskTransformer and ResidualTransformer as motion encoder

**MaskTransformer:**
- Purpose: Masked motion generation conditioned on text
- Implementation: `D:\tpami\ReMoMask\models\transformer\transformer_aux.py` (MaskTransformer class)
- Checkpoint: `checkpoints/{dataset_name}/{mtrans_name}/model/net_best_fid.tar`
- Architecture: Transformer with:
  - Masking ratio for corrupted input handling
  - Text conditioning via CLIP embeddings
  - RAG retrieval augmentation
  - Latent dimension: 512
  - 8 attention heads, 9 layers (default)
- Training: Stage 2b (`train_mask_transformer_ddp.py` with DDP on 8 GPUs)
- Used by: Inference pipeline and downstream residual transformer

**ResidualTransformer:**
- Purpose: Residual motion refinement after mask transformer
- Implementation: `D:\tpami\ReMoMask\models\transformer\transformer_ts.py` (MaskTransformer2D class)
- Checkpoint: `checkpoints/{dataset_name}/{rtrans_name}/model/net_best_fid.tar`
- Configuration: Similar to MaskTransformer
- Training: Stage 2c (`train_res_transformer_ddp.py` with DDP)
- Used by: Final stage of inference pipeline

## Data Pipeline

**HumanML3D Dataset:**
- Format: NPY files (motion vectors) + TXT files (text annotations)
- Motion representation:
  - 22 joints (HumanML3D and MotionX)
  - 21 joints (KIT-ML)
  - 12 degrees of freedom per joint (4 + 63 + 66 + 66 + 4 dimensions)
  - Features: root rotation velocity, root linear velocity, joint RIC positions, 6D rotations, local velocities, foot contact
- Location: `./dataset/HumanML3D/new_joint_vecs/` and `./dataset/HumanML3D/texts/`
- Split files: `train.txt`, `val.txt`, `test.txt`
- Normalization: Mean/Std statistics in `Mean.npy`, `Std.npy`
- Loading: `D:\tpami\ReMoMask\data\t2m_dataset.py` (MotionDataset, Text2MotionDataset classes)
- Preprocessing: Z-normalization, window sampling, length filtering

**KIT-ML Dataset:**
- Format: Similar to HumanML3D
- Motion representation: 21 joints, same feature structure
- Frame rate: 12.5 fps (vs 20 fps for HumanML3D)
- Location: `./dataset/KIT-ML/`

**SnapMoGen Dataset:**
- Format: NPY + TXT
- Additional dataset for evaluation
- Location: `/home/weihao/Documents/github/human/weichao_motion/dataset/MotionX/`

**Data Loaders:**
- Training: `D:\tpami\ReMoMask\motion_loaders\dataset_motion_loader.py` (get_dataset_motion_loader function)
- Batch collation: Custom `collate_fn` in `D:\tpami\ReMoMask\data\t2m_dataset.py` (sorts by sequence length)
- Evaluation: Separate evaluation dataset class `Text2MotionDatasetEval`

## Evaluation Pipeline

**Metrics Framework:**
- torchmetrics 1.7.4 for metric computation
- Reference: `D:\tpami\ReMoMask\models\t2m_eval_modules.py` (evaluation metric implementations)

**VQ-VAE Evaluation:**
- Script: `D:\tpami\ReMoMask\eval_vq.py`
- Metrics: FID (Fréchet Inception Distance) reconstruction quality
- Checkpoint selection: `net_best_fid.tar`

**Mask Transformer Evaluation:**
- Script: `D:\tpami\ReMoMask\eval_mask.py`
- Metrics: Text-motion similarity, motion diversity, FID, multimodal metrics
- Guidance scale: Configurable (default 4)
- Denoising steps: Configurable (default 10)
- Sampling modes: Single sample (NUM_SAMPLES=1) or multimodal (MM_NUM_SAMPLES=100)

**Residual Transformer Evaluation:**
- Script: `D:\tpami\ReMoMask\eval_res.py`
- Cascaded: Runs mask transformer then residual transformer
- Final motion output in NPY format
- Visualization: Integrated with video rendering pipeline

**Part_TMR (RAG) Evaluation:**
- Script: `D:\tpami\ReMoMask\Part_TMR/scripts/test.py`
- Metrics: Retrieval accuracy, text-motion alignment quality
- Uses HuggingFace datasets and model evaluation

**Visualization Pipeline:**
- Motion fitting: `D:\tpami\ReMoMask\fit.py` (pyopengl, smplx)
- Video rendering: `D:\tpami\ReMoMask\render.py` (Blender batch mode)
- 3D plotting: `D:\tpami\ReMoMask\utils\plot_script.py`

## Third-Party Code Dependencies

**Imported Frameworks:**
- MoMask (original motion generation framework)
- MoGenTS (time series generation techniques)
- ReMoDiffuse (diffusion-based motion generation)
- MDM (Motion Diffusion Model)
- TMR (Text-Motion Retriever)
- ReMoGPT (GPT-based motion generation)
- References acknowledged in `README.md`

**SMPL Body Model:**
- Package: `smplx==0.1.28`
- Purpose: Human pose representation and skinning
- Used in: `D:\tpami\ReMoMask\fit.py`, visualization pipeline
- Model format: PKL files (SMPL-X model parameters)
- Reference: https://github.com/vchoutas/smplx

**Motion Processing Utilities:**
- Location: `D:\tpami\ReMoMask\utils/`
- Key modules:
  - `motion_process.py` - Recover joint positions from relative coordinates
  - `paramUtil.py` - Kinematic chains for HumanML3D/KIT
  - `plot_script.py` - 3D motion visualization
  - `word_vectorizer.py` - Text feature extraction

**CLIP Text Encoding:**
- Token max length: Configurable, typically 20 tokens per text description
- Frozen embeddings: 512-dimensional (ViT-B/32 output)
- Preprocessing: ftfy text cleanup, tokenization via transformers

---

*Integration audit: 2026-06-27*
