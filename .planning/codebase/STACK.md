# Technology Stack

**Analysis Date:** 2026-06-27

## Languages

**Primary:**
- Python 3.10 - Core implementation language for training and inference

**Secondary:**
- Blender Python (3.9.2) - Used for motion visualization and rendering

## Runtime

**Environment:**
- CUDA 11.8 - GPU compute backend
- PyTorch 2.1.0 (cu118) - Deep learning framework
- Python 3.10 runtime

**Package Manager:**
- pip - Python package management
- Conda - Environment management (environment: `remomask`)
- Lockfile: `requirements.txt` (`D:\tpami\ReMoMask\requirements.txt`)

## Frameworks

**Core:**
- PyTorch 2.1.0 - Neural network implementation
- Hydra 1.3.2 - Configuration management via YAML, used for Part_TMR training config (`D:\tpami\ReMoMask\Part_TMR\conf\config.yaml`)
- Transformers 4.52.4 - HuggingFace library for text encoders and tokenizers

**Training & Distributed:**
- Distributed Data Parallel (DDP) via `torch.distributed` - Multi-GPU training (8x A800 Tesla GPUs)
- NCCL - Distributed communication backend
- TensorBoard 2.19.0 - Training visualization and logging
- PyTorch Lightning utilities 0.14.3 - Training utilities

**Testing/Evaluation:**
- torchmetrics 1.7.4 - Metric computation
- scikit-learn 1.6.1 - ML utilities for evaluation

**Build/Dev:**
- OmegaConf 2.3.0 - Configuration object management (dependency of Hydra)

## Key Dependencies

**Critical:**
- torch==2.1.0+cu118 - GPU-accelerated tensor operations
- transformers==4.52.4 - Text encoder models (distilbert, ViT-B-32)
- clip @ git+https://github.com/openai/CLIP.git - OpenAI CLIP for frozen text encoding (specific commit: dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1)
- vector-quantize-pytorch==1.6.30 - VQ codebook implementation for RVQVAE
- einops==0.6.1 - Tensor rearrangement operations (critical for reshaping in dual-branch VQ-VAE)

**Infrastructure:**
- smplx==0.1.28 - SMPL-X body model for motion representation
- chumpy==0.70 - Sparse matrix operations for SMPL
- trimesh==4.6.11 - Mesh I/O and operations for motion fitting
- h5py==3.14.0 - HDF5 file format support for motion data
- scipy==1.10.1 - Scientific computing (optimization in motion fitting)

**Data Processing:**
- numpy==1.21.5 - Numerical computing
- pandas==2.0.3 - Tabular data handling
- scikit-learn==1.6.1 - Motion preprocessing utilities
- tqdm==4.67.1 - Progress bars

**Visualization & Output:**
- matplotlib==3.1.3 - Plotting and 2D visualization
- Pillow==9.2.2 - Image processing
- ffmpy==0.3.1 - FFmpeg wrapper for video encoding
- moviepy==1.0.3 (Blender env) - Video composition

**Configuration & Logging:**
- PyYAML==6.0 - YAML parsing for configs
- colorlog==6.9.0 - Colored logging output

**Text Processing:**
- tokenizers==0.21.2 - Fast tokenization for transformers
- ftfy==6.1.1 - Unicode normalization for text
- regex==2024.11.6 - Advanced regex for text parsing

**Utilities:**
- gdown==4.7.1 - Google Drive file downloads
- requests==2.32.4 - HTTP library for model/data downloads
- huggingface-hub==0.33.1 - HuggingFace model hub integration

## Configuration

**Environment:**
- TOKENIZERS_PARALLELISM environment variable - Controls tokenizer parallelization
- OMP_NUM_THREADS=1 - OpenMP threading for numeric libraries
- Device specification via command-line args (`--gpu_id`, `--device`)

**Build:**
- Config files: `D:\tpami\ReMoMask\configs\base.yaml`, `D:\tpami\ReMoMask\configs\config_vae_humanml3d.yaml`
- Part_TMR config: `D:\tpami\ReMoMask\Part_TMR\conf\config.yaml`
- Runtime options via Python argparse in `options/` directory:
  - `D:\tpami\ReMoMask\options\base_option.py`
  - `D:\tpami\ReMoMask\options\train_option.py`
  - `D:\tpami\ReMoMask\options\vq_option.py`
  - `D:\tpami\ReMoMask\options\eval_option.py`

## Platform Requirements

**Development:**
- A800 or H20 GPUs (8x for distributed training)
- CUDA Compute Capability 8.0+ (A100/A800 level)
- 256GB+ GPU memory for training with batch_size=128 (VAE) or batch_size=64 (MaskTransformer)
- 32+ CPU cores, 256GB+ RAM for data loading and preprocessing
- Linux/Unix environment (development tested on Linux)

**Production:**
- Single GPU minimum (CUDA 11.8 compatible)
- Inference batch_size=1-4 typical
- Model checkpoints stored locally in `./checkpoints/` and `./logs/`

**Dataset Storage:**
- HumanML3D: ~25GB (motion vectors + text annotations)
- KIT-ML: ~12GB
- SnapMoGen: additional dataset storage required
- Local database for RAG: `./database/` directory

---

*Stack analysis: 2026-06-27*
