#!/bin/bash
#SBATCH --job-name=p4_e04_zq
#SBATCH --nodes=1
#SBATCH --nodelist=hades
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ywan0794/ReMoMask-2/slurm_logs/p4_e04_zq_%j.log

set -e
eval "$(/home/ywan0794/miniconda3/bin/conda shell.bash hook)"
conda activate remomask
cd /home/ywan0794/ReMoMask-2
mkdir -p slurm_logs

echo "=== Phase4 E04: build quantized-latent (z_q) retrieval database ==="
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python build_rag_database_ze.py \
    --vq_name pretrain_vq \
    --use_quantized \
    --output_dir database_zq \
    --device cuda:0 \
    2>&1

echo "=== verify ==="
python -c "import numpy as np, json; m=np.load('database_zq/encoded_motions.npy'); print('shape', m.shape); print(json.load(open('database_zq/metadata.json')))"
echo "End: $(date)"
