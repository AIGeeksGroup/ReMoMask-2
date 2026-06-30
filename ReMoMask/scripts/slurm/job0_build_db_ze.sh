#!/bin/bash
#SBATCH --job-name=build_db_ze
#SBATCH --nodes=1
#SBATCH --nodelist=persephone
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ywan0794/ReMoMask-2/slurm_logs/build_db_ze_%j.log

eval "$(/home/ywan0794/miniconda3/bin/conda shell.bash hook)"
conda activate remomask
cd /home/ywan0794/ReMoMask-2
mkdir -p slurm_logs

echo "=== Job: Build z_e Database ==="
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python build_rag_database_ze.py \
    --vq_name pretrain_vq \
    --output_dir database_ze \
    --device cuda:0 \
    2>&1

echo "=== Verifying output ==="
if [ -f database_ze/metadata.json ]; then
    echo "SUCCESS: database_ze built"
    cat database_ze/metadata.json
else
    echo "FAILED: database_ze/metadata.json not found"
    exit 1
fi
echo "End: $(date)"
