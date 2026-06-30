#!/bin/bash
#SBATCH --job-name=proj_train
#SBATCH --nodes=1
#SBATCH --nodelist=persephone
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=03:00:00
#SBATCH --output=/home/ywan0794/ReMoMask-2/slurm_logs/proj_train_%j.log

eval "$(/home/ywan0794/miniconda3/bin/conda shell.bash hook)"
conda activate remomask
cd /home/ywan0794/ReMoMask-2
mkdir -p slurm_logs

echo "=== Job: Train Query Projector ==="
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Verify database_ze exists (should be built by job0)
if [ ! -f database_ze/metadata.json ]; then
    echo "ERROR: database_ze not found. Was job0 successful?"
    exit 1
fi

python train_query_projector.py \
    --database_ze_path database_ze \
    --database_bmm_path database \
    --output_dir logs/query_projector \
    --epochs 200 \
    --batch_size 128 \
    --device cuda:0 \
    2>&1

echo "=== Verifying output ==="
if [ -f logs/query_projector/best_projector.pt ]; then
    echo "SUCCESS: projector trained"
    ls -la logs/query_projector/best_projector.pt
else
    echo "FAILED: best_projector.pt not found"
    exit 1
fi
echo "End: $(date)"
