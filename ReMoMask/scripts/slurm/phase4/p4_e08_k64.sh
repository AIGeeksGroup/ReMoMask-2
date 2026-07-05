#!/bin/bash
#SBATCH --job-name=p4_e08_k64
#SBATCH --nodes=1
#SBATCH --nodelist=hades
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ywan0794/ReMoMask-2/slurm_logs/p4_e08_k64_%j.log

set -e
eval "$(/home/ywan0794/miniconda3/bin/conda shell.bash hook)"
conda activate remomask
cd /home/ywan0794/ReMoMask-2
mkdir -p slurm_logs

echo "=== Phase4 E08 (kappa sensitivity): projector retrain, teacher_topk=64 ==="
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train_query_projector.py \
    --database_ze_path database_ze \
    --database_bmm_path database \
    --output_dir logs/query_projector_k64 \
    --epochs 200 --batch_size 128 \
    --lr 1e-4 --temperature 0.07 \
    --teacher_topk 64 --eval_interval 10 \
    --device cuda:0 \
    2>&1

echo "=== Final metrics ==="
python -c "import json; d=json.load(open('logs/query_projector_k64/training_log.json')); print(d[-1] if isinstance(d, list) else d)" 2>/dev/null || true
echo "End: $(date)"
