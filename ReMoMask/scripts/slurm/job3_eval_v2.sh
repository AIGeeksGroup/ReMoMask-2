#!/bin/bash
#SBATCH --job-name=eval_v2
#SBATCH --nodes=1
#SBATCH --nodelist=persephone
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=06:00:00
#SBATCH --output=/home/ywan0794/ReMoMask-2/slurm_logs/eval_v2_%j.log

eval "$(/home/ywan0794/miniconda3/bin/conda shell.bash hook)"
conda activate remomask
cd /home/ywan0794/ReMoMask-2
mkdir -p slurm_logs

echo "=== Job: Evaluate V2 (z_e retrieval, 20 repeats) ==="
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Verify V2 checkpoint exists
if [ ! -f logs/humanml3d/v2_ze_rtval/model/net_best_fid.tar ]; then
    echo "ERROR: V2 checkpoint not found. Was job2 successful?"
    exit 1
fi

python eval_mask.py \
    --mtrans_name v2_ze_rtval \
    --dataset_name humanml3d \
    --vq_name pretrain_vq \
    --gpu_id 0 \
    --repeat_times 20 \
    --which_epoch best_fid \
    --use_ze_retrieval \
    --ze_database_path database_ze \
    --projector_path logs/query_projector/best_projector.pt \
    --rt_in_value \
    --retrieval_dim 1024 \
    2>&1

echo "=== Results ==="
cat logs/humanml3d/v2_ze_rtval/eval/*.log 2>/dev/null | tail -20
echo "End: $(date)"
