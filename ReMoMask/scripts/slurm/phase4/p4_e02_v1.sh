#!/bin/bash
#SBATCH --job-name=p4_e02_v1
#SBATCH --nodes=1
#SBATCH --nodelist=hades
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ywan0794/ReMoMask-2/slurm_logs/p4_e02_v1_%j.log

set -e
eval "$(/home/ywan0794/miniconda3/bin/conda shell.bash hook)"
conda activate remomask
cd /home/ywan0794/ReMoMask-2
mkdir -p slurm_logs

echo "=== Phase4 E02: mask-only eval V1 (v1_retrain_rtval @ ep0316, 20 repeats) ==="
echo "Protocol: cond_scale=4 time_steps=10 seed=10107"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

test -f logs/humanml3d/v1_retrain_rtval/model/net_best_fid_ep0316.tar

python eval_mask.py \
    --mtrans_name v1_retrain_rtval \
    --which_epoch net_best_fid_ep0316 \
    --dataset_name humanml3d \
    --vq_name pretrain_vq \
    --gpu_id 0 \
    --repeat_times 20 \
    --cond_scale 4 \
    --time_steps 10 \
    --seed 10107 \
    --rt_in_value \
    --ext phase4_e02_v1 \
    2>&1

echo "=== Results tail ==="
tail -30 logs/humanml3d/v1_retrain_rtval/eval/*phase4_e02_v1* 2>/dev/null || true
echo "End: $(date)"
