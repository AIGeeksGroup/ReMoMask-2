#!/bin/bash
#SBATCH --job-name=p4_e01_v2
#SBATCH --nodes=1
#SBATCH --nodelist=persephone
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=24:00:00
#SBATCH --output=/home/ywan0794/ReMoMask-2/slurm_logs/p4_e01_v2_%j.log

set -e
eval "$(/home/ywan0794/miniconda3/bin/conda shell.bash hook)"
conda activate remomask
cd /home/ywan0794/ReMoMask-2
mkdir -p slurm_logs

echo "=== Phase4 E01: full-pipeline eval V2 (v2_ze_rtval @ ep0409, 20 repeats) ==="
echo "Protocol: cond_scale=4 time_steps=10 seed=10107 (Phase-1 protocol, same for V1/V2)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

test -f logs/humanml3d/v2_ze_rtval/model/net_best_fid_ep0409.tar
test -f logs/humanml3d/pretrain_rtrans/model/net_best_fid.tar
test -f logs/query_projector/best_projector.pt
test -f database_ze/metadata.json

python eval_res.py \
    --mtrans_name v2_ze_rtval \
    --rtrans_name pretrain_rtrans \
    --which_ckpt net_best_fid.tar \
    --which_epoch net_best_fid_ep0409 \
    --dataset_name humanml3d \
    --vq_name pretrain_vq \
    --gpu_id 0 \
    --repeat_times 20 \
    --cond_scale 4 \
    --time_steps 10 \
    --seed 10107 \
    --use_ze_retrieval \
    --ze_database_path database_ze \
    --projector_path logs/query_projector/best_projector.pt \
    --rt_in_value \
    --retrieval_dim 1024 \
    --ext phase4_e01_v2 \
    2>&1

echo "=== Results tail ==="
tail -30 logs/humanml3d/v2_ze_rtval/eval/*phase4_e01_v2* 2>/dev/null || true
echo "End: $(date)"
