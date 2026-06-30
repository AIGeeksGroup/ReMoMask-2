#!/bin/bash
#SBATCH --job-name=v2_train
#SBATCH --nodes=1
#SBATCH --nodelist=persephone
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH --output=/home/ywan0794/ReMoMask-2/slurm_logs/v2_train_%j.log

eval "$(/home/ywan0794/miniconda3/bin/conda shell.bash hook)"
conda activate remomask
cd /home/ywan0794/ReMoMask-2
mkdir -p slurm_logs

echo "=== Job: Train V2 MaskTransformer (4-GPU DDP) ==="
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Verify prerequisites
if [ ! -f logs/query_projector/best_projector.pt ]; then
    echo "ERROR: projector not found. Was job1 successful?"
    exit 1
fi
if [ ! -f database_ze/metadata.json ]; then
    echo "ERROR: database_ze not found. Was job0 successful?"
    exit 1
fi

# 4-GPU DDP training with torchrun
torchrun --nproc_per_node=4 \
    train_mask_transformer_ddp.py \
    --name v2_ze_rtval \
    --dataset_name humanml3d \
    --use_ze_retrieval \
    --ze_database_path database_ze \
    --projector_path logs/query_projector/best_projector.pt \
    --rt_in_value \
    --vq_name pretrain_vq \
    --batch_size 64 \
    --max_epoch 2000 \
    --train_split train.txt \
    --val_split val.txt \
    --attnj --attnt \
    2>&1

echo "=== Verifying output ==="
if [ -f logs/humanml3d/v2_ze_rtval/model/net_best_fid.tar ]; then
    echo "SUCCESS: V2 training complete"
    ls -la logs/humanml3d/v2_ze_rtval/model/net_best_fid.tar
else
    echo "WARNING: net_best_fid.tar not found (training may still be in progress or failed)"
    ls -la logs/humanml3d/v2_ze_rtval/model/ 2>/dev/null || echo "Model dir not found"
fi
echo "End: $(date)"
