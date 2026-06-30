#!/bin/bash
#SBATCH --job-name=eval_v1
#SBATCH --nodes=1
#SBATCH --nodelist=persephone
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=06:00:00
#SBATCH --output=/home/ywan0794/ReMoMask-2/slurm_logs/eval_v1_%j.log

eval "$(/home/ywan0794/miniconda3/bin/conda shell.bash hook)"
conda activate remomask
cd /home/ywan0794/ReMoMask-2
mkdir -p slurm_logs

echo "=== Job: Evaluate V1 (Part_TMR retrieval, 20 repeats) ==="
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Verify V1 checkpoint exists
if [ ! -f logs/humanml3d/pretrain_mtrans/model/net_best_fid.tar ]; then
    echo "ERROR: V1 checkpoint not found"
    exit 1
fi

python eval_mask.py \
    --mtrans_name pretrain_mtrans \
    --dataset_name humanml3d \
    --vq_name pretrain_vq \
    --gpu_id 0 \
    --repeat_times 20 \
    --which_epoch best_fid \
    2>&1

echo "=== Results ==="
cat logs/humanml3d/pretrain_mtrans/eval/*.log 2>/dev/null | tail -20

echo ""
echo "=== Both V1 and V2 eval complete ==="
echo "Run compare_v1_v2.py to generate comparison table"
echo "End: $(date)"
