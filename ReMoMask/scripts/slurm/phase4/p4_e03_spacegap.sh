#!/bin/bash
#SBATCH --job-name=p4_e03_gap
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ywan0794/ReMoMask-2/slurm_logs/p4_e03_%j.log

set -e
eval "$(/home/ywan0794/miniconda3/bin/conda shell.bash hook)"
conda activate remomask
cd /home/ywan0794/ReMoMask-2
mkdir -p slurm_logs results/phase4

echo "=== Phase4 E03: cross-space rank correlation (BMM vs z_e), N=2000 ==="
echo "Start: $(date)"

# full databases required (local dev copy of database/ is the 32-entry stub; remote is the 66,912 rebuild)
python -c "import numpy as np; m=np.load('database/encoded_motions.npy'); assert m.shape[0]>60000, f'BMM db too small: {m.shape}'; print('BMM db OK:', m.shape)"

python scripts/analyze_space_gap.py \
    --database_bmm database \
    --database_ze database_ze \
    --sample 2000 --seed 3407 \
    --topk 1 5 10 50 \
    --out results/phase4/e03_space_gap \
    2>&1

echo "=== summary.json ==="
cat results/phase4/e03_space_gap/summary.json
echo "End: $(date)"
