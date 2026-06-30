#!/bin/bash
# ============================================================================
# Submit full V2 training + evaluation pipeline to SLURM
#
# Chain: build_db_ze -> train_projector -> train_v2 -> eval_v2 -> eval_v1
# Each job depends on the previous one via --dependency=afterok
#
# Usage:
#   bash scripts/slurm/submit_pipeline.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================"
echo " Submitting V2 Training Pipeline (5 jobs, dependency-chained)"
echo "================================================================"

# Job 0: Build z_e database (~30 min)
JOB0=$(sbatch --parsable job0_build_db_ze.sh)
echo "[Job 0] Build database_ze:    $JOB0"

# Job 1: Train query projector (~2h, depends on db)
JOB1=$(sbatch --parsable --dependency=afterok:$JOB0 job1_train_projector.sh)
echo "[Job 1] Train projector:      $JOB1 (after $JOB0)"

# Job 2: Train V2 MaskTransformer (~24-48h, 4-GPU DDP, depends on projector)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 job2_train_v2.sh)
echo "[Job 2] Train V2 (4xL40):     $JOB2 (after $JOB1)"

# Job 3: Evaluate V2 (20 repeats, ~2-4h, depends on V2 training)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 job3_eval_v2.sh)
echo "[Job 3] Eval V2:              $JOB3 (after $JOB2)"

# Job 4: Evaluate V1 baseline (20 repeats, ~2-4h, depends on V2 eval)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 job4_eval_v1.sh)
echo "[Job 4] Eval V1 baseline:     $JOB4 (after $JOB3)"

echo ""
echo "================================================================"
echo " Pipeline submitted! Monitor with: squeue -u \$USER"
echo ""
echo " Job chain: $JOB0 -> $JOB1 -> $JOB2 -> $JOB3 -> $JOB4"
echo ""
echo " Estimated total time: ~30-54 hours"
echo "   Job 0 (build db):       ~30 min"
echo "   Job 1 (projector):      ~2 hours"
echo "   Job 2 (V2 training):    ~24-48 hours"
echo "   Job 3 (eval V2):        ~2-4 hours"
echo "   Job 4 (eval V1):        ~2-4 hours"
echo ""
echo " After completion, run comparison:"
echo "   python scripts/compare_v1_v2.py \\"
echo "     --v1_dir logs/humanml3d/pretrain_mtrans/eval \\"
echo "     --v2_dir logs/humanml3d/v2_ze_rtval/eval \\"
echo "     --output results/plan_a_ablation/"
echo "================================================================"
