#!/bin/bash
# ============================================================================
# Plan A Ablation: V1 (Part_TMR retrieval) vs V2 (z_e retrieval)
#
# Single-variable ablation: only the retrieval module differs.
# VQ-VAE, MaskTransformer architecture, training hyperparams, and evaluation
# method are identical between V1 and V2.
#
# Usage:
#   bash scripts/run_plan_a_ablation.sh <V1_EXP> <V2_EXP> <VQ_NAME> <GPU_ID> [REPEAT]
#
# Example:
#   bash scripts/run_plan_a_ablation.sh pretrain_mtrans v2_ze_rtval pretrain_vq 0 20
# ============================================================================

set -euo pipefail

V1_EXP="${1:?Usage: $0 <V1_EXP> <V2_EXP> <VQ_NAME> <GPU_ID> [REPEAT]}"
V2_EXP="${2:?Missing V2_EXP}"
VQ_NAME="${3:?Missing VQ_NAME}"
GPU_ID="${4:-0}"
REPEAT="${5:-20}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

OUTPUT_DIR="results/plan_a_ablation"
mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo " Plan A Ablation: V1 vs V2"
echo "================================================================"
echo " V1 (Part_TMR): $V1_EXP"
echo " V2 (z_e):      $V2_EXP"
echo " VQ:            $VQ_NAME"
echo " GPU:           $GPU_ID"
echo " Repeats:       $REPEAT"
echo "================================================================"

# ---- Step 1: Evaluate V1 (Part_TMR retrieval) ----
echo ""
echo "[Step 1/3] Evaluating V1 (Part_TMR retrieval): $V1_EXP"
echo "------------------------------------------------------------"

V1_DIR="logs/humanml3d/$V1_EXP/eval"
if [ -f "$V1_DIR/eval_results.log" ] && [ "$SKIP_EXISTING" = "1" ] 2>/dev/null; then
    echo "V1 eval already exists, skipping (set SKIP_EXISTING=0 to re-run)"
else
    python eval_mask.py \
        --mtrans_name "$V1_EXP" \
        --dataset_name humanml3d \
        --vq_name "$VQ_NAME" \
        --gpu_id "$GPU_ID" \
        --repeat_times "$REPEAT" \
        --which_epoch best_fid
fi
echo "[Step 1/3] V1 evaluation complete."

# ---- Step 2: Evaluate V2 (z_e retrieval) ----
echo ""
echo "[Step 2/3] Evaluating V2 (z_e retrieval): $V2_EXP"
echo "------------------------------------------------------------"

V2_DIR="logs/humanml3d/$V2_EXP/eval"
if [ -f "$V2_DIR/eval_results.log" ] && [ "$SKIP_EXISTING" = "1" ] 2>/dev/null; then
    echo "V2 eval already exists, skipping (set SKIP_EXISTING=0 to re-run)"
else
    python eval_mask.py \
        --mtrans_name "$V2_EXP" \
        --dataset_name humanml3d \
        --vq_name "$VQ_NAME" \
        --gpu_id "$GPU_ID" \
        --repeat_times "$REPEAT" \
        --which_epoch best_fid \
        --use_ze_retrieval \
        --ze_database_path database_ze \
        --projector_path logs/query_projector/best_projector.pt \
        --rt_in_value \
        --retrieval_dim 1024
fi
echo "[Step 2/3] V2 evaluation complete."

# ---- Step 3: Compare results ----
echo ""
echo "[Step 3/3] Comparing V1 vs V2"
echo "------------------------------------------------------------"

python scripts/compare_v1_v2.py \
    --v1_dir "$V1_DIR" \
    --v2_dir "$V2_DIR" \
    --v1_name "$V1_EXP" \
    --v2_name "$V2_EXP" \
    --output "$OUTPUT_DIR"

echo ""
echo "================================================================"
echo " Ablation complete. Results in: $OUTPUT_DIR/"
echo "================================================================"
