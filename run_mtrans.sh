#!/usr/bin/env bash
set -e
NAME=$1
GPU_COUNT=$2
GPU_IDS=$3
DATASET=$4
shift 4
CUDA_VISIBLE_DEVICES="$GPU_IDS" torchrun --standalone --nnodes=1 --nproc_per_node="$GPU_COUNT" train_mask_transformer_ddp.py --name "$NAME" --dataset_name "$DATASET" "$@"
