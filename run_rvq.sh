#!/usr/bin/env bash
set -e
NAME=$1
GPU=$2
DATASET=$3
shift 3
python train_vq.py --name "$NAME" --gpu_id "$GPU" --dataset_name "$DATASET" "$@"
