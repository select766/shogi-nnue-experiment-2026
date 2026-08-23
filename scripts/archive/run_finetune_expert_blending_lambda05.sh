#!/bin/bash
# Fine-tune Expert Blending v4 model with lambda=0.5 (mix evaluation + game results).
# Starts from a specific checkpoint of the lambda=1.0 training run.
# LR is 1/10 of the original training script.
#
# Usage: bash scripts/run_finetune_expert_blending_lambda05.sh
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Source checkpoint to fine-tune from
SRC_CKPT="${SCRIPT_ROOT}/logs/expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/510.ckpt"

LOGDIR="${SCRIPT_ROOT}/logs/expert_blending_8experts_v4_paired_uniform50_noise0_lambda05"
LOG_FILE="/tmp/train_expert_blending_lambda05.log"

bash "${SCRIPT_ROOT}/scripts/run_train_expert_blending_8experts_v4.sh" \
  --logdir "$LOGDIR" \
  --log-file "$LOG_FILE" \
  --train "${SCRIPT_ROOT}/dataset/split_v1_paired_uniform_50/train" \
  --val "${SCRIPT_ROOT}/dataset/split_v1_paired_uniform_50/val1" \
  -- \
  --feature-set "HalfKP" \
  --n-experts 8 \
  --adapter-hidden 128 \
  --adapter-noise-scale 0.0 \
  --batch-size 256 \
  --train-shuffle-buffer-size 64 \
  --epoch-size 1000000 \
  --lr-nnue 0.001 \
  --lr-adapter 0.01 \
  --lambda 0.5 \
  --label-smoothing-eps 0.001 \
  --score-scaling 361 \
  --num-batches-warmup 10000 \
  --newbob-decay 0.5 \
  --num-epochs-to-adjust-lr 20 \
  --min-newbob-scale 1e-5 \
  --momentum 0.9 \
  --network-save-period 10 \
  --max-epochs 1000000 \
  --gpus 1 \
  --seed 42 \
  --load-weights-only "$SRC_CKPT"
