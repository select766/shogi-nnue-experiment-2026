#!/bin/bash
# Expert Blending v4 (8 experts, paired) entry script.
# Hyperparameters are fixed in this script; common resume logic is delegated.
#
# Usage: bash scripts/run_train_expert_blending_8experts_v4_paired_noise1p0.sh
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="${SCRIPT_ROOT}/logs/expert_blending_8experts_v4_paired_noise1p0"
LOG_FILE="/tmp/train_expert_blending_8experts_v4_paired_noise1p0.log"

bash "${SCRIPT_ROOT}/scripts/run_train_expert_blending_8experts_v4.sh" \
  --logdir "$LOGDIR" \
  --log-file "$LOG_FILE" \
  --paired \
  --paired-nnue-cache-dir "${SCRIPT_ROOT}/tmp/paired_nnue_cache" \
  -- \
  --feature-set "HalfKP" \
  --n-experts 8 \
  --adapter-hidden 128 \
  --adapter-noise-scale 1.0 \
  --batch-size 256 \
  --epoch-size 1000000 \
  --lr-nnue 0.01 \
  --lr-adapter 0.1 \
  --lambda 1.0 \
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
  --seed 42
