#!/bin/bash
# Train HalfKP NNUE model with auto-resume support.
# Stop with Ctrl-C, then re-run the same script to resume.
#
# Usage: bash scripts/run_train_halfkp.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NNUE_PYTORCH_DIR="${SCRIPT_DIR}/nnue-pytorch"
SPLIT_BASE="./dataset/split_v1"
LOGDIR="${SCRIPT_DIR}/logs/halfkp_v1"
CKPT_DIR="${LOGDIR}/checkpoints"
LOG_FILE="/tmp/train_nnue_halfkp.log"

TRAIN_BIN="${SPLIT_BASE}/train.bin"
VAL_BIN="${SPLIT_BASE}/val1.bin"

# Check data files exist
if [ ! -f "$TRAIN_BIN" ]; then
    echo "ERROR: Training data not found: ${TRAIN_BIN}"
    echo "Run split_and_shuffle.py and run_shuffle_splits.sh first."
    exit 1
fi
if [ ! -f "$VAL_BIN" ]; then
    echo "ERROR: Validation data not found: ${VAL_BIN}"
    exit 1
fi

# Find latest checkpoint for resume
RESUME_ARG=""
if [ -d "$CKPT_DIR" ]; then
    LATEST_CKPT=$(ls -t "${CKPT_DIR}"/*.ckpt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "Found checkpoint: ${LATEST_CKPT}"
        echo "Resuming training..."
        RESUME_ARG="--resume-from-checkpoint ${LATEST_CKPT}"
    fi
fi

if [ -z "$RESUME_ARG" ]; then
    echo "Starting new training run."
fi

cd "$NNUE_PYTORCH_DIR"
source .venv/bin/activate

echo "Log file: ${LOG_FILE}"
echo "Monitor with: tail -f ${LOG_FILE}"

python train.py \
  --features "HalfKP" \
  --batch-size 16384 \
  --max_epochs 1000000 \
  --enable_progress_bar False \
  --default_root_dir "$LOGDIR" \
  --threads 8 \
  --lr 0.5 0.05 \
  --num-workers 1 \
  --lambda 1.0 0.5 \
  --label-smoothing-eps 0.001 \
  --accelerator gpu \
  --devices 1 \
  --score-scaling 361 \
  --min-newbob-scale 1e-5 \
  --epoch-size 1000000 \
  --num-epochs-to-adjust-lr 500 \
  --momentum 0.9 \
  --network-save-period 500 \
  $RESUME_ARG \
  "$TRAIN_BIN" \
  "$VAL_BIN" \
  > "$LOG_FILE" 2>&1
