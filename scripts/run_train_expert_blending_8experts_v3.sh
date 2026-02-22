#!/bin/bash
# Expert Blending モデル (8 experts) の学習スクリプト。
# dataset/split_v1_paired/ (train/{dnn.bin,nnue.bin}, val1/{dnn.bin,nnue.bin}) を使用する。
# Stop with Ctrl-C, then re-run to resume from latest checkpoint.
#
# Usage: bash scripts/run_train_expert_blending_8experts_v3.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NNUE_PYTORCH_DIR="${SCRIPT_DIR}/nnue-pytorch"
LOGDIR="${SCRIPT_DIR}/logs/expert_blending_8experts_v3"
CKPT_DIR="${LOGDIR}/checkpoints"
LOG_FILE="/tmp/train_expert_blending_8experts_v3.log"

# Data
SPLIT_BASE="${SCRIPT_DIR}/dataset/split_v1_paired"
TRAIN_BIN="${SPLIT_BASE}/train"
VAL_BIN="${SPLIT_BASE}/val1"

# Model weights
BACKBONE_WEIGHTS="${SCRIPT_DIR}/tmp/dlshogi-model/model_resnet10_swish-072"
NNUE_CKPT="${SCRIPT_DIR}/logs/halfkp_v1/checkpoints/83000.ckpt"

# Check files exist
for d in "$TRAIN_BIN" "$VAL_BIN"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: Not found directory: ${d}"
        exit 1
    fi
    for f in dnn.bin nnue.bin; do
        if [ ! -f "${d}/${f}" ]; then
            echo "ERROR: Not found file: ${d}/${f}"
            exit 1
        fi
    done
done
for f in "$BACKBONE_WEIGHTS" "$NNUE_CKPT"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Not found: ${f}"
        exit 1
    fi
done

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

PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}" python -m train_nnue.train_expert_blending \
  --train "$TRAIN_BIN" \
  --val "$VAL_BIN" \
  --backbone-weights "$BACKBONE_WEIGHTS" \
  --nnue-checkpoint "$NNUE_CKPT" \
  --feature-set "HalfKP" \
  --n-experts 8 \
  --adapter-hidden 128 \
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
  --default-root-dir "$LOGDIR" \
  --seed 42 \
  $RESUME_ARG \
  > "$LOG_FILE" 2>&1
