#!/bin/bash
# Internal runner for Expert Blending v4 training.
# Resume/venv/log redirection are centralized here.
#
# This script is not intended to be called directly by users.
# Call from hyperparameter entry scripts and pass training options after `--`.
#
# Usage:
#   bash scripts/run_train_expert_blending_8experts_v4.sh \
#     --logdir <path> \
#     --log-file <path> \
#     [--train <path>] [--val <path>] \
#     [--backbone-weights <path>] [--nnue-checkpoint <path>] \
#     -- <train_expert_blending.py options>
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NNUE_PYTORCH_DIR="${SCRIPT_ROOT}/nnue-pytorch"

TRAIN_BIN="${SCRIPT_ROOT}/dataset/split_v1_paired/train"
VAL_BIN="${SCRIPT_ROOT}/dataset/split_v1_paired/val1"
BACKBONE_WEIGHTS="${SCRIPT_ROOT}/tmp/dlshogi-model/model_resnet10_swish-072"
NNUE_CKPT="${SCRIPT_ROOT}/logs/halfkp_v1/checkpoints/83000.ckpt"
LOGDIR=""
LOG_FILE=""
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --logdir)
            [[ $# -ge 2 ]] || { echo "ERROR: --logdir requires a value"; exit 1; }
            LOGDIR="$2"
            shift 2
            ;;
        --log-file)
            [[ $# -ge 2 ]] || { echo "ERROR: --log-file requires a value"; exit 1; }
            LOG_FILE="$2"
            shift 2
            ;;
        --train)
            [[ $# -ge 2 ]] || { echo "ERROR: --train requires a value"; exit 1; }
            TRAIN_BIN="$2"
            shift 2
            ;;
        --val)
            [[ $# -ge 2 ]] || { echo "ERROR: --val requires a value"; exit 1; }
            VAL_BIN="$2"
            shift 2
            ;;
        --backbone-weights)
            [[ $# -ge 2 ]] || { echo "ERROR: --backbone-weights requires a value"; exit 1; }
            BACKBONE_WEIGHTS="$2"
            shift 2
            ;;
        --nnue-checkpoint)
            [[ $# -ge 2 ]] || { echo "ERROR: --nnue-checkpoint requires a value"; exit 1; }
            NNUE_CKPT="$2"
            shift 2
            ;;
        --)
            shift
            TRAIN_ARGS=("$@")
            break
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$LOGDIR" || -z "$LOG_FILE" ]]; then
    echo "ERROR: --logdir and --log-file are required."
    exit 1
fi

for d in "$TRAIN_BIN" "$VAL_BIN"; do
    if [[ ! -d "$d" ]]; then
        echo "ERROR: Not found directory: ${d}"
        exit 1
    fi
    for f in dnn.bin nnue.bin; do
        if [[ ! -f "${d}/${f}" ]]; then
            echo "ERROR: Not found file: ${d}/${f}"
            exit 1
        fi
    done
done
for f in "$BACKBONE_WEIGHTS" "$NNUE_CKPT"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Not found: ${f}"
        exit 1
    fi
done

mkdir -p "$LOGDIR"
CKPT_DIR="${LOGDIR}/checkpoints"

RESUME_ARGS=()
if [[ -d "$CKPT_DIR" ]]; then
    LATEST_CKPT=$(ls -t "${CKPT_DIR}"/*.ckpt 2>/dev/null | head -1 || true)
    if [[ -n "${LATEST_CKPT}" ]]; then
        echo "Found checkpoint: ${LATEST_CKPT}"
        echo "Resuming training..."
        RESUME_ARGS=(--resume-from-checkpoint "$LATEST_CKPT")
    fi
fi

if [[ ${#RESUME_ARGS[@]} -eq 0 ]]; then
    echo "Starting new training run."
fi

echo "Log file: ${LOG_FILE}"
echo "Monitor with: tail -f ${LOG_FILE}"

cd "$NNUE_PYTORCH_DIR"
source .venv/bin/activate

CMD=(
    python -m train_nnue.train_expert_blending
    --train "$TRAIN_BIN"
    --val "$VAL_BIN"
    --backbone-weights "$BACKBONE_WEIGHTS"
    --nnue-checkpoint "$NNUE_CKPT"
    --default-root-dir "$LOGDIR"
)

CMD+=("${TRAIN_ARGS[@]}")
CMD+=("${RESUME_ARGS[@]}")

PYTHONPATH="${SCRIPT_ROOT}/src:${PYTHONPATH:-}" "${CMD[@]}" > "$LOG_FILE" 2>&1
