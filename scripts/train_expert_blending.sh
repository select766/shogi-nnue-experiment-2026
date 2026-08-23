#!/usr/bin/env bash
# Canonical Expert Blending training entry point with automatic resume.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="${REPO_ROOT}/dataset/split_v1_paired_uniform_50/train"
VAL_DIR="${REPO_ROOT}/dataset/split_v1_paired_uniform_50/val1"
BACKBONE_WEIGHTS="${REPO_ROOT}/tmp/dlshogi-model/model_resnet10_swish-072"
NNUE_CHECKPOINT="${REPO_ROOT}/logs/halfkp_v1/checkpoints/83000.ckpt"
RUN_NAME=""
LOG_FILE=""
TRAIN_ARGS=()
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: bash scripts/train_expert_blending.sh --run-name NAME [options] -- TRAIN_ARGS...

Options:
  --train-dir PATH
  --val-dir PATH
  --backbone-weights PATH
  --nnue-checkpoint PATH
  --log-file PATH
  --dry-run

The current uniform-50 dataset and HalfKP checkpoint 83000 are defaults.
Existing RUN_NAME/checkpoints/*.ckpt causes an automatic full-state resume.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name) RUN_NAME="${2:?missing value}"; shift 2 ;;
        --train-dir) TRAIN_DIR="$(readlink -f "${2:?missing value}")"; shift 2 ;;
        --val-dir) VAL_DIR="$(readlink -f "${2:?missing value}")"; shift 2 ;;
        --backbone-weights) BACKBONE_WEIGHTS="$(readlink -f "${2:?missing value}")"; shift 2 ;;
        --nnue-checkpoint) NNUE_CHECKPOINT="$(readlink -f "${2:?missing value}")"; shift 2 ;;
        --log-file) LOG_FILE="${2:?missing value}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --) shift; TRAIN_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$RUN_NAME" ]] || { echo "ERROR: --run-name is required" >&2; exit 2; }
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "ERROR: run name may contain only letters, numbers, dot, underscore and hyphen" >&2
    exit 2
}

for dir in "$TRAIN_DIR" "$VAL_DIR"; do
    for name in dnn.bin nnue.bin; do
        [[ -f "${dir}/${name}" ]] || { echo "ERROR: missing ${dir}/${name}" >&2; exit 1; }
    done
done
[[ -f "$BACKBONE_WEIGHTS" ]] || { echo "ERROR: missing $BACKBONE_WEIGHTS" >&2; exit 1; }
[[ -f "$NNUE_CHECKPOINT" ]] || { echo "ERROR: missing $NNUE_CHECKPOINT" >&2; exit 1; }

RUN_DIR="${REPO_ROOT}/logs/${RUN_NAME}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
LOG_FILE="${LOG_FILE:-/tmp/train_nnue_${RUN_NAME}.log}"

RESUME_ARGS=()
LATEST_CHECKPOINT=""
if [[ -d "$CHECKPOINT_DIR" ]]; then
    LATEST_CHECKPOINT="$(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name '*.ckpt' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2- || true)"
fi
if [[ -n "$LATEST_CHECKPOINT" ]]; then
    RESUME_ARGS=(--resume-from-checkpoint "$LATEST_CHECKPOINT")
    echo "Resume checkpoint: $LATEST_CHECKPOINT"
else
    echo "Starting a new run: $RUN_NAME"
fi

echo "Training log: $LOG_FILE"
echo "Monitor: tail -f $LOG_FILE"

CMD=("${REPO_ROOT}/scripts/gpu_python.sh" -m train_nnue.train_expert_blending \
    --train "$TRAIN_DIR" \
    --val "$VAL_DIR" \
    --backbone-weights "$BACKBONE_WEIGHTS" \
    --nnue-checkpoint "$NNUE_CHECKPOINT" \
    --default-root-dir "$RUN_DIR" \
    "${TRAIN_ARGS[@]}" \
    "${RESUME_ARGS[@]}")

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'Command:'
    printf ' %q' "${CMD[@]}"
    printf ' > %q 2>&1\n' "$LOG_FILE"
    exit 0
fi

mkdir -p "$CHECKPOINT_DIR"
"${CMD[@]}" > "$LOG_FILE" 2>&1
