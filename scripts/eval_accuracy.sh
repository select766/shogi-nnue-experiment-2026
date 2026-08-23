#!/usr/bin/env bash
# Canonical move-accuracy evaluation entry point.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?Usage: $0 CONFIG DATASET OUTPUT [LOG_FILE]}"
DATASET="${2:?Usage: $0 CONFIG DATASET OUTPUT [LOG_FILE]}"
OUTPUT="${3:?Usage: $0 CONFIG DATASET OUTPUT [LOG_FILE]}"
LOG_FILE="${4:-/tmp/eval_accuracy_$(basename "$OUTPUT" .json).log}"

echo "Evaluation log: $LOG_FILE"
echo "Result JSON: $OUTPUT"
"${REPO_ROOT}/scripts/gpu_python.sh" -c 'pass'
"${REPO_ROOT}/scripts/project_python.sh" -m train_nnue.eval_accuracy \
    --project-root "$REPO_ROOT" \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    > "$LOG_FILE" 2>&1
