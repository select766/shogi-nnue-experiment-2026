#!/usr/bin/env bash
# Canonical GPU gate-diagnostics entry point for the current paired validation set.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:?Usage: $0 CHECKPOINT OUTPUT_JSON [diagnose_gate options...]}"
OUTPUT="${2:?Usage: $0 CHECKPOINT OUTPUT_JSON [diagnose_gate options...]}"
shift 2
LOG_FILE="/tmp/diagnose_gate_$(basename "$OUTPUT" .json).log"

echo "Gate diagnosis log: $LOG_FILE"
echo "Result JSON: $OUTPUT"
"${REPO_ROOT}/scripts/gpu_python.sh" -u -m train_nnue.diagnose_gate \
    --checkpoint "$CHECKPOINT" \
    --val dataset/split_v1_paired_uniform_50/val1 \
    --backbone-weights tmp/dlshogi-model/model_resnet10_swish-072 \
    --nnue-checkpoint logs/halfkp_v1/checkpoints/83000.ckpt \
    --feature-set HalfKP \
    --max-positions 10000 \
    --output "$OUTPUT" \
    "$@" \
    > "$LOG_FILE" 2>&1
