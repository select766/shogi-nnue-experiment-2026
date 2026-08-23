#!/usr/bin/env bash
# Canonical fixed-search self-play evaluation entry point.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:?Usage: $0 OUTPUT_JSON [run_match options...]}"
shift
LOG_FILE="/tmp/eval_match_$(basename "$OUTPUT" .json).log"

echo "Match log: $LOG_FILE"
echo "Result JSON: $OUTPUT"
"${REPO_ROOT}/scripts/project_python.sh" -m train_nnue.run_match \
    --output "$OUTPUT" \
    "$@" \
    > "$LOG_FILE" 2>&1
