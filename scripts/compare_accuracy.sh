#!/usr/bin/env bash
# Compare two accuracy result files position by position.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${1:?Usage: $0 BASELINE_RESULT CANDIDATE_RESULT DATASET OUTPUT}"
CANDIDATE="${2:?Usage: $0 BASELINE_RESULT CANDIDATE_RESULT DATASET OUTPUT}"
DATASET="${3:?Usage: $0 BASELINE_RESULT CANDIDATE_RESULT DATASET OUTPUT}"
OUTPUT="${4:?Usage: $0 BASELINE_RESULT CANDIDATE_RESULT DATASET OUTPUT}"

"${REPO_ROOT}/scripts/project_python.sh" -m train_nnue.compare_accuracy \
    --baseline "$BASELINE" \
    --candidate "$CANDIDATE" \
    --dataset "$DATASET" \
    --output "$OUTPUT"
