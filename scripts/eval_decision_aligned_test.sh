#!/usr/bin/env bash
# Run the pre-registered test split after a positive validation result.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${REPO_ROOT}/data/accuracy_eval_10k/test.jsonl"
CONTROL="${REPO_ROOT}/results/accuracy_eval_decision_aligned_control_test.json"
CANDIDATE="${REPO_ROOT}/results/accuracy_eval_decision_aligned_candidate_test.json"

bash "${REPO_ROOT}/scripts/eval_accuracy.sh" \
    "${REPO_ROOT}/configs/accuracy_eval_decision_aligned_control_test.json" \
    "$DATASET" "$CONTROL" /tmp/accuracy_eval_decision_aligned_control_test.log
bash "${REPO_ROOT}/scripts/eval_accuracy.sh" \
    "${REPO_ROOT}/configs/accuracy_eval_decision_aligned_candidate_test.json" \
    "$DATASET" "$CANDIDATE" /tmp/accuracy_eval_decision_aligned_candidate_test.log
"${REPO_ROOT}/scripts/project_python.sh" -m train_nnue.compare_accuracy \
    --baseline "$CONTROL" --candidate "$CANDIDATE" --dataset "$DATASET" \
    --output "${REPO_ROOT}/results/accuracy_comparison_decision_aligned_test.json"
