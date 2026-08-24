#!/usr/bin/env bash
# Paired best-move comparisons for the pre-registered H1/H2 contrasts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${REPO_ROOT}/results"
DATASET="${REPO_ROOT}/data/accuracy_eval_10k/validation.jsonl"

compare() {
    local baseline="$1"
    local candidate="$2"
    local output="$3"
    "${REPO_ROOT}/scripts/project_python.sh" -m train_nnue.compare_accuracy \
        --baseline "${RESULTS}/accuracy_eval_10k_objective_onpolicy_${baseline}_validation.json" \
        --candidate "${RESULTS}/accuracy_eval_10k_objective_onpolicy_${candidate}_validation.json" \
        --dataset "$DATASET" \
        --output "${RESULTS}/${output}.json"
}

compare c00 c10 accuracy_comparison_objective_onpolicy_h1_baseline_validation
compare c01 c11 accuracy_comparison_objective_onpolicy_h1_onpolicy_validation
compare c00 c01 accuracy_comparison_objective_onpolicy_h2_validation
