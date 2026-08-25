#!/usr/bin/env bash
# Paired comparisons against the previously evaluated M0 result.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for spec in router_only:8 combined:7; do
    condition="${spec%%:*}"
    epoch="${spec##*:}"
    "${REPO_ROOT}/scripts/project_python.sh" -m train_nnue.compare_accuracy \
        --baseline "${REPO_ROOT}/results/accuracy_eval_10k_proxy_gap_actual500k_validation.json" \
        --candidate "${REPO_ROOT}/results/accuracy_eval_search_utility_${condition}_epoch${epoch}_validation.json" \
        --dataset "${REPO_ROOT}/data/accuracy_eval_10k/validation.jsonl" \
        --output "${REPO_ROOT}/results/accuracy_comparison_search_utility_${condition}_epoch${epoch}_validation.json"
done
