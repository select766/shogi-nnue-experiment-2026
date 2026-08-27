#!/usr/bin/env bash
# Export and evaluate the shallow-root-statistics candidate on fixed validation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${REPO_ROOT}/logs/root_search_stats_from_m0/checkpoints/0.ckpt"
RELEASE="${REPO_ROOT}/tmp/root_search_stats_release"

bash "${REPO_ROOT}/scripts/export_expert_blending.sh" "$CHECKPOINT" "$RELEASE" 8
"${REPO_ROOT}/scripts/project_python.sh" -m train_nnue.eval_accuracy_root_stats \
    --project-root "$REPO_ROOT" \
    --config "${REPO_ROOT}/configs/accuracy_eval_root_search_stats.json" \
    --dataset "${REPO_ROOT}/data/accuracy_eval_10k/validation.jsonl" \
    --output "${REPO_ROOT}/results/accuracy_eval_root_search_stats_validation.json" \
    > /tmp/accuracy_eval_root_search_stats_validation.log 2>&1
bash "${REPO_ROOT}/scripts/compare_accuracy.sh" \
    "${REPO_ROOT}/results/accuracy_eval_root_features_combined_validation.json" \
    "${REPO_ROOT}/results/accuracy_eval_root_search_stats_validation.json" \
    "${REPO_ROOT}/data/accuracy_eval_10k/validation.jsonl" \
    "${REPO_ROOT}/results/accuracy_comparison_root_search_stats_validation.json"
