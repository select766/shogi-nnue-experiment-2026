#!/usr/bin/env bash
# Evaluate the deployment-qualified combined root-feature candidate.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${REPO_ROOT}/data/accuracy_eval_10k/validation.jsonl"
CONTROL="${REPO_ROOT}/results/accuracy_eval_tail_objective_mean_validation.json"
CANDIDATE="${REPO_ROOT}/results/accuracy_eval_root_features_combined_validation.json"

bash "${REPO_ROOT}/scripts/export_expert_blending.sh" \
    "${REPO_ROOT}/logs/root_features_combined_from_m0/checkpoints/0.ckpt" \
    "${REPO_ROOT}/tmp/root_features_combined_release" 8
bash "${REPO_ROOT}/scripts/eval_accuracy.sh" \
    "${REPO_ROOT}/configs/accuracy_eval_root_features_combined.json" \
    "$DATASET" "$CANDIDATE" \
    /tmp/accuracy_eval_root_features_combined_validation.log
"${REPO_ROOT}/scripts/project_python.sh" -m train_nnue.compare_accuracy \
    --baseline "$CONTROL" --candidate "$CANDIDATE" --dataset "$DATASET" \
    --output "${REPO_ROOT}/results/accuracy_comparison_root_features_combined_validation.json"
