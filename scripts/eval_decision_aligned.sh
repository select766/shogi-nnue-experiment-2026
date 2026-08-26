#!/usr/bin/env bash
# Export the pre-selected decision model and run matched fixed validation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${REPO_ROOT}/data/accuracy_eval_10k/validation.jsonl"

bash "${REPO_ROOT}/scripts/export_expert_blending.sh" \
    "${REPO_ROOT}/logs/decision_aligned_control_from_m0/checkpoints/9.ckpt" \
    "${REPO_ROOT}/tmp/decision_aligned_control_release" 8 \
    > /tmp/export_decision_aligned_control.log 2>&1
bash "${REPO_ROOT}/scripts/export_expert_blending.sh" \
    "${REPO_ROOT}/logs/decision_aligned_weight005_from_m0/checkpoints/9.ckpt" \
    "${REPO_ROOT}/tmp/decision_aligned_candidate_release" 8 \
    > /tmp/export_decision_aligned_candidate.log 2>&1

bash "${REPO_ROOT}/scripts/eval_accuracy.sh" \
    "${REPO_ROOT}/configs/accuracy_eval_decision_aligned_control.json" \
    "$DATASET" \
    "${REPO_ROOT}/results/accuracy_eval_decision_aligned_control_validation.json" \
    /tmp/accuracy_eval_decision_aligned_control_validation.log
bash "${REPO_ROOT}/scripts/eval_accuracy.sh" \
    "${REPO_ROOT}/configs/accuracy_eval_decision_aligned_candidate.json" \
    "$DATASET" \
    "${REPO_ROOT}/results/accuracy_eval_decision_aligned_candidate_validation.json" \
    /tmp/accuracy_eval_decision_aligned_candidate_validation.log
"${REPO_ROOT}/scripts/project_python.sh" -m train_nnue.compare_accuracy \
    --baseline "${REPO_ROOT}/results/accuracy_eval_decision_aligned_control_validation.json" \
    --candidate "${REPO_ROOT}/results/accuracy_eval_decision_aligned_candidate_validation.json" \
    --dataset "$DATASET" \
    --output "${REPO_ROOT}/results/accuracy_comparison_decision_aligned_validation.json"
