#!/usr/bin/env bash
# Export all four pilots and evaluate them on the fixed validation split.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${REPO_ROOT}/data/accuracy_eval_10k/validation.jsonl"

for condition in c00 c10 c01 c11; do
    case "$condition" in
        c00) run_name="c00_baseline_qsearch" ;;
        c10) run_name="c10_baseline_deep" ;;
        c01) run_name="c01_onpolicy_qsearch" ;;
        c11) run_name="c11_onpolicy_deep" ;;
    esac
    checkpoint="${REPO_ROOT}/logs/objective_onpolicy_pilot_${run_name}_from510/checkpoints/0.ckpt"
    release="${REPO_ROOT}/tmp/objective_onpolicy_${condition}_release"
    bash "${REPO_ROOT}/scripts/export_expert_blending.sh" \
        "$checkpoint" "$release" 8 \
        > "/tmp/export_objective_onpolicy_${condition}.log" 2>&1
done

pids=()
for condition in c00 c10 c01 c11; do
    bash "${REPO_ROOT}/scripts/eval_accuracy.sh" \
        "${REPO_ROOT}/configs/accuracy_eval_objective_onpolicy_${condition}.json" \
        "$DATASET" \
        "${REPO_ROOT}/results/accuracy_eval_10k_objective_onpolicy_${condition}_validation.json" \
        "/tmp/accuracy_eval_objective_onpolicy_${condition}_validation.log" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
