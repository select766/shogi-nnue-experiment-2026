#!/usr/bin/env bash
# Export the held-out-teacher-selected epochs and evaluate fixed validation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${REPO_ROOT}/data/accuracy_eval_10k/validation.jsonl"

for spec in router_only:8 combined:7; do
    condition="${spec%%:*}"
    epoch="${spec##*:}"
    checkpoint="${REPO_ROOT}/logs/search_utility_pilot_${condition}_from_m0/checkpoints/${epoch}.ckpt"
    release="${REPO_ROOT}/tmp/search_utility_${condition}_epoch${epoch}_release"
    bash "${REPO_ROOT}/scripts/export_expert_blending.sh" "$checkpoint" "$release" 8 \
        > "/tmp/export_search_utility_${condition}_epoch${epoch}.log" 2>&1
done

pids=()
for spec in router_only:8 combined:7; do
    condition="${spec%%:*}"
    epoch="${spec##*:}"
    bash "${REPO_ROOT}/scripts/eval_accuracy.sh" \
        "${REPO_ROOT}/configs/accuracy_eval_search_utility_${condition}_epoch${epoch}.json" \
        "$DATASET" \
        "${REPO_ROOT}/results/accuracy_eval_search_utility_${condition}_epoch${epoch}_validation.json" \
        "/tmp/accuracy_eval_search_utility_${condition}_epoch${epoch}_validation.log" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
exit "$status"
