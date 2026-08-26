#!/usr/bin/env bash
# Collect matched 1M-node utility for M0 and the stage-2-selected candidate.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANDIDATE_CHECKPOINT="${1:?Usage: $0 CANDIDATE_CHECKPOINT CANDIDATE_TAG}"
CANDIDATE_TAG="${2:?Usage: $0 CANDIDATE_CHECKPOINT CANDIDATE_TAG}"
ROOTS="${REPO_ROOT}/tmp/proxy_gap_v2/valA/roots.bin"
M0_RELEASE="${REPO_ROOT}/tmp/proxy_gap_actual500k_release"
CANDIDATE_RELEASE="${REPO_ROOT}/tmp/task_aligned_experts_${CANDIDATE_TAG}_release"

bash "${REPO_ROOT}/scripts/export_expert_blending.sh" \
    "$CANDIDATE_CHECKPOINT" "$CANDIDATE_RELEASE" 8 \
    > "/tmp/export_task_aligned_experts_${CANDIDATE_TAG}.log" 2>&1

collect() {
    local name="$1"
    local release="$2"
    local output_dir="${REPO_ROOT}/tmp/task_aligned_experts_utility_${name}"
    mkdir -p "$output_dir"
    "${REPO_ROOT}/scripts/project_python.sh" \
        "${REPO_ROOT}/scripts/collect_search_utility_teachers.py" \
        --roots-file "$ROOTS" \
        --candidate-engine "${REPO_ROOT}/bin/YaneuraOu-expert-blending" \
        --candidate-eval-dir "${REPO_ROOT}/bin/eval" \
        --expert-blending-dir "$release" \
        --reference-engine "${REPO_ROOT}/bin/YaneuraOu-by-gcc" \
        --reference-eval-dir "${REPO_ROOT}/bin/eval" \
        --candidate-nodes 1000000 --reference-nodes 1000000 \
        --logit-bias 1.0 --candidate-geometry positive-axis \
        --minimum-improvement-cp 10 --start-root 0 --max-roots 64 \
        --workers 4 --hash-mb 64 \
        --output "${output_dir}/teacher.npy" \
        --details "${output_dir}/details.jsonl" \
        > "/tmp/task_aligned_experts_utility_${name}.log" 2>&1
}

collect m0 "$M0_RELEASE"
collect "$CANDIDATE_TAG" "$CANDIDATE_RELEASE"
"${REPO_ROOT}/scripts/project_python.sh" \
    "${REPO_ROOT}/scripts/compare_search_utility_summaries.py" \
    --baseline "${REPO_ROOT}/tmp/task_aligned_experts_utility_m0/teacher.json" \
    --candidate "${REPO_ROOT}/tmp/task_aligned_experts_utility_${CANDIDATE_TAG}/teacher.json" \
    --output "${REPO_ROOT}/results/task_aligned_experts_utility.json" \
    > /tmp/task_aligned_experts_utility_comparison.log 2>&1
