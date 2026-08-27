#!/usr/bin/env bash
# Collect and evaluate a fresh axis-plus-joint candidate-direction pool.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/tmp/joint_candidate_directions_20260827"
DETAILS="${OUTPUT_ROOT}/details.jsonl"
POOLED="${OUTPUT_ROOT}/pooled_utilities.jsonl"
ANALYSIS="${REPO_ROOT}/results/joint_candidate_directions_20260827.json"
WORKERS="${JOINT_DIRECTION_WORKERS:-2}"

if (( WORKERS < 1 || WORKERS > 4 )); then
    echo "JOINT_DIRECTION_WORKERS must be between 1 and 4" >&2
    exit 2
fi
mkdir -p "$OUTPUT_ROOT"
exec 9>"${OUTPUT_ROOT}/run.lock"
if ! flock -n 9; then
    echo "Another joint candidate-direction experiment is already running" >&2
    exit 2
fi

if [[ ! -s "$DETAILS" ]]; then
    "${REPO_ROOT}/scripts/project_python.sh" \
        "${REPO_ROOT}/scripts/collect_search_utility_teachers.py" \
        --roots-file "${REPO_ROOT}/tmp/proxy_gap_v2/valA/roots.bin" \
        --candidate-engine "${REPO_ROOT}/bin/YaneuraOu-expert-blending" \
        --candidate-eval-dir "${REPO_ROOT}/bin/eval" \
        --expert-blending-dir "${REPO_ROOT}/tmp/proxy_gap_actual500k_release" \
        --reference-engine "${REPO_ROOT}/bin/YaneuraOu-by-gcc" \
        --reference-eval-dir "${REPO_ROOT}/bin/eval" \
        --candidate-nodes 1000000 --reference-nodes 1000000 \
        --bias-file "${REPO_ROOT}/configs/joint_candidate_biases_20260827.json" \
        --minimum-improvement-cp 10 --start-root 1200 --max-roots 768 \
        --workers "$WORKERS" --hash-mb 64 \
        --output "${OUTPUT_ROOT}/teacher.npy" --details "$DETAILS" \
        > /tmp/joint_candidate_directions_collect.log 2>&1
fi

if [[ ! -s "$POOLED" ]]; then
    "${REPO_ROOT}/scripts/project_python.sh" \
        "${REPO_ROOT}/scripts/rescore_disagreement_pool.py" \
        --details "$DETAILS" \
        --reference-engine "${REPO_ROOT}/bin/YaneuraOu-by-gcc" \
        --reference-eval-dir "${REPO_ROOT}/bin/eval" \
        --nodes 1000000 --workers "$WORKERS" --hash-mb 64 \
        --output "$POOLED" \
        > /tmp/joint_candidate_directions_rescore.log 2>&1
fi

"${REPO_ROOT}/scripts/project_python.sh" \
    "${REPO_ROOT}/scripts/analyze_joint_candidate_directions.py" \
    --details "$DETAILS" --pooled-utilities "$POOLED" \
    --bias-file "${REPO_ROOT}/configs/joint_candidate_biases_20260827.json" \
    --selection-roots 384 --output "$ANALYSIS" \
    > /tmp/joint_candidate_directions_analysis.log 2>&1
