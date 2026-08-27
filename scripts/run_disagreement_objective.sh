#!/usr/bin/env bash
# Collect a fresh matched radius pool and test a coverage-aware nine-candidate set.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/tmp/disagreement_objective_20260827"
mkdir -p "$OUTPUT_ROOT"

for radius in 0.5 1.0 2.0 4.0; do
    tag="${radius/./p}"
    output_dir="${OUTPUT_ROOT}/radius_${tag}"
    mkdir -p "$output_dir"
    "${REPO_ROOT}/scripts/project_python.sh" \
        "${REPO_ROOT}/scripts/collect_search_utility_teachers.py" \
        --roots-file "${REPO_ROOT}/tmp/proxy_gap_v2/valA/roots.bin" \
        --candidate-engine "${REPO_ROOT}/bin/YaneuraOu-expert-blending" \
        --candidate-eval-dir "${REPO_ROOT}/bin/eval" \
        --expert-blending-dir "${REPO_ROOT}/tmp/proxy_gap_actual500k_release" \
        --reference-engine "${REPO_ROOT}/bin/YaneuraOu-by-gcc" \
        --reference-eval-dir "${REPO_ROOT}/bin/eval" \
        --candidate-nodes 1000000 --reference-nodes 1000000 \
        --logit-bias "$radius" --candidate-geometry positive-axis \
        --minimum-improvement-cp 10 --start-root 1000 --max-roots 192 \
        --workers 4 --hash-mb 64 \
        --output "${output_dir}/teacher.npy" \
        --details "${output_dir}/details.jsonl" \
        > "/tmp/disagreement_objective_${tag}.log" 2>&1
done

"${REPO_ROOT}/scripts/project_python.sh" \
    "${REPO_ROOT}/scripts/rescore_disagreement_pool.py" \
    --details "${OUTPUT_ROOT}/radius_0p5/details.jsonl" \
    --details "${OUTPUT_ROOT}/radius_1p0/details.jsonl" \
    --details "${OUTPUT_ROOT}/radius_2p0/details.jsonl" \
    --details "${OUTPUT_ROOT}/radius_4p0/details.jsonl" \
    --reference-engine "${REPO_ROOT}/bin/YaneuraOu-by-gcc" \
    --reference-eval-dir "${REPO_ROOT}/bin/eval" \
    --nodes 1000000 --workers 4 --hash-mb 64 \
    --output "${OUTPUT_ROOT}/pooled_utilities.jsonl" \
    > /tmp/disagreement_objective_rescore.log 2>&1

"${REPO_ROOT}/scripts/project_python.sh" \
    "${REPO_ROOT}/scripts/analyze_disagreement_objective.py" \
    --details 0.5 "${OUTPUT_ROOT}/radius_0p5/details.jsonl" \
    --details 1.0 "${OUTPUT_ROOT}/radius_1p0/details.jsonl" \
    --details 2.0 "${OUTPUT_ROOT}/radius_2p0/details.jsonl" \
    --details 4.0 "${OUTPUT_ROOT}/radius_4p0/details.jsonl" \
    --pooled-utilities "${OUTPUT_ROOT}/pooled_utilities.jsonl" \
    --selection-roots 96 --fixed-radius 4.0 \
    --output "${REPO_ROOT}/results/disagreement_objective_20260827.json" \
    > /tmp/disagreement_objective_analysis.log 2>&1
