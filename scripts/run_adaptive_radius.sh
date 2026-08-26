#!/usr/bin/env bash
# Collect matched 1M-node utility candidates for the pre-registered radius set.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/tmp/adaptive_radius_20260826"
mkdir -p "$OUTPUT_ROOT"

for radius in 0.5 1.0 2.0 4.0; do
    tag="${radius/./p}"
    output_dir="${OUTPUT_ROOT}/radius_${tag}"
    mkdir -p "$output_dir"
    "${REPO_ROOT}/scripts/project_python.sh" \
        "${REPO_ROOT}/scripts/collect_search_utility_teachers.py" \
        --roots-file "${REPO_ROOT}/tmp/search_utility_v1/val/roots.bin" \
        --candidate-engine "${REPO_ROOT}/bin/YaneuraOu-expert-blending" \
        --candidate-eval-dir "${REPO_ROOT}/bin/eval" \
        --expert-blending-dir "${REPO_ROOT}/tmp/proxy_gap_actual500k_release" \
        --reference-engine "${REPO_ROOT}/bin/YaneuraOu-by-gcc" \
        --reference-eval-dir "${REPO_ROOT}/bin/eval" \
        --candidate-nodes 1000000 \
        --reference-nodes 1000000 \
        --logit-bias "$radius" \
        --candidate-geometry positive-axis \
        --minimum-improvement-cp 10 \
        --start-root 192 \
        --max-roots 192 \
        --workers 4 \
        --hash-mb 64 \
        --output "${output_dir}/teacher.npy" \
        --details "${output_dir}/details.jsonl" \
        > "/tmp/adaptive_radius_${tag}.log" 2>&1
done

"${REPO_ROOT}/scripts/project_python.sh" \
    "${REPO_ROOT}/scripts/analyze_adaptive_radius.py" \
    --details 0.5 "${OUTPUT_ROOT}/radius_0p5/details.jsonl" \
    --details 1.0 "${OUTPUT_ROOT}/radius_1p0/details.jsonl" \
    --details 2.0 "${OUTPUT_ROOT}/radius_2p0/details.jsonl" \
    --details 4.0 "${OUTPUT_ROOT}/radius_4p0/details.jsonl" \
    --selection-roots 96 \
    --output "${REPO_ROOT}/results/adaptive_radius_20260826.json" \
    > /tmp/adaptive_radius_analysis.log 2>&1
