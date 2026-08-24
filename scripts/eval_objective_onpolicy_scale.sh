#!/usr/bin/env bash
# Compare the five on-policy scale epochs directly with baseline-policy M0.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${REPO_ROOT}/tmp/objective_onpolicy_v1"
RUN="${REPO_ROOT}/logs/objective_onpolicy_scale_curve_from510/checkpoints"
M0="${REPO_ROOT}/logs/proxy_gap_actual500k_curve_from510/checkpoints/4.ckpt"

for distribution in val_baseline_qsearch val_onpolicy; do
    command=(
        "${REPO_ROOT}/scripts/gpu_python.sh" -u -m
        train_nnue.compare_root_grouped_checkpoints
        --control "$M0"
        --data "${DATA}/${distribution}"
        --backbone-weights "${REPO_ROOT}/tmp/dlshogi-model/model_resnet10_swish-072"
        --nnue-checkpoint "${REPO_ROOT}/logs/halfkp_v1/checkpoints/83000.ckpt"
        --max-roots 17920
        --root-batch-size 256
        --output "${REPO_ROOT}/results/objective_onpolicy_scale_${distribution}.json"
    )
    for epoch in 0 1 2 3 4; do
        command+=(--candidate "${RUN}/${epoch}.ckpt")
    done
    "${command[@]}" \
        > "/tmp/objective_onpolicy_scale_${distribution}.log" 2>&1
done
