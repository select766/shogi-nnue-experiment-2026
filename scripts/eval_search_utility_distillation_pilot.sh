#!/usr/bin/env bash
# Compare every pilot epoch with M0 on held-out search-utility teachers.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for condition in router_only combined; do
    command=(
        "${REPO_ROOT}/scripts/gpu_python.sh" -u -m
        train_nnue.compare_root_grouped_checkpoints
        --control "${REPO_ROOT}/logs/proxy_gap_actual500k_curve_from510/checkpoints/4.ckpt"
        --data "${REPO_ROOT}/tmp/search_utility_v1/val"
        --teacher "${REPO_ROOT}/tmp/search_utility_v1/val_teacher.npy"
        --backbone-weights "${REPO_ROOT}/tmp/dlshogi-model/model_resnet10_swish-072"
        --nnue-checkpoint "${REPO_ROOT}/logs/halfkp_v1/checkpoints/83000.ckpt"
        --max-roots 384 --root-batch-size 64
        --output "${REPO_ROOT}/results/search_utility_pilot_${condition}_curve.json"
    )
    for epoch in {0..9}; do
        command+=(--candidate "${REPO_ROOT}/logs/search_utility_pilot_${condition}_from_m0/checkpoints/${epoch}.ckpt")
    done
    "${command[@]}" > "/tmp/search_utility_pilot_${condition}_eval.log" 2>&1
done
