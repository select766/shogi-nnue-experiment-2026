#!/usr/bin/env bash
# Evaluate valB-positive root-feature candidates once on unused valA.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${REPO_ROOT}/tmp/proxy_gap_v2"
FEATURE_ROOT="${REPO_ROOT}/tmp/root_features_20260826"
CONTROL="${REPO_ROOT}/logs/tail_objective_mean_from_m0/checkpoints/0.ckpt"

for condition in ply phase combined; do
    "${REPO_ROOT}/scripts/gpu_python.sh" -u -m train_nnue.compare_root_grouped_checkpoints \
        --control "$CONTROL" \
        --candidate "${REPO_ROOT}/logs/root_features_${condition}_from_m0/checkpoints/0.ckpt" \
        --data "${DATA}/valA" \
        --root-features "${FEATURE_ROOT}/${condition}_valA.npy" \
        --backbone-weights "${REPO_ROOT}/tmp/dlshogi-model/model_resnet10_swish-072" \
        --nnue-checkpoint "${REPO_ROOT}/logs/halfkp_v1/checkpoints/83000.ckpt" \
        --max-roots 19968 --root-batch-size 256 \
        --output "${REPO_ROOT}/results/root_features_${condition}_valA.json" \
        > "/tmp/root_features_${condition}_valA.log" 2>&1
done
