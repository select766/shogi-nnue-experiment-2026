#!/usr/bin/env bash
# Confirm a valB-positive candidate once on unused valA.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${REPO_ROOT}/tmp/proxy_gap_v2"
FEATURES="${REPO_ROOT}/tmp/root_search_stats_20260827"
STATIC="${REPO_ROOT}/tmp/root_features_20260826"
CONTROL="${REPO_ROOT}/logs/root_features_combined_from_m0/checkpoints/0.ckpt"

"${REPO_ROOT}/scripts/gpu_python.sh" -u -m train_nnue.compare_root_grouped_checkpoints \
    --control "$CONTROL" \
    --candidate "${REPO_ROOT}/logs/root_search_stats_from_m0/checkpoints/0.ckpt" \
    --data "${DATA}/valA" \
    --control-root-features "${STATIC}/combined_valA.npy" \
    --root-features "${FEATURES}/valA.npy" \
    --backbone-weights "${REPO_ROOT}/tmp/dlshogi-model/model_resnet10_swish-072" \
    --nnue-checkpoint "${REPO_ROOT}/logs/halfkp_v1/checkpoints/83000.ckpt" \
    --max-roots 19968 --root-batch-size 256 \
    --output "${REPO_ROOT}/results/root_search_stats_valA.json" \
    > /tmp/root_search_stats_valA.log 2>&1
