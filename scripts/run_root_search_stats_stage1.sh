#!/usr/bin/env bash
# Train static+search features and compare against matched static-only control on valB.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${REPO_ROOT}/tmp/proxy_gap_v2"
FEATURES="${REPO_ROOT}/tmp/root_search_stats_20260827"
STATIC="${REPO_ROOT}/tmp/root_features_20260826"
INITIAL="${REPO_ROOT}/logs/proxy_gap_actual500k_curve_from510/checkpoints/4.ckpt"
CONTROL="${REPO_ROOT}/logs/root_features_combined_from_m0/checkpoints/0.ckpt"
EXPANDED="${FEATURES}/m0_static_search.ckpt"

"${REPO_ROOT}/scripts/nnue_python.sh" "${REPO_ROOT}/scripts/expand_adapter_input.py" \
    --input "$INITIAL" --output "$EXPANDED" --auxiliary-dim 13

bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
    --run-name root_search_stats_from_m0 \
    --train-dir "${DATA}/train" --val-dir "${DATA}/valB" --root-grouped -- \
    --feature-set HalfKP --n-experts 8 --adapter-hidden 128 \
    --adapter-noise-scale 0.0 --root-feature-dim 13 \
    --train-root-features "${FEATURES}/train.npy" \
    --val-root-features "${FEATURES}/valB.npy" \
    --batch-size 256 --train-shuffle-buffer-size 64 \
    --epoch-size 99840 --max-val-positions 19968 \
    --lr-nnue 0.001 --lr-adapter 0.01 --lambda 1.0 \
    --label-smoothing-eps 0.001 --score-scaling 361 \
    --num-batches-warmup 100 --newbob-decay 1.0 \
    --num-epochs-to-adjust-lr 20 --min-newbob-scale 1e-5 \
    --momentum 0.9 --network-save-period 1 --max-epochs 1 \
    --gpus 1 --seed 42 --freeze-experts --group-loss-mode mean \
    --load-weights-only "$EXPANDED"

"${REPO_ROOT}/scripts/gpu_python.sh" -u -m train_nnue.compare_root_grouped_checkpoints \
    --control "$CONTROL" \
    --candidate "${REPO_ROOT}/logs/root_search_stats_from_m0/checkpoints/0.ckpt" \
    --data "${DATA}/valB" \
    --control-root-features "${STATIC}/combined_valB.npy" \
    --root-features "${FEATURES}/valB.npy" \
    --backbone-weights "${REPO_ROOT}/tmp/dlshogi-model/model_resnet10_swish-072" \
    --nnue-checkpoint "${REPO_ROOT}/logs/halfkp_v1/checkpoints/83000.ckpt" \
    --max-roots 19968 --root-batch-size 256 \
    --output "${REPO_ROOT}/results/root_search_stats_valB.json" \
    > /tmp/root_search_stats_valB.log 2>&1
