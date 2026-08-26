#!/usr/bin/env bash
# Build feature caches, train matched ablations, and compare on selection valB.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${REPO_ROOT}/tmp/proxy_gap_v2"
FEATURE_ROOT="${REPO_ROOT}/tmp/root_features_20260826"
INITIAL="${REPO_ROOT}/logs/proxy_gap_actual500k_curve_from510/checkpoints/4.ckpt"
CONTROL="${REPO_ROOT}/logs/tail_objective_mean_from_m0/checkpoints/0.ckpt"
mkdir -p "$FEATURE_ROOT"

for specification in ply:1 phase:6 combined:7; do
    condition="${specification%%:*}"
    dimension="${specification##*:}"
    for split in train valA valB; do
        case "$split" in
            # A 64-batch shuffle buffer reads 16,384 roots ahead of the
            # 99,840-root epoch before yielding its final batch.
            train) roots=116224 ;;
            *) roots=19968 ;;
        esac
        "${REPO_ROOT}/scripts/project_python.sh" \
            "${REPO_ROOT}/scripts/build_root_feature_cache.py" \
            --roots-file "${DATA}/${split}/roots.bin" \
            --feature-set "$condition" \
            --max-roots "$roots" \
            --output "${FEATURE_ROOT}/${condition}_${split}.npy"
    done
    expanded="${FEATURE_ROOT}/m0_${condition}.ckpt"
    "${REPO_ROOT}/scripts/nnue_python.sh" \
        "${REPO_ROOT}/scripts/expand_adapter_input.py" \
        --input "$INITIAL" --output "$expanded" --auxiliary-dim "$dimension"
    bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
        --run-name "root_features_${condition}_from_m0" \
        --train-dir "${DATA}/train" --val-dir "${DATA}/valB" --root-grouped -- \
        --feature-set HalfKP --n-experts 8 --adapter-hidden 128 \
        --adapter-noise-scale 0.0 --root-feature-dim "$dimension" \
        --train-root-features "${FEATURE_ROOT}/${condition}_train.npy" \
        --val-root-features "${FEATURE_ROOT}/${condition}_valB.npy" \
        --batch-size 256 --train-shuffle-buffer-size 64 \
        --epoch-size 99840 --max-val-positions 19968 \
        --lr-nnue 0.001 --lr-adapter 0.01 --lambda 1.0 \
        --label-smoothing-eps 0.001 --score-scaling 361 \
        --num-batches-warmup 100 --newbob-decay 1.0 \
        --num-epochs-to-adjust-lr 20 --min-newbob-scale 1e-5 \
        --momentum 0.9 --network-save-period 1 --max-epochs 1 \
        --gpus 1 --seed 42 --freeze-experts --group-loss-mode mean \
        --load-weights-only "$expanded"
    "${REPO_ROOT}/scripts/gpu_python.sh" -u -m train_nnue.compare_root_grouped_checkpoints \
        --control "$CONTROL" \
        --candidate "${REPO_ROOT}/logs/root_features_${condition}_from_m0/checkpoints/0.ckpt" \
        --data "${DATA}/valB" \
        --root-features "${FEATURE_ROOT}/${condition}_valB.npy" \
        --backbone-weights "${REPO_ROOT}/tmp/dlshogi-model/model_resnet10_swish-072" \
        --nnue-checkpoint "${REPO_ROOT}/logs/halfkp_v1/checkpoints/83000.ckpt" \
        --max-roots 19968 --root-batch-size 256 \
        --output "${REPO_ROOT}/results/root_features_${condition}_valB.json" \
        > "/tmp/root_features_${condition}_valB.log" 2>&1
done
