#!/usr/bin/env bash
# Adapter-only five-epoch curve on the ~500k-root on-policy dataset.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
    --run-name objective_onpolicy_scale_curve_from510 \
    --train-dir "${REPO_ROOT}/tmp/objective_onpolicy_v1/train_onpolicy_scale" \
    --val-dir "${REPO_ROOT}/tmp/objective_onpolicy_v1/val_onpolicy_train8" \
    --root-grouped -- \
    --feature-set HalfKP \
    --n-experts 8 \
    --adapter-hidden 128 \
    --adapter-noise-scale 0.0 \
    --batch-size 256 \
    --train-shuffle-buffer-size 64 \
    --epoch-size 499712 \
    --max-val-positions 17920 \
    --lr-nnue 0.001 \
    --lr-adapter 0.01 \
    --lambda 1.0 \
    --label-smoothing-eps 0.001 \
    --score-scaling 361 \
    --num-batches-warmup 100 \
    --newbob-decay 0.5 \
    --num-epochs-to-adjust-lr 1 \
    --min-newbob-scale 0.03125 \
    --momentum 0.9 \
    --network-save-period 1 \
    --max-epochs 5 \
    --gpus 1 \
    --seed 42 \
    --freeze-experts \
    --load-weights-only \
        "${REPO_ROOT}/logs/expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/510.ckpt"
