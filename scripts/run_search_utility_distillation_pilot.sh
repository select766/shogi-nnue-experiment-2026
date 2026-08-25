#!/usr/bin/env bash
# Two pre-registered adapter-only search-utility distillation conditions.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${REPO_ROOT}/tmp/search_utility_v1"
INITIAL="${REPO_ROOT}/logs/proxy_gap_actual500k_curve_from510/checkpoints/4.ckpt"

common=(
    --feature-set HalfKP --n-experts 8 --gate-transform softmax
    --adapter-hidden 128 --adapter-noise-scale 0.0
    --batch-size 64 --train-shuffle-buffer-size 16
    --epoch-size 1600 --max-val-positions 384
    --train-teacher-cache "${DATA}/train_teacher.npy"
    --val-teacher-cache "${DATA}/val_teacher.npy"
    --router-teacher-mode cached --freeze-experts
    --lr-nnue 0.001 --lr-adapter 0.003
    --lambda 1.0 --label-smoothing-eps 0.001 --score-scaling 361
    --num-batches-warmup 10 --newbob-decay 1.0
    --num-epochs-to-adjust-lr 20 --min-newbob-scale 1e-5
    --momentum 0.9 --network-save-period 1 --max-epochs 10
    --gpus 1 --seed 42 --load-weights-only "$INITIAL"
)

bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
    --run-name search_utility_pilot_router_only_from_m0 \
    --train-dir "${DATA}/train" --val-dir "${DATA}/val" --root-grouped -- \
    "${common[@]}" --task-loss-weight 0 --lambda-router 1

bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
    --run-name search_utility_pilot_combined_from_m0 \
    --train-dir "${DATA}/train" --val-dir "${DATA}/val" --root-grouped -- \
    "${common[@]}" --task-loss-weight 1 --lambda-router 0.01
