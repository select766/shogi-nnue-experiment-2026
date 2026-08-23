#!/usr/bin/env bash
# Continue the task control and weakest mixed condition to three total epochs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${REPO_ROOT}/tmp/root_grouped_router_pilot"
CACHE_ROOT="${DATA_ROOT}/teachers"

common_args=(
    --feature-set HalfKP --n-experts 8 --gate-transform softmax
    --adapter-hidden 128 --adapter-noise-scale 0.0
    --batch-size 64 --train-shuffle-buffer-size 32
    --epoch-size 99840 --max-val-positions 9728
    --freeze-experts --lr-nnue 0.001 --lr-adapter 0.003
    --lambda 1.0 --label-smoothing-eps 0.001 --score-scaling 361
    --num-batches-warmup 100 --newbob-decay 1.0
    --num-epochs-to-adjust-lr 20 --min-newbob-scale 1e-5
    --momentum 0.9 --network-save-period 1 --max-epochs 2
    --gpus 1 --seed 42
)

bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
    --run-name root_grouped_continue2_task_from_short1 \
    --train-dir "${DATA_ROOT}/train" \
    --val-dir "${DATA_ROOT}/val_disjoint_b" \
    --root-grouped -- "${common_args[@]}" \
    --router-teacher-mode none --lambda-router 0 --task-loss-weight 1 \
    --load-weights-only "${REPO_ROOT}/logs/root_grouped_short1_task_control_from510/lightning_logs/version_0/final.ckpt"

bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
    --run-name root_grouped_continue2_g025_from_short1 \
    --train-dir "${DATA_ROOT}/train" \
    --val-dir "${DATA_ROOT}/val_disjoint_b" \
    --root-grouped -- "${common_args[@]}" \
    --train-teacher-cache "${CACHE_ROOT}/train.npy" \
    --val-teacher-cache "${CACHE_ROOT}/val_b.npy" \
    --router-teacher-mode cached --lambda-router 0.009580039 \
    --task-loss-weight 1 \
    --load-weights-only "${REPO_ROOT}/logs/root_grouped_short1_teacher_g025_from510/lightning_logs/version_0/final.ckpt"
