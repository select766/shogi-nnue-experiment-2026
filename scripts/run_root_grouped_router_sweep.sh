#!/usr/bin/env bash
# Root-grouped task-only and shared-teacher distillation from checkpoint 510.
set -euo pipefail

LAMBDA_025="${1:?Usage: $0 LAMBDA_FOR_0.25X LAMBDA_FOR_1.00X}"
LAMBDA_100="${2:?Usage: $0 LAMBDA_FOR_0.25X LAMBDA_FOR_1.00X}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${REPO_ROOT}/tmp/root_grouped_router_pilot"
CACHE_ROOT="${DATA_ROOT}/teachers"
INITIAL="${REPO_ROOT}/logs/expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/510.ckpt"

common_args=(
    --feature-set HalfKP --n-experts 8 --gate-transform softmax
    --adapter-hidden 128 --adapter-noise-scale 0.0
    --batch-size 64 --train-shuffle-buffer-size 32
    --epoch-size 99840 --max-val-positions 9728
    --freeze-experts --lr-nnue 0.001 --lr-adapter 0.003
    --lambda 1.0 --label-smoothing-eps 0.001 --score-scaling 361
    --num-batches-warmup 100 --newbob-decay 1.0
    --num-epochs-to-adjust-lr 20 --min-newbob-scale 1e-5
    --momentum 0.9 --network-save-period 1 --max-epochs 1
    --gpus 1 --seed 42 --load-weights-only "$INITIAL"
)

run_control() {
    bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
        --run-name root_grouped_short1_task_control_from510 \
        --train-dir "${DATA_ROOT}/train" \
        --val-dir "${DATA_ROOT}/val_disjoint_b" \
        --root-grouped -- "${common_args[@]}" \
        --router-teacher-mode none --lambda-router 0 --task-loss-weight 1
}

run_distill() {
    local name="$1"
    local task_weight="$2"
    local router_weight="$3"
    bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
        --run-name "root_grouped_short1_${name}_from510" \
        --train-dir "${DATA_ROOT}/train" \
        --val-dir "${DATA_ROOT}/val_disjoint_b" \
        --root-grouped -- "${common_args[@]}" \
        --train-teacher-cache "${CACHE_ROOT}/train.npy" \
        --val-teacher-cache "${CACHE_ROOT}/val_b.npy" \
        --router-teacher-mode cached \
        --lambda-router "$router_weight" --task-loss-weight "$task_weight"
}

run_control
run_distill teacher_pure 0 1
run_distill teacher_g025 1 "$LAMBDA_025"
run_distill teacher_g100 1 "$LAMBDA_100"
