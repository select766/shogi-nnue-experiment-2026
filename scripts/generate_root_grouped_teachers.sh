#!/usr/bin/env bash
# Generate shared-gate teachers for the root-grouped pilot.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${REPO_ROOT}/tmp/root_grouped_router_pilot"
CACHE_ROOT="${DATA_ROOT}/teachers"
CHECKPOINT="${REPO_ROOT}/logs/expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/510.ckpt"
mkdir -p "$CACHE_ROOT"

generate() {
    local split="$1"
    local roots="$2"
    local output="$3"
    local log="$4"
    "${REPO_ROOT}/scripts/gpu_python.sh" -u \
        -m train_nnue.generate_root_grouped_teacher_cache \
        --checkpoint "$CHECKPOINT" \
        --data "${DATA_ROOT}/${split}" \
        --backbone-weights "${REPO_ROOT}/tmp/dlshogi-model/model_resnet10_swish-072" \
        --nnue-checkpoint "${REPO_ROOT}/logs/halfkp_v1/checkpoints/83000.ckpt" \
        --max-roots "$roots" --root-batch-size 256 \
        --steps 10 --lr 0.1 --prior-kl 0.01 \
        --output "${CACHE_ROOT}/${output}" > "$log" 2>&1
}

generate val_disjoint_a 9728 val_a.npy /tmp/root_grouped_teacher_val_a.log
generate val_disjoint_b 9728 val_b.npy /tmp/root_grouped_teacher_val_b.log
generate train 100000 train.npy /tmp/root_grouped_teacher_train.log

"${REPO_ROOT}/scripts/gpu_python.sh" -u \
    -m train_nnue.evaluate_root_grouped_teachers \
    --checkpoint "$CHECKPOINT" \
    --data-b "${DATA_ROOT}/val_disjoint_b" \
    --teacher-a "${CACHE_ROOT}/val_a.npy" \
    --teacher-b "${CACHE_ROOT}/val_b.npy" \
    --backbone-weights "${REPO_ROOT}/tmp/dlshogi-model/model_resnet10_swish-072" \
    --nnue-checkpoint "${REPO_ROOT}/logs/halfkp_v1/checkpoints/83000.ckpt" \
    --max-roots 9728 --root-batch-size 256 \
    --output "${REPO_ROOT}/results/root_grouped_teacher_stability.json" \
    > /tmp/root_grouped_teacher_evaluation.log 2>&1
