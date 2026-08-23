#!/usr/bin/env bash
# Task + soft-router distillation with gradient-calibrated coefficients.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INITIAL_CHECKPOINT="${REPO_ROOT}/logs/expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/510.ckpt"
CACHE_DIR="${REPO_ROOT}/tmp/router_teacher_cache/checkpoint510"

run_condition() {
    local name="$1"
    local temperature="$2"
    local router_weight="$3"
    bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
        --run-name "router_distill_combined_short3_${name}_from510" \
        -- \
        --feature-set HalfKP --n-experts 8 --gate-transform softmax \
        --adapter-hidden 128 --adapter-noise-scale 0.0 \
        --batch-size 256 --train-shuffle-buffer-size 64 \
        --epoch-size 100000 --max-val-positions 10000 \
        --train-teacher-cache "$CACHE_DIR/train_losses.npy" \
        --val-teacher-cache "$CACHE_DIR/val_losses.npy" \
        --router-teacher-mode soft \
        --router-teacher-temperature "$temperature" \
        --lambda-router "$router_weight" --task-loss-weight 1 \
        --freeze-experts --lr-nnue 0.001 --lr-adapter 0.003 \
        --lambda 1.0 --label-smoothing-eps 0.001 --score-scaling 361 \
        --num-batches-warmup 100 --newbob-decay 1.0 \
        --num-epochs-to-adjust-lr 20 --min-newbob-scale 1e-5 \
        --momentum 0.9 --network-save-period 1 --max-epochs 3 \
        --gpus 1 --seed 42 --load-weights-only "$INITIAL_CHECKPOINT"
}

run_condition soft005_g025 0.005 0.004507073
run_condition soft005_g100 0.005 0.018028291
run_condition soft020_g025 0.020 0.004242436
run_condition soft020_g100 0.020 0.016969742
