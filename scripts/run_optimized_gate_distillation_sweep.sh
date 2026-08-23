#!/usr/bin/env bash
# Distill per-position gates optimized against actual blended NNUE loss.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INITIAL_CHECKPOINT="${REPO_ROOT}/logs/expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/510.ckpt"
CACHE_DIR="${REPO_ROOT}/tmp/router_teacher_cache/checkpoint510"

run_condition() {
    local name="$1"
    local task_weight="$2"
    local router_weight="$3"
    bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
        --run-name "router_gate_teacher_short3_${name}_from510" \
        -- --feature-set HalfKP --n-experts 8 --gate-transform softmax \
        --adapter-hidden 128 --adapter-noise-scale 0.0 \
        --batch-size 256 --train-shuffle-buffer-size 64 \
        --epoch-size 100000 --max-val-positions 10000 \
        --train-teacher-cache "$CACHE_DIR/train_optimized_gate.npy" \
        --val-teacher-cache "$CACHE_DIR/val_optimized_gate.npy" \
        --router-teacher-mode cached --lambda-router "$router_weight" \
        --task-loss-weight "$task_weight" --freeze-experts \
        --lr-nnue 0.001 --lr-adapter 0.003 \
        --lambda 1.0 --label-smoothing-eps 0.001 --score-scaling 361 \
        --num-batches-warmup 100 --newbob-decay 1.0 \
        --num-epochs-to-adjust-lr 20 --min-newbob-scale 1e-5 \
        --momentum 0.9 --network-save-period 1 --max-epochs 3 \
        --gpus 1 --seed 42 --load-weights-only "$INITIAL_CHECKPOINT"
}

run_condition pure 0 1
run_condition g025 1 0.028802019
run_condition g100 1 0.115208074
