#!/usr/bin/env bash
# Adapter-only router distillation screening from checkpoint 510.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INITIAL_CHECKPOINT="${REPO_ROOT}/logs/expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/510.ckpt"
CACHE_DIR="${REPO_ROOT}/tmp/router_teacher_cache/checkpoint510"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/run_router_distillation_sweep.sh [--dry-run]" >&2
    exit 2
fi
for path in "$INITIAL_CHECKPOINT" "$CACHE_DIR/train_losses.npy" "$CACHE_DIR/val_losses.npy"; do
    [[ -f "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }
done

run_condition() {
    local name="$1"
    local mode="$2"
    local temperature="$3"
    local task_weight="$4"
    local router_weight="$5"
    local wrapper_args=(--run-name "router_distill_short3_${name}_from510")
    if [[ "$DRY_RUN" -eq 1 ]]; then
        wrapper_args+=(--dry-run)
    fi
    bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
        "${wrapper_args[@]}" \
        -- \
        --feature-set HalfKP \
        --n-experts 8 \
        --gate-transform softmax \
        --adapter-hidden 128 \
        --adapter-noise-scale 0.0 \
        --batch-size 256 \
        --train-shuffle-buffer-size 64 \
        --epoch-size 100000 \
        --max-val-positions 10000 \
        --train-teacher-cache "$CACHE_DIR/train_losses.npy" \
        --val-teacher-cache "$CACHE_DIR/val_losses.npy" \
        --router-teacher-mode "$mode" \
        --router-teacher-temperature "$temperature" \
        --lambda-router "$router_weight" \
        --task-loss-weight "$task_weight" \
        --freeze-experts \
        --lr-nnue 0.001 \
        --lr-adapter 0.003 \
        --lambda 1.0 \
        --label-smoothing-eps 0.001 \
        --score-scaling 361 \
        --num-batches-warmup 100 \
        --newbob-decay 1.0 \
        --num-epochs-to-adjust-lr 20 \
        --min-newbob-scale 1e-5 \
        --momentum 0.9 \
        --network-save-period 1 \
        --max-epochs 3 \
        --gpus 1 \
        --seed 42 \
        --load-weights-only "$INITIAL_CHECKPOINT"
}

run_condition task_control none 0.005 1 0
run_condition hard hard 0.005 0 1
run_condition soft005 soft 0.005 0 1
run_condition soft020 soft 0.020 0 1
