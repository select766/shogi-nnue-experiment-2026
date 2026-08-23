#!/usr/bin/env bash
# Generate actual blended-loss optimized gate teachers for router distillation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${REPO_ROOT}/tmp/router_teacher_cache/checkpoint510"
CHECKPOINT="${REPO_ROOT}/logs/expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/510.ckpt"

generate() {
    local split="$1"
    local positions="$2"
    local losses="$3"
    local output="$4"
    local log_file="$5"
    "${REPO_ROOT}/scripts/gpu_python.sh" -u \
        -m train_nnue.generate_optimized_gate_teacher_cache \
        --checkpoint "$CHECKPOINT" \
        --bin-dir "${REPO_ROOT}/dataset/split_v1_paired_uniform_50/${split}" \
        --expert-loss-cache "$losses" \
        --backbone-weights "${REPO_ROOT}/tmp/dlshogi-model/model_resnet10_swish-072" \
        --nnue-checkpoint "${REPO_ROOT}/logs/halfkp_v1/checkpoints/83000.ckpt" \
        --max-positions "$positions" --batch-size 256 \
        --steps 10 --lr 0.1 --prior-kl 0.01 \
        --output "$output" > "$log_file" 2>&1
}

generate train 400000 "$CACHE_DIR/train_losses.npy" \
    "$CACHE_DIR/train_optimized_gate.npy" /tmp/optimized_gate_teacher_train.log
generate val1 20480 "$CACHE_DIR/val_losses.npy" \
    "$CACHE_DIR/val_optimized_gate.npy" /tmp/optimized_gate_teacher_val.log

echo "Optimized gate teacher caches written to: $CACHE_DIR"
