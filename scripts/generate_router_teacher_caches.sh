#!/usr/bin/env bash
# Generate checkpoint-510 expert-loss caches aligned with current train/val data.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${REPO_ROOT}/logs/expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/510.ckpt"
OUTPUT_DIR="${REPO_ROOT}/tmp/router_teacher_cache/checkpoint510"
mkdir -p "$OUTPUT_DIR"

generate() {
    local split="$1"
    local positions="$2"
    local output="$3"
    local log_file="$4"
    "${REPO_ROOT}/scripts/gpu_python.sh" -u \
        -m train_nnue.generate_router_teacher_cache \
        --checkpoint "$CHECKPOINT" \
        --bin-dir "${REPO_ROOT}/dataset/split_v1_paired_uniform_50/${split}" \
        --backbone-weights "${REPO_ROOT}/tmp/dlshogi-model/model_resnet10_swish-072" \
        --nnue-checkpoint "${REPO_ROOT}/logs/halfkp_v1/checkpoints/83000.ckpt" \
        --feature-set HalfKP \
        --batch-size 256 \
        --max-positions "$positions" \
        --output "$output" \
        > "$log_file" 2>&1
}

generate train 1000000 "$OUTPUT_DIR/train_losses.npy" /tmp/router_teacher_cache_train.log
generate val1 20480 "$OUTPUT_DIR/val_losses.npy" /tmp/router_teacher_cache_val.log

echo "Teacher caches written to: $OUTPUT_DIR"
