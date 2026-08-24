#!/usr/bin/env bash
# Four-condition adapter-only pilot: policy distribution x teacher objective.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INITIAL="${REPO_ROOT}/logs/expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/510.ckpt"
DATA="${REPO_ROOT}/tmp/objective_onpolicy_v1"

run_condition() {
    local name="$1"
    local train_dir="$2"
    local val_dir="$3"
    bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
        --run-name "objective_onpolicy_pilot_${name}_from510" \
        --train-dir "$train_dir" \
        --val-dir "$val_dir" \
        --root-grouped -- \
        --feature-set HalfKP \
        --n-experts 8 \
        --adapter-hidden 128 \
        --adapter-noise-scale 0.0 \
        --batch-size 256 \
        --train-shuffle-buffer-size 64 \
        --epoch-size 99840 \
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
        --max-epochs 1 \
        --gpus 1 \
        --seed 42 \
        --freeze-experts \
        --load-weights-only "$INITIAL"
}

run_condition c00_baseline_qsearch \
    "${DATA}/train_baseline_qsearch" "${DATA}/val_baseline_qsearch_train8"
run_condition c10_baseline_deep \
    "${DATA}/train_baseline_deep" "${DATA}/val_baseline_deep_train8"
run_condition c01_onpolicy_qsearch \
    "${DATA}/train_onpolicy" "${DATA}/val_onpolicy_train8"
run_condition c11_onpolicy_deep \
    "${DATA}/train_onpolicy_deep" "${DATA}/val_onpolicy_deep_train8"
