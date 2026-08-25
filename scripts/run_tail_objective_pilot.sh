#!/usr/bin/env bash
# Matched mean/CVaR adapter-only training from M0 on actual search leaves.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INITIAL="${REPO_ROOT}/logs/proxy_gap_actual500k_curve_from510/checkpoints/4.ckpt"
DATA="${REPO_ROOT}/tmp/proxy_gap_v2"

run_condition() {
    local name="$1"
    local mode="$2"
    bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
        --run-name "tail_objective_${name}_from_m0" \
        --train-dir "${DATA}/train" \
        --val-dir "${DATA}/valB" \
        --root-grouped -- \
        --feature-set HalfKP \
        --n-experts 8 \
        --adapter-hidden 128 \
        --adapter-noise-scale 0.0 \
        --batch-size 256 \
        --train-shuffle-buffer-size 64 \
        --epoch-size 99840 \
        --max-val-positions 19968 \
        --lr-nnue 0.001 \
        --lr-adapter 0.01 \
        --lambda 1.0 \
        --label-smoothing-eps 0.001 \
        --score-scaling 361 \
        --num-batches-warmup 100 \
        --newbob-decay 1.0 \
        --num-epochs-to-adjust-lr 20 \
        --min-newbob-scale 1e-5 \
        --momentum 0.9 \
        --network-save-period 1 \
        --max-epochs 1 \
        --gpus 1 \
        --seed 42 \
        --freeze-experts \
        --group-loss-mode "$mode" \
        --group-cvar-fraction 0.25 \
        --group-cvar-weight 0.5 \
        --load-weights-only "$INITIAL"
}

run_condition mean mean
run_condition cvar cvar
run_condition mixed mixed
