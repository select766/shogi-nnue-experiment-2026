#!/usr/bin/env bash
# Train function-preserving widened root adapters on actual search leaves.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${REPO_ROOT}/tmp/proxy_gap_v2"
M0="${REPO_ROOT}/logs/proxy_gap_actual500k_curve_from510/checkpoints/4.ckpt"

for width in 256 512; do
    "${REPO_ROOT}/scripts/nnue_python.sh" \
        "${REPO_ROOT}/scripts/widen_adapter_checkpoint.py" \
        --input "$M0" \
        --output "${REPO_ROOT}/tmp/root_representation_hidden${width}_init.ckpt" \
        --hidden-dim "$width"
done

run_width() {
    local width="$1"
    local initial="${REPO_ROOT}/tmp/root_representation_hidden${width}_init.ckpt"
    bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
        --run-name "root_representation_hidden${width}_from_m0" \
        --train-dir "${DATA}/train" \
        --val-dir "${DATA}/valB" \
        --root-grouped -- \
        --feature-set HalfKP \
        --n-experts 8 \
        --adapter-hidden "$width" \
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
        --load-weights-only "$initial"
}

run_width 256
run_width 512
