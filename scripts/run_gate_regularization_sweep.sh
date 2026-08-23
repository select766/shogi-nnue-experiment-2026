#!/usr/bin/env bash
# Short entropy/balance fine-tuning sweep initialized from checkpoint 510.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INITIAL_CHECKPOINT="${REPO_ROOT}/logs/expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/510.ckpt"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/run_gate_regularization_sweep.sh [--dry-run]" >&2
    exit 2
fi
if [[ ! -f "$INITIAL_CHECKPOINT" ]]; then
    echo "ERROR: missing initial checkpoint: $INITIAL_CHECKPOINT" >&2
    exit 1
fi

SPARSE_VALUES=(0 1e-4 1e-3 1e-2)
BALANCE_VALUES=(1e-3 1e-2)

for lambda_sparse in "${SPARSE_VALUES[@]}"; do
    for lambda_balance in "${BALANCE_VALUES[@]}"; do
        sparse_name="${lambda_sparse//./p}"
        sparse_name="${sparse_name//-/m}"
        balance_name="${lambda_balance//./p}"
        balance_name="${balance_name//-/m}"
        run_name="gate_reg_short3_s${sparse_name}_b${balance_name}_from510"
        wrapper_args=(--run-name "$run_name")
        if [[ "$DRY_RUN" -eq 1 ]]; then
            wrapper_args+=(--dry-run)
        fi
        bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
            "${wrapper_args[@]}" \
            -- \
            --feature-set HalfKP \
            --n-experts 8 \
            --adapter-hidden 128 \
            --adapter-noise-scale 0.0 \
            --batch-size 256 \
            --train-shuffle-buffer-size 64 \
            --epoch-size 100000 \
            --max-val-positions 10000 \
            --lr-nnue 0.001 \
            --lr-adapter 0.01 \
            --lambda 1.0 \
            --lambda-sparse "$lambda_sparse" \
            --lambda-balance "$lambda_balance" \
            --label-smoothing-eps 0.001 \
            --score-scaling 361 \
            --num-batches-warmup 100 \
            --newbob-decay 1.0 \
            --num-epochs-to-adjust-lr 20 \
            --min-newbob-scale 1e-5 \
            --momentum 0.9 \
            --network-save-period 10 \
            --max-epochs 3 \
            --gpus 1 \
            --seed 42 \
            --load-weights-only "$INITIAL_CHECKPOINT"
    done
done
