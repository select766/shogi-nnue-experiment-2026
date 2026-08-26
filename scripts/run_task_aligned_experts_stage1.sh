#!/usr/bin/env bash
# Train phase-role expert candidates and compare blended loss on selection valB.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${REPO_ROOT}/tmp/proxy_gap_v2"
ROLE_ROOT="${REPO_ROOT}/tmp/task_aligned_experts_20260826"
INITIAL="${REPO_ROOT}/logs/proxy_gap_actual500k_curve_from510/checkpoints/4.ckpt"
mkdir -p "$ROLE_ROOT"

for split in train valA valB; do
    case "$split" in
        # A 64-batch shuffle buffer at root batch 64 reads 4,096 roots ahead.
        train) roots=103936 ;;
        *) roots=19968 ;;
    esac
    "${REPO_ROOT}/scripts/project_python.sh" \
        "${REPO_ROOT}/scripts/build_phase_role_cache.py" \
        --roots-file "${DATA}/${split}/roots.bin" --max-roots "$roots" \
        --output "${ROLE_ROOT}/${split}_roles.npy"
done

train_condition() {
    local name="$1"
    local role_weight="$2"
    local role_arguments=()
    if [[ "$role_weight" != "0" ]]; then
        role_arguments=(
            --lambda-role-expert "$role_weight"
            --train-expert-roles "${ROLE_ROOT}/train_roles.npy"
            --val-expert-roles "${ROLE_ROOT}/valB_roles.npy"
        )
    fi
    bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
        --run-name "task_aligned_experts_${name}_from_m0" \
        --train-dir "${DATA}/train" --val-dir "${DATA}/valB" --root-grouped -- \
        --feature-set HalfKP --n-experts 8 --adapter-hidden 128 \
        --adapter-noise-scale 0.0 --batch-size 64 --train-shuffle-buffer-size 64 \
        --epoch-size 99840 --max-val-positions 19968 \
        --lr-nnue 0.001 --lr-adapter 0.001 --lambda 1.0 \
        --label-smoothing-eps 0.001 --score-scaling 361 \
        --num-batches-warmup 100 --newbob-decay 1.0 \
        --num-epochs-to-adjust-lr 20 --min-newbob-scale 1e-5 \
        --momentum 0.9 --network-save-period 1 --max-epochs 1 \
        --gpus 1 --seed 42 --freeze-adapter --group-loss-mode mean \
        --load-weights-only "$INITIAL" "${role_arguments[@]}"
}

train_condition control 0
train_condition role002 0.02
train_condition role005 0.05

"${REPO_ROOT}/scripts/gpu_python.sh" -u -m train_nnue.compare_root_grouped_checkpoints \
    --control "${REPO_ROOT}/logs/task_aligned_experts_control_from_m0/checkpoints/0.ckpt" \
    --candidate "${REPO_ROOT}/logs/task_aligned_experts_role002_from_m0/checkpoints/0.ckpt" \
    --candidate "${REPO_ROOT}/logs/task_aligned_experts_role005_from_m0/checkpoints/0.ckpt" \
    --data "${DATA}/valB" \
    --backbone-weights "${REPO_ROOT}/tmp/dlshogi-model/model_resnet10_swish-072" \
    --nnue-checkpoint "${REPO_ROOT}/logs/halfkp_v1/checkpoints/83000.ckpt" \
    --max-roots 19968 --root-batch-size 256 \
    --output "${REPO_ROOT}/results/task_aligned_experts_valB.json" \
    > /tmp/task_aligned_experts_valB.log 2>&1
