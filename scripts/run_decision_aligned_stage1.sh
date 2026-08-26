#!/usr/bin/env bash
# Train matched task-only and hard utility-direction router conditions.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${REPO_ROOT}/tmp/search_utility_v1"
OUTPUT_ROOT="${REPO_ROOT}/tmp/decision_aligned_20260826"
DETAILS="${DATA}/pilot/details.jsonl"
INITIAL="${REPO_ROOT}/logs/proxy_gap_actual500k_curve_from510/checkpoints/4.ckpt"
mkdir -p "$OUTPUT_ROOT"

"${REPO_ROOT}/scripts/project_python.sh" "${REPO_ROOT}/scripts/build_decision_teacher_cache.py" \
    --details "$DETAILS" --start-root 0 --num-roots 1600 \
    --output "${OUTPUT_ROOT}/train_teacher.npy"
"${REPO_ROOT}/scripts/project_python.sh" "${REPO_ROOT}/scripts/build_decision_teacher_cache.py" \
    --details "$DETAILS" --start-root 1600 --num-roots 400 \
    --output "${OUTPUT_ROOT}/val_teacher.npy"

train_condition() {
    local name="$1"
    local decision_weight="$2"
    bash "${REPO_ROOT}/scripts/train_expert_blending.sh" \
        --run-name "decision_aligned_${name}_from_m0" \
        --train-dir "${DATA}/train" --val-dir "${DATA}/val" --root-grouped -- \
        --feature-set HalfKP --n-experts 8 --adapter-hidden 128 \
        --adapter-noise-scale 0.0 --batch-size 64 --train-shuffle-buffer-size 16 \
        --epoch-size 1600 --max-val-positions 384 \
        --train-teacher-cache "${OUTPUT_ROOT}/train_teacher.npy" \
        --val-teacher-cache "${OUTPUT_ROOT}/val_teacher.npy" \
        --router-teacher-mode cached --freeze-experts \
        --lr-nnue 0.001 --lr-adapter 0.003 --lambda 1.0 \
        --label-smoothing-eps 0.001 --score-scaling 361 \
        --num-batches-warmup 10 --newbob-decay 1.0 \
        --num-epochs-to-adjust-lr 20 --min-newbob-scale 1e-5 \
        --momentum 0.9 --network-save-period 1 --max-epochs 10 \
        --gpus 1 --seed 42 --task-loss-weight 1 \
        --lambda-router "$decision_weight" --load-weights-only "$INITIAL"
    "${REPO_ROOT}/scripts/nnue_python.sh" "${REPO_ROOT}/scripts/summarize_training_curve.py" \
        --log-dir "${REPO_ROOT}/logs/decision_aligned_${name}_from_m0/lightning_logs/version_0" \
        --output "${REPO_ROOT}/results/decision_aligned_${name}_curve.json"
}

train_condition control 0
train_condition weight001 0.01
train_condition weight005 0.05

"${REPO_ROOT}/scripts/project_python.sh" "${REPO_ROOT}/scripts/select_decision_aligned_checkpoint.py" \
    --control-curve "${REPO_ROOT}/results/decision_aligned_control_curve.json" \
    --candidate weight001 "${REPO_ROOT}/results/decision_aligned_weight001_curve.json" \
        "${REPO_ROOT}/logs/decision_aligned_weight001_from_m0/checkpoints" \
    --candidate weight005 "${REPO_ROOT}/results/decision_aligned_weight005_curve.json" \
        "${REPO_ROOT}/logs/decision_aligned_weight005_from_m0/checkpoints" \
    --output "${REPO_ROOT}/results/decision_aligned_selection.json"
