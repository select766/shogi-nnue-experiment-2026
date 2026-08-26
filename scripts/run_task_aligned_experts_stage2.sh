#!/usr/bin/env bash
# Run independent valA blended and phase-specialization diagnostics.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${REPO_ROOT}/tmp/proxy_gap_v2"
ROLE_ROOT="${REPO_ROOT}/tmp/task_aligned_experts_20260826"
BACKBONE="${REPO_ROOT}/tmp/dlshogi-model/model_resnet10_swish-072"
NNUE="${REPO_ROOT}/logs/halfkp_v1/checkpoints/83000.ckpt"
M0="${REPO_ROOT}/logs/proxy_gap_actual500k_curve_from510/checkpoints/4.ckpt"
CONTROL="${REPO_ROOT}/logs/task_aligned_experts_control_from_m0/checkpoints/0.ckpt"
CANDIDATES=(
    "${REPO_ROOT}/logs/task_aligned_experts_role002_from_m0/checkpoints/0.ckpt"
    "${REPO_ROOT}/logs/task_aligned_experts_role005_from_m0/checkpoints/0.ckpt"
)

"${REPO_ROOT}/scripts/gpu_python.sh" -u -m train_nnue.compare_root_grouped_checkpoints \
    --control "$CONTROL" \
    --candidate "${CANDIDATES[0]}" --candidate "${CANDIDATES[1]}" \
    --data "${DATA}/valA" --backbone-weights "$BACKBONE" \
    --nnue-checkpoint "$NNUE" --max-roots 19968 --root-batch-size 256 \
    --output "${REPO_ROOT}/results/task_aligned_experts_valA.json" \
    > /tmp/task_aligned_experts_valA.log 2>&1

"${REPO_ROOT}/scripts/gpu_python.sh" -u -m train_nnue.compare_phase_specialization \
    --control "$CONTROL" --reference "$M0" \
    --candidate "${CANDIDATES[0]}" --candidate "${CANDIDATES[1]}" \
    --data "${DATA}/valA" --role-cache "${ROLE_ROOT}/valA_roles.npy" \
    --backbone-weights "$BACKBONE" --nnue-checkpoint "$NNUE" \
    --max-roots 19968 --root-batch-size 128 \
    --output "${REPO_ROOT}/results/task_aligned_experts_specialization_valA.json" \
    > /tmp/task_aligned_experts_specialization_valA.log 2>&1
