#!/usr/bin/env bash
# Cross-evaluate the four objective/on-policy pilot checkpoints.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${REPO_ROOT}/tmp/objective_onpolicy_v1"
BACKBONE="${REPO_ROOT}/tmp/dlshogi-model/model_resnet10_swish-072"
NNUE="${REPO_ROOT}/logs/halfkp_v1/checkpoints/83000.ckpt"
CONTROL="${REPO_ROOT}/logs/expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/510.ckpt"
M0="${REPO_ROOT}/logs/proxy_gap_actual500k_curve_from510/checkpoints/4.ckpt"
C00="${REPO_ROOT}/logs/objective_onpolicy_pilot_c00_baseline_qsearch_from510/checkpoints/0.ckpt"
C10="${REPO_ROOT}/logs/objective_onpolicy_pilot_c10_baseline_deep_from510/checkpoints/0.ckpt"
C01="${REPO_ROOT}/logs/objective_onpolicy_pilot_c01_onpolicy_qsearch_from510/checkpoints/0.ckpt"
C11="${REPO_ROOT}/logs/objective_onpolicy_pilot_c11_onpolicy_deep_from510/checkpoints/0.ckpt"

compare() {
    local control="$1"
    local output="$2"
    local data="$3"
    shift 3
    local command=(
        "${REPO_ROOT}/scripts/gpu_python.sh" -u -m
        train_nnue.compare_root_grouped_checkpoints
        --control "$control"
        --data "$data"
        --backbone-weights "$BACKBONE"
        --nnue-checkpoint "$NNUE"
        --max-roots 17920
        --root-batch-size 256
        --output "${REPO_ROOT}/results/${output}.json"
    )
    local candidate
    for candidate in "$@"; do
        command+=(--candidate "$candidate")
    done
    "${command[@]}" > "/tmp/${output}.log" 2>&1
}

for distribution in \
    val_baseline_qsearch val_baseline_deep val_onpolicy val_onpolicy_deep; do
    compare "$CONTROL" "objective_onpolicy_matrix_${distribution}" \
        "${DATA}/${distribution}" "$M0" "$C00" "$C10" "$C01" "$C11"
done

# Direct paired tests used by the pre-registered continuation rules.
compare "$C00" objective_onpolicy_direct_h2_onpolicy \
    "${DATA}/val_onpolicy" "$C01"
compare "$C00" objective_onpolicy_direct_h2_baseline \
    "${DATA}/val_baseline_qsearch" "$C01"
compare "$C00" objective_onpolicy_direct_h1_baseline_deep \
    "${DATA}/val_baseline_deep" "$C10"
compare "$C00" objective_onpolicy_direct_h1_baseline_qsearch \
    "${DATA}/val_baseline_qsearch" "$C10"
compare "$C01" objective_onpolicy_direct_h1_onpolicy_deep \
    "${DATA}/val_onpolicy_deep" "$C11"
compare "$C01" objective_onpolicy_direct_h1_onpolicy_qsearch \
    "${DATA}/val_onpolicy" "$C11"
