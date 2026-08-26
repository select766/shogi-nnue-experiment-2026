#!/usr/bin/env bash
# Run the fixed-size follow-up match for H-LEAF-LOSS-STRENGTH.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${REPO_ROOT}/tmp/leaf_loss_strength_match"
ALL_OPENINGS="${WORK}/openings_1800.jsonl"
PARALLEL_CHUNKS="${MATCH_PARALLEL_CHUNKS:-4}"
MIN_AVAILABLE_KIB=$((20 * 1024 * 1024))

if (( PARALLEL_CHUNKS < 1 || PARALLEL_CHUNKS > 4 )); then
    echo "MATCH_PARALLEL_CHUNKS must be between 1 and 4 (OOM safety cap)" >&2
    exit 2
fi

check_available_memory() {
    local available_kib
    available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    if (( available_kib < MIN_AVAILABLE_KIB )); then
        echo "Need at least 20 GiB MemAvailable before starting a match wave" >&2
        exit 2
    fi
}

mkdir -p "$WORK"
exec 9>"${WORK}/run.lock"
if ! flock -n 9; then
    echo "Another leaf-loss strength match is already running" >&2
    exit 2
fi

"${REPO_ROOT}/scripts/project_python.sh" \
    "${REPO_ROOT}/scripts/create_match_openings.py" \
    --input "${REPO_ROOT}/data/accuracy_eval_10k/validation.jsonl" \
    --output "$ALL_OPENINGS" \
    --count 1800 \
    --start 200 \
    --seed 20260824 \
    --max-abs-eval 500

split -l 50 -d -a 2 --additional-suffix=.jsonl \
    "$ALL_OPENINGS" "${WORK}/openings_chunk_"

run_chunk() {
    local index="$1"
    local result="${REPO_ROOT}/results/match_leaf_loss_strength_chunk_${index}.json"
    if [[ -s "$result" ]]; then
        echo "Skipping completed chunk: $index"
        return
    fi
    bash "${REPO_ROOT}/scripts/eval_match.sh" \
        "$result" \
        --engine1 "${REPO_ROOT}/bin/YaneuraOu-expert-blending.sh" \
        --engine1-options "Threads=1,EvalDir=${REPO_ROOT}/bin/eval,ExpertBlendingDir=${REPO_ROOT}/tmp/proxy_gap_actual500k_release" \
        --engine2 "${REPO_ROOT}/bin/YaneuraOu-expert-blending.sh" \
        --engine2-options "Threads=1,EvalDir=${REPO_ROOT}/bin/eval,ExpertBlendingDir=${REPO_ROOT}/tmp/proxy_gap_control510_release" \
        --games 100 \
        --nodes 100000 \
        --openings "${WORK}/openings_chunk_${index}.jsonl"
}

pids=()
for index in $(seq -w 0 35); do
    if [[ -s "${REPO_ROOT}/results/match_leaf_loss_strength_chunk_${index}.json" ]]; then
        echo "Skipping completed chunk: $index"
        continue
    fi
    if (( ${#pids[@]} == 0 )); then
        check_available_memory
    fi
    run_chunk "$index" &
    pids+=("$!")
    if (( ${#pids[@]} == PARALLEL_CHUNKS )); then
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
        pids=()
    fi
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

old_results=(
    "${REPO_ROOT}/results/match_proxy_gap_nodes100000_chunk0.json"
    "${REPO_ROOT}/results/match_proxy_gap_nodes100000_chunk1.json"
    "${REPO_ROOT}/results/match_proxy_gap_nodes100000_chunk2.json"
    "${REPO_ROOT}/results/match_proxy_gap_nodes100000_chunk3.json"
)
new_results=()
for index in $(seq -w 0 35); do
    new_results+=("${REPO_ROOT}/results/match_leaf_loss_strength_chunk_${index}.json")
done
"${REPO_ROOT}/scripts/project_python.sh" \
    "${REPO_ROOT}/scripts/merge_match_results.py" \
    --input "${old_results[@]}" "${new_results[@]}" \
    --expected-games 4000 \
    --require-color-pairs \
    --output "${REPO_ROOT}/results/match_leaf_loss_strength_4000.json"
