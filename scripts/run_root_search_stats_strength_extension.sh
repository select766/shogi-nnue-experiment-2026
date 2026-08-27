#!/usr/bin/env bash
# Add the preregistered 4,000 fresh games to a positive, uncertain first run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${REPO_ROOT}/tmp/root_search_stats_strength_extension"
OPENINGS="${WORK}/openings_2000.jsonl"
PARALLEL_CHUNKS="${MATCH_PARALLEL_CHUNKS:-4}"
MIN_AVAILABLE_KIB=$((20 * 1024 * 1024))

if (( PARALLEL_CHUNKS < 1 || PARALLEL_CHUNKS > 4 )); then
    echo "MATCH_PARALLEL_CHUNKS must be between 1 and 4" >&2
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
    echo "Another root-search-statistics extension is already running" >&2
    exit 2
fi
"${REPO_ROOT}/scripts/project_python.sh" \
    "${REPO_ROOT}/scripts/create_match_openings.py" \
    --input "${REPO_ROOT}/data/accuracy_eval_10k/validation.jsonl" \
    --output "$OPENINGS" --count 2000 --start 4200 \
    --seed 20260826 --max-abs-eval 500
split -l 50 -d -a 2 --additional-suffix=.jsonl \
    "$OPENINGS" "${WORK}/openings_chunk_"

run_chunk() {
    local index="$1"
    local result="${REPO_ROOT}/results/match_root_search_stats_extension_chunk_${index}.json"
    if [[ -s "$result" ]]; then
        echo "Skipping completed chunk: $index"
        return
    fi
    bash "${REPO_ROOT}/scripts/eval_match.sh" "$result" \
        --engine1 "${REPO_ROOT}/bin/YaneuraOu-expert-blending.sh" \
        --engine1-options "Threads=1,EvalDir=${REPO_ROOT}/bin/eval,ExpertBlendingDir=${REPO_ROOT}/tmp/root_search_stats_release" \
        --engine2 "${REPO_ROOT}/bin/YaneuraOu-expert-blending.sh" \
        --engine2-options "Threads=1,EvalDir=${REPO_ROOT}/bin/eval,ExpertBlendingDir=${REPO_ROOT}/tmp/root_features_combined_release" \
        --engine1-root-stats-engine "${REPO_ROOT}/bin/YaneuraOu-by-gcc" \
        --engine1-root-stats-options "Threads=1,USI_Hash=16,EvalDir=${REPO_ROOT}/bin/eval,USI_OwnBook=false" \
        --root-stats-nodes 1024 --root-stats-multipv 4 \
        --clear-hash-each-move --games 100 --nodes 100000 \
        --openings "${WORK}/openings_chunk_${index}.jsonl"
}

pids=()
for index in $(seq -w 0 39); do
    if [[ -s "${REPO_ROOT}/results/match_root_search_stats_extension_chunk_${index}.json" ]]; then
        continue
    fi
    if (( ${#pids[@]} == 0 )); then check_available_memory; fi
    run_chunk "$index" &
    pids+=("$!")
    if (( ${#pids[@]} == PARALLEL_CHUNKS )); then
        for pid in "${pids[@]}"; do wait "$pid"; done
        pids=()
    fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

results=()
for index in $(seq -w 0 39); do
    results+=("${REPO_ROOT}/results/match_root_search_stats_chunk_${index}.json")
done
for index in $(seq -w 0 39); do
    results+=("${REPO_ROOT}/results/match_root_search_stats_extension_chunk_${index}.json")
done
"${REPO_ROOT}/scripts/project_python.sh" \
    "${REPO_ROOT}/scripts/merge_match_results.py" \
    --input "${results[@]}" --expected-games 8000 --require-color-pairs \
    --output "${REPO_ROOT}/results/match_root_search_stats_8000.json"
