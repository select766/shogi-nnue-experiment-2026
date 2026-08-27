#!/usr/bin/env bash
# Build aligned 13-dimensional static plus shallow-search feature caches.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/tmp/root_search_stats_20260827"
mkdir -p "$OUTPUT_ROOT"

for split in train valA valB; do
    case "$split" in
        train) roots=116224 ;;
        *) roots=19968 ;;
    esac
    "${REPO_ROOT}/scripts/project_python.sh" \
        "${REPO_ROOT}/scripts/collect_root_search_statistics.py" \
        --roots-file "${REPO_ROOT}/tmp/proxy_gap_v2/${split}/roots.bin" \
        --engine "${REPO_ROOT}/bin/YaneuraOu-by-gcc" \
        --eval-dir "${REPO_ROOT}/bin/eval" \
        --max-roots "$roots" --nodes 1024 --multipv 4 --workers 8 --hash-mb 16 \
        --output "${OUTPUT_ROOT}/${split}.npy" \
        > "/tmp/root_search_stats_collect_${split}.log" 2>&1
done
