#!/bin/bash
# Run shuffle_kifu (qsearch + shuffle) on each split directory.
# Prerequisite: split_and_shuffle.py has been run to create input_* directories.
#
# Usage: bash run_shuffle_splits.sh
#
# Output: split_v1/{train,val1,val2,val3,val4,test}.bin
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPLIT_BASE="/home/select766/exthdd/dev/train-nnue/split_v1"
THREADS=8

SPLITS="train val1 val2 val3 val4 test"

for SPLIT in $SPLITS; do
    INPUT_DIR="${SPLIT_BASE}/input_${SPLIT}"
    OUTPUT_DIR="${SPLIT_BASE}/output_${SPLIT}"
    FINAL_BIN="${SPLIT_BASE}/${SPLIT}.bin"

    if [ -f "$FINAL_BIN" ]; then
        echo "=== Skipping ${SPLIT}: ${FINAL_BIN} already exists ==="
        continue
    fi

    echo "=== Processing split: ${SPLIT} ==="
    echo "  Input:  ${INPUT_DIR}"
    echo "  Output: ${OUTPUT_DIR}"

    # Clean output directory
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    # Run shuffle_kifu
    bash "${SCRIPT_DIR}/run_shuffle.sh" "$INPUT_DIR" "$OUTPUT_DIR" "$THREADS"

    # Rename shuffled.bin to split-specific name
    if [ -f "${OUTPUT_DIR}/shuffled.bin" ]; then
        mv "${OUTPUT_DIR}/shuffled.bin" "$FINAL_BIN"
        echo "  Renamed to: ${FINAL_BIN}"
        # Clean up output directory
        rm -rf "$OUTPUT_DIR"
    else
        echo "  ERROR: ${OUTPUT_DIR}/shuffled.bin not found!"
        exit 1
    fi

    echo "=== Done: ${SPLIT} ==="
    echo ""
done

echo "All splits processed."
ls -lh "${SPLIT_BASE}"/*.bin
