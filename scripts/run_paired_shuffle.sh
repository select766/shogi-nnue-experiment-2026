#!/bin/bash
# Run shuffle_kifu in paired mode:
# output record format = [DNN 40B (no qsearch) | NNUE 40B (qsearch)]
#
# Usage: bash scripts/run_paired_shuffle.sh <input_dir> <output_dir> [threads] [max_output_samples] [offset_uniform_max]
# Example: bash scripts/run_paired_shuffle.sh dataset/split_v1/input_train dataset/split_v1_paired/output_train 8 480000000 50
#
# IMPORTANT: input_dir must contain only .bin files.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_DIR="${1:?Usage: $0 <input_dir> <output_dir> [threads] [max_output_samples] [offset_uniform_max]}"
OUTPUT_DIR="${2:?Usage: $0 <input_dir> <output_dir> [threads] [max_output_samples] [offset_uniform_max]}"
THREADS="${3:-8}"
MAX_OUTPUT_SAMPLES="${4:-0}"
OFFSET_UNIFORM_MAX="${5:-50}"
ENGINE="${SCRIPT_DIR}/bin/shuffle/tanuki-learner"
ENGINE_CWD="${SCRIPT_DIR}/bin/shuffle"

# Convert to absolute paths
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

FIFO="/tmp/yaneuraou_paired_shuffle_fifo_$$"
mkfifo "$FIFO"

cd "$ENGINE_CWD"
"$ENGINE" < "$FIFO" 2>&1 &
ENGINE_PID=$!
echo "Engine PID: $ENGINE_PID"

exec 3>"$FIFO"

echo "setoption name Threads value $THREADS" >&3
echo "setoption name KifuDir value $INPUT_DIR" >&3
echo "setoption name ShuffledKifuDir value $OUTPUT_DIR" >&3
echo "setoption name ApplyQSearch value true" >&3
echo "setoption name PairedShuffle value true" >&3
if [ "$MAX_OUTPUT_SAMPLES" -gt 0 ]; then
    echo "setoption name MaxOutputSamples value $MAX_OUTPUT_SAMPLES" >&3
fi
echo "setoption name OffsetDistribution value uniform" >&3
echo "setoption name OffsetUniformMax value $OFFSET_UNIFORM_MAX" >&3
echo "isready" >&3
sleep 3
echo "shuffle_kifu" >&3

for i in $(seq 1 100000); do
    if ! kill -0 $ENGINE_PID 2>/dev/null; then
        wait $ENGINE_PID 2>/dev/null
        EXIT_CODE=$?
        echo "=== Engine process exited with code: $EXIT_CODE ==="
        if [ $EXIT_CODE -gt 128 ]; then
            SIG=$((EXIT_CODE - 128))
            echo "=== Killed by signal: $SIG ==="
        fi
        break
    fi
    if [ -f "$OUTPUT_DIR/shuffled.bin" ]; then
        sleep 5
        echo "quit" >&3
        wait $ENGINE_PID 2>/dev/null
        EXIT_CODE=$?
        echo "=== Completed. Engine exit code: $EXIT_CODE ==="
        break
    fi
    sleep 1
done

exec 3>&-
rm -f "$FIFO"
