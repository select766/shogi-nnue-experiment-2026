#!/bin/bash
# Run shuffle_kifu with qsearch applied
# Usage: bash run_shuffle.sh <input_dir> <output_dir> [threads]
# Example: bash run_shuffle.sh test_input_noreadme dataset_qsearch_shuffled 8
#
# IMPORTANT: input_dir must contain only .bin files (no README.md etc.)
# See how-to-qsearch-shuffle.md for details.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_DIR="${1:?Usage: $0 <input_dir> <output_dir> [threads]}"
OUTPUT_DIR="${2:?Usage: $0 <input_dir> <output_dir> [threads]}"
THREADS="${3:-8}"
ENGINE="${SCRIPT_DIR}/tanuki-learner/source/YaneuraOu-by-gcc"

# Convert to absolute paths
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

FIFO="/tmp/yaneuraou_shuffle_fifo_$$"
mkfifo "$FIFO"

cd "$(dirname "$ENGINE")"
"$ENGINE" < "$FIFO" 2>&1 &
ENGINE_PID=$!
echo "Engine PID: $ENGINE_PID"

exec 3>"$FIFO"

echo "setoption name Threads value $THREADS" >&3
echo "setoption name KifuDir value $INPUT_DIR" >&3
echo "setoption name ShuffledKifuDir value $OUTPUT_DIR" >&3
echo "setoption name ApplyQSearch value true" >&3
echo "isready" >&3
sleep 3
echo "shuffle_kifu" >&3

for i in $(seq 1 3600); do
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
