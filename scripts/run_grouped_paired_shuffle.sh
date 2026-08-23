#!/usr/bin/env bash
# Generate root-grouped qsearch data while preserving each group through shuffle.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${1:?Usage: $0 INPUT_DIR OUTPUT_DIR GROUP_SIZE MAX_ROOTS SEED [THREADS]}"
OUTPUT_DIR="${2:?Usage: $0 INPUT_DIR OUTPUT_DIR GROUP_SIZE MAX_ROOTS SEED [THREADS]}"
GROUP_SIZE="${3:?missing group size}"
MAX_ROOTS="${4:?missing max roots}"
SEED="${5:?missing seed}"
THREADS="${6:-8}"
ENGINE="${REPO_ROOT}/bin/shuffle/tanuki-learner"

[[ "$GROUP_SIZE" -gt 1 ]] || { echo "ERROR: GROUP_SIZE must exceed 1" >&2; exit 2; }
[[ "$MAX_ROOTS" -gt 0 ]] || { echo "ERROR: MAX_ROOTS must be positive" >&2; exit 2; }
[[ "$SEED" -gt 0 ]] || { echo "ERROR: SEED must be positive" >&2; exit 2; }
[[ -x "$ENGINE" ]] || { echo "ERROR: missing engine: $ENGINE" >&2; exit 1; }
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
MAX_PAIRS=$((GROUP_SIZE * MAX_ROOTS))
TEMP_DIR="$(mktemp -d /tmp/yaneuraou_grouped_shuffle.XXXXXX)"
FIFO="${TEMP_DIR}/commands.fifo"
mkfifo "$FIFO"
cleanup() {
    rm -f "$FIFO"
    rmdir "$TEMP_DIR" 2>/dev/null || true
}
trap cleanup EXIT

cd "${REPO_ROOT}/bin/shuffle"
"$ENGINE" < "$FIFO" 2>&1 &
ENGINE_PID=$!
exec 3>"$FIFO"
echo "setoption name Threads value $THREADS" >&3
echo "setoption name KifuDir value $INPUT_DIR" >&3
echo "setoption name ShuffledKifuDir value $OUTPUT_DIR" >&3
echo "setoption name ApplyQSearch value true" >&3
echo "setoption name PairedShuffle value true" >&3
echo "setoption name PairedGroupSize value $GROUP_SIZE" >&3
echo "setoption name MaxOutputSamples value $MAX_PAIRS" >&3
echo "setoption name OffsetDistribution value uniform" >&3
echo "setoption name OffsetUniformMax value 50" >&3
echo "setoption name ShuffleSeed value $SEED" >&3
echo "isready" >&3
sleep 3
echo "shuffle_kifu" >&3

while kill -0 "$ENGINE_PID" 2>/dev/null; do
    if [[ -f "$OUTPUT_DIR/shuffled.bin" ]]; then
        sleep 2
        echo "quit" >&3
        break
    fi
    sleep 1
done
set +e
wait "$ENGINE_PID"
EXIT_CODE=$?
set -e
exec 3>&-
[[ "$EXIT_CODE" -eq 0 ]] || { echo "ERROR: shuffler exited $EXIT_CODE" >&2; exit "$EXIT_CODE"; }
[[ -f "$OUTPUT_DIR/shuffled.bin" ]] || { echo "ERROR: shuffled.bin missing" >&2; exit 1; }

"${REPO_ROOT}/scripts/project_python.sh" \
    "${REPO_ROOT}/scripts/split_grouped_paired_bin.py" \
    --input "$OUTPUT_DIR/shuffled.bin" \
    --output-dir "$OUTPUT_DIR" \
    --group-size "$GROUP_SIZE" \
    --seed "$SEED" \
    --source "$INPUT_DIR"
rm -f "$OUTPUT_DIR/shuffled.bin"
