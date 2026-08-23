#!/bin/bash
# Benchmark different num_workers values to find the optimal GPU utilization.
# Each run trains for a small number of epochs and logs GPU utilization.
# Results are saved under logs/benchmark_num_workers/ (does NOT touch existing results).
#
# Usage: bash scripts/benchmark_num_workers.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NNUE_PYTORCH_DIR="${SCRIPT_DIR}/nnue-pytorch"
SPLIT_BASE="./dataset/split_v1"
BENCHMARK_DIR="${SCRIPT_DIR}/logs/benchmark_num_workers"

TRAIN_BIN="${SPLIT_BASE}/train.bin"
VAL_BIN="${SPLIT_BASE}/val1.bin"

# Benchmark parameters
BENCHMARK_EPOCHS=50       # number of epochs per run (small, just for measuring throughput)
EPOCH_SIZE=1000000        # same as original
BATCH_SIZE=16384          # same as original
NUM_WORKERS_LIST="1 8 16 64"  # values to test

# Check data files exist
if [ ! -f "$TRAIN_BIN" ]; then
    echo "ERROR: Training data not found: ${TRAIN_BIN}"
    exit 1
fi
if [ ! -f "$VAL_BIN" ]; then
    echo "ERROR: Validation data not found: ${VAL_BIN}"
    exit 1
fi

mkdir -p "$BENCHMARK_DIR"

cd "$NNUE_PYTORCH_DIR"
source .venv/bin/activate

echo "=== num_workers benchmark ==="
echo "Epochs per run: ${BENCHMARK_EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Epoch size: ${EPOCH_SIZE}"
echo "Testing num_workers: ${NUM_WORKERS_LIST}"
echo ""

for NW in $NUM_WORKERS_LIST; do
    RUN_DIR="${BENCHMARK_DIR}/nw${NW}"
    LOG_FILE="${RUN_DIR}/train.log"
    GPU_LOG="${RUN_DIR}/gpu_util.csv"

    if [ -d "$RUN_DIR" ]; then
        echo ">>> Skipping num_workers=${NW} (already exists: ${RUN_DIR})"
        continue
    fi

    mkdir -p "$RUN_DIR"
    echo ">>> Starting num_workers=${NW} ..."

    # Start GPU monitoring in background (sample every 1 second)
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
        --format=csv,nounits -l 1 > "$GPU_LOG" 2>&1 &
    GPU_MON_PID=$!

    START_TIME=$(date +%s)

    python train.py \
      --features "HalfKP" \
      --batch-size "$BATCH_SIZE" \
      --max_epochs "$BENCHMARK_EPOCHS" \
      --enable_progress_bar False \
      --default_root_dir "$RUN_DIR" \
      --threads 8 \
      --lr 0.5 0.05 \
      --num-workers "$NW" \
      --lambda 1.0 0.5 \
      --label-smoothing-eps 0.001 \
      --accelerator gpu \
      --devices 1 \
      --score-scaling 361 \
      --min-newbob-scale 0 \
      --epoch-size "$EPOCH_SIZE" \
      --num-epochs-to-adjust-lr 100000 \
      --momentum 0.9 \
      --network-save-period 100000 \
      "$TRAIN_BIN" \
      "$VAL_BIN" \
      > "$LOG_FILE" 2>&1

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    # Stop GPU monitoring
    kill "$GPU_MON_PID" 2>/dev/null || true
    wait "$GPU_MON_PID" 2>/dev/null || true

    # Compute average GPU utilization (skip header line)
    AVG_GPU=$(awk -F', ' 'NR>1 {sum+=$2; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}' "$GPU_LOG")

    echo "    Elapsed: ${ELAPSED}s, Avg GPU util: ${AVG_GPU}%"
    echo "num_workers=${NW} elapsed=${ELAPSED}s avg_gpu_util=${AVG_GPU}%" >> "${BENCHMARK_DIR}/summary.txt"
    echo ""
done

echo "=== Benchmark complete ==="
echo "Summary:"
if [ -f "${BENCHMARK_DIR}/summary.txt" ]; then
    cat "${BENCHMARK_DIR}/summary.txt"
fi
echo ""
echo "Detailed GPU logs: ${BENCHMARK_DIR}/nw*/gpu_util.csv"
