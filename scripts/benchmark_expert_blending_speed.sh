#!/bin/bash
# Expert Blending の評価関数合成にかかる時間を計測するスクリプト (iter6+ 用)。
#
# iter6 以降は Python サーバーを廃止し、やねうら王本体が
# backbone.onnx + head.bin + head.json を直接ロードする。
# 本スクリプトは export_for_yaneuraou.py で一時ディレクトリへエクスポートし、
# やねうら王に渡して go nodes を回す。
#
# Usage:
#   bash scripts/benchmark_expert_blending_speed.sh \
#     [checkpoint] [backbone_weights] [n_experts] [nodes] [iters]
set -e

CHECKPOINT=${1:-logs/expert_blending_8experts_v4_paired_uniform50_noise0_lambda05/checkpoints/180.ckpt}
BACKBONE_WEIGHTS=${2:-tmp/dlshogi-model/model_resnet10_swish-072}
N_EXPERTS=${3:-8}
NODES=${4:-1000}
ITERS=${5:-20}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

abspath_from_root() {
    case "$1" in
        /*) echo "$1" ;;
        *) echo "$ROOT_DIR/$1" ;;
    esac
}

CHECKPOINT="$(abspath_from_root "$CHECKPOINT")"
BACKBONE_WEIGHTS="$(abspath_from_root "$BACKBONE_WEIGHTS")"
ENGINE="$ROOT_DIR/bin/YaneuraOu-expert-blending"
EVAL_DIR="$ROOT_DIR/bin/eval"
ONNX_LIB_DIR="$ROOT_DIR/YaneuraOu/extra/onnxruntime/linux/current/lib"

if [ ! -x "$ENGINE" ]; then
    echo "Error: engine not found or not executable: $ENGINE" >&2
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi
if [ ! -f "$BACKBONE_WEIGHTS" ]; then
    echo "Error: backbone weights not found: $BACKBONE_WEIGHTS" >&2
    exit 1
fi

cd "$ROOT_DIR/nnue-pytorch"
source .venv/bin/activate

PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/dlshogi-source:$ROOT_DIR/nnue-pytorch:$PYTHONPATH" \
LD_LIBRARY_PATH="$ONNX_LIB_DIR:$LD_LIBRARY_PATH" \
python -m train_nnue.benchmark_blending_speed \
    --engine "$ENGINE" \
    --checkpoint "$CHECKPOINT" \
    --backbone-weights "$BACKBONE_WEIGHTS" \
    --baseline-eval-dir "$EVAL_DIR" \
    --n-experts "$N_EXPERTS" \
    --nodes "$NODES" \
    --iters "$ITERS" \
    --onnxruntime-libdir "$ONNX_LIB_DIR"
