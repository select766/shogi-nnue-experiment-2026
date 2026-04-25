#!/bin/bash
# Expert Blending の評価関数合成にかかる時間を計測するスクリプト。
#
# go nodes <N> を回して bestmove までの時間を測る。
# N が小さければ探索コストは無視できるので、合成パイプラインの代理指標になる。
#
# Usage:
#   bash scripts/benchmark_expert_blending_speed.sh \
#     <expert_blending_checkpoint> \
#     [backbone_weights] [n_experts] [nodes] [iters]
#
# 例 (デフォルトの 180.ckpt を測る):
#   bash scripts/benchmark_expert_blending_speed.sh \
#     logs/expert_blending_8experts_v4_paired_uniform50_noise0_lambda05/checkpoints/180.ckpt
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
NNUE_PYTHON="$ROOT_DIR/nnue-pytorch/.venv/bin/python"
EVAL_DIR="$ROOT_DIR/bin/eval"
SERVER_LOG=${DNN_SERVER_LOG:-/tmp/dnn_inference_server_bench.log}

if [ ! -x "$ENGINE" ]; then
    echo "Error: engine not found or not executable: $ENGINE" >&2
    exit 1
fi
if [ ! -x "$NNUE_PYTHON" ]; then
    echo "Error: python not found: $NNUE_PYTHON" >&2
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

PYTHONPATH="$ROOT_DIR/src:$PYTHONPATH" python -m train_nnue.benchmark_blending_speed \
    --engine "$ENGINE" \
    --checkpoint "$CHECKPOINT" \
    --backbone-weights "$BACKBONE_WEIGHTS" \
    --baseline-eval-dir "$EVAL_DIR" \
    --n-experts "$N_EXPERTS" \
    --nodes "$NODES" \
    --iters "$ITERS" \
    --server-log "$SERVER_LOG" \
    --python "$NNUE_PYTHON"
