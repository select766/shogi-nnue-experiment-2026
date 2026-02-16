#!/bin/bash
# Expert Blending vs ベースラインNNUE の対局スクリプト
#
# 使い方:
#   bash scripts/run_expert_blending_match.sh \
#     <expert_blending_checkpoint> \
#     <backbone_weights> \
#     <baseline_nnue_dir> \
#     [n_experts] [games] [byoyomi_ms]
#
# 例:
#   bash scripts/run_expert_blending_match.sh \
#     logs/expert_blending_v1/checkpoints/1000.ckpt \
#     tmp/dlshogi-model/model_resnet10_swish-072 \
#     bin/eval \
#     8 100 3000

set -e

CHECKPOINT=${1:?Usage: $0 <checkpoint> <backbone_weights> <baseline_eval_dir> [n_experts] [games] [byoyomi]}
BACKBONE_WEIGHTS=${2:?Usage: $0 <checkpoint> <backbone_weights> <baseline_eval_dir> [n_experts] [games] [byoyomi]}
BASELINE_EVAL_DIR=${3:?Usage: $0 <checkpoint> <backbone_weights> <baseline_eval_dir> [n_experts] [games] [byoyomi]}
N_EXPERTS=${4:-8}
GAMES=${5:-100}
BYOYOMI=${6:-3000}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ENGINE_EXPERT="$ROOT_DIR/bin/YaneuraOu-expert-blending"
ENGINE_BASELINE="$ROOT_DIR/bin/YaneuraOu-by-gcc"
NNUE_PYTHON="$ROOT_DIR/nnue-pytorch/.venv/bin/python"

abspath_from_root() {
    case "$1" in
        /*) echo "$1" ;;
        *) echo "$ROOT_DIR/$1" ;;
    esac
}

CHECKPOINT="$(abspath_from_root "$CHECKPOINT")"
BACKBONE_WEIGHTS="$(abspath_from_root "$BACKBONE_WEIGHTS")"
BASELINE_EVAL_DIR="$(abspath_from_root "$BASELINE_EVAL_DIR")"

if [ ! -x "$ENGINE_EXPERT" ]; then
    echo "Error: engine not found or not executable: $ENGINE_EXPERT" >&2
    exit 1
fi
if [ ! -x "$ENGINE_BASELINE" ]; then
    echo "Error: engine not found or not executable: $ENGINE_BASELINE" >&2
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
if [ ! -f "$BASELINE_EVAL_DIR/nn.bin" ]; then
    echo "Error: baseline eval file not found: $BASELINE_EVAL_DIR/nn.bin" >&2
    exit 1
fi

# DNN推論サーバーのコマンド
DNN_CMD="PYTHONPATH=$ROOT_DIR/src:\$PYTHONPATH $NNUE_PYTHON -m train_nnue.dnn_inference_server --checkpoint $CHECKPOINT --backbone-weights $BACKBONE_WEIGHTS --features HalfKP --n-experts $N_EXPERTS"

echo "=== Expert Blending Match ==="
echo "Checkpoint: $CHECKPOINT"
echo "Backbone: $BACKBONE_WEIGHTS"
echo "Baseline eval dir: $BASELINE_EVAL_DIR"
echo "N experts: $N_EXPERTS"
echo "Games: $GAMES"
echo "Byoyomi: ${BYOYOMI}ms"
echo ""

cd "$ROOT_DIR/nnue-pytorch"
source .venv/bin/activate

PYTHONPATH="$ROOT_DIR/src:$PYTHONPATH" python -m train_nnue.run_match \
    --engine1 "$ENGINE_EXPERT" \
    --engine1-options "EvalDir=$BASELINE_EVAL_DIR,DNNServerCmd=$DNN_CMD" \
    --engine2 "$ENGINE_BASELINE" \
    --engine2-options "EvalDir=$BASELINE_EVAL_DIR" \
    --games "$GAMES" \
    --byoyomi "$BYOYOMI"
