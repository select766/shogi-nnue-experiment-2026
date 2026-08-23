#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:?Usage: $0 CHECKPOINT OUTPUT_DIR [N_EXPERTS] [BACKBONE_WEIGHTS]}"
OUTPUT_DIR="${2:?Usage: $0 CHECKPOINT OUTPUT_DIR [N_EXPERTS] [BACKBONE_WEIGHTS]}"
N_EXPERTS="${3:-8}"
BACKBONE_WEIGHTS="${4:-${REPO_ROOT}/tmp/dlshogi-model/model_resnet10_swish-072}"

"${REPO_ROOT}/scripts/nnue_python.sh" -m train_nnue.export_for_yaneuraou \
    --checkpoint "$(readlink -f "$CHECKPOINT")" \
    --backbone-weights "$(readlink -f "$BACKBONE_WEIGHTS")" \
    --features HalfKP \
    --n-experts "$N_EXPERTS" \
    --output-dir "$OUTPUT_DIR"
