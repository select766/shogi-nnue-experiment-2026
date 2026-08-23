#!/usr/bin/env bash
# Run training/export modules in the dedicated nnue-pytorch environment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/nnue-pytorch/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: training Python not found: $PYTHON_BIN" >&2
    echo "Run: bash scripts/setup_python_envs.sh" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/dlshogi-source:${REPO_ROOT}/nnue-pytorch${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/train-nnue-matplotlib}"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$@"
