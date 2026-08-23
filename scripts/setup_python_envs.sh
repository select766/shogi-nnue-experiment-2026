#!/usr/bin/env bash
# Create/update both Python environments without shell activation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NNUE_PYTHON="${REPO_ROOT}/nnue-pytorch/.venv/bin/python"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}"

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
}

cd "$REPO_ROOT"
uv sync --frozen --no-dev

if [[ ! -x "$NNUE_PYTHON" ]]; then
    uv venv --python 3.11 "${REPO_ROOT}/nnue-pytorch/.venv"
fi

uv pip install --python "$NNUE_PYTHON" \
    --index-url https://download.pytorch.org/whl/cu121 \
    'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121' 'torchaudio==2.5.1+cu121'
uv pip install --python "$NNUE_PYTHON" -r requirements/nnue.txt

echo "Python environments are ready."
echo "Verify CPU-side imports: bash scripts/check_environment.sh"
echo "Verify GPU outside the sandbox: bash scripts/check_environment.sh --require-gpu"
