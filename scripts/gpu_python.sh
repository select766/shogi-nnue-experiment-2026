#!/usr/bin/env bash
# Run a PyTorch command only after a real CUDA operation succeeds.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NNUE_PYTHON="${REPO_ROOT}/scripts/nnue_python.sh"

if ! GPU_INFO="$($NNUE_PYTHON -c '
import torch

if not torch.cuda.is_available():
    raise SystemExit(1)

device = torch.device("cuda:0")
x = torch.tensor([2.0, 3.0], device=device)
y = (x * x).sum()
torch.cuda.synchronize(device)
if y.item() != 13.0:
    raise SystemExit(1)

p = torch.cuda.get_device_properties(device)
print(f"GPU preflight: {p.name}; CUDA {torch.version.cuda}; capability {p.major}.{p.minor}")
')"; then
    cat >&2 <<'EOF'
ERROR: CUDA device execution is unavailable.
GPU-required commands must be run outside the Codex sandbox.
Re-run the same command with sandbox escalation; do not use CPU fallback.
EOF
    exit 1
fi

echo "$GPU_INFO"
exec "$NNUE_PYTHON" "$@"
