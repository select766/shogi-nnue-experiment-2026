#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQUIRE_GPU=0
if [[ "${1:-}" == "--require-gpu" ]]; then
    REQUIRE_GPU=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--require-gpu]" >&2
    exit 2
fi

"${REPO_ROOT}/scripts/project_python.sh" -c \
    'import cshogi, numpy, train_nnue; print("root environment: OK")'

"${REPO_ROOT}/scripts/nnue_python.sh" -c \
    'import cshogi, dlshogi, nnue_dataset, onnx, pytorch_lightning, torch, train_nnue; print("training environment: OK"); print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")'

if [[ "$REQUIRE_GPU" -eq 1 ]]; then
    "${REPO_ROOT}/scripts/gpu_python.sh" -c \
        'import torch; x = torch.randn(1024, 1024, device="cuda"); y = x @ x; torch.cuda.synchronize(); print(f"CUDA matmul: OK ({y.device}, shape={tuple(y.shape)})")'
fi

test -f "${REPO_ROOT}/nnue-pytorch/libtraining_data_loader.so" || {
    echo "ERROR: nnue-pytorch/libtraining_data_loader.so is missing." >&2
    exit 1
}
echo "native data loader: OK"
