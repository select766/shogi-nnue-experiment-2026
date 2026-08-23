#!/usr/bin/env bash
# Run repository data-processing modules in the root uv environment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}"

cd "$REPO_ROOT"
exec uv run --frozen --no-dev python "$@"
