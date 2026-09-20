#!/usr/bin/env bash
# Queue administration is sandbox-safe; `run` belongs in the user's host shell.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${REPO_ROOT}/scripts/project_python.sh" -m train_nnue.research_loop "$@"
