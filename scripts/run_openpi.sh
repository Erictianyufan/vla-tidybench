#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)/openpi}"
VLA_TIDYBENCH_DATA="${VLA_TIDYBENCH_DATA:-/data/${USER}/vla-tidybench}"

if [[ ! -x "${OPENPI_ROOT}/.venv/bin/python" ]]; then
  echo "OpenPI environment not found: ${OPENPI_ROOT}/.venv" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${OPENPI_CUDA_VISIBLE_DEVICES:-1}"
export OPENPI_ROOT VLA_TIDYBENCH_DATA
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${VLA_TIDYBENCH_DATA}/checkpoints}"
export HF_HOME="${HF_HOME:-${VLA_TIDYBENCH_DATA}/cache/huggingface}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONPATH="${PROJECT_ROOT}/source:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${VLA_TIDYBENCH_DATA}/checkpoints" "${VLA_TIDYBENCH_DATA}/cache/huggingface" "${VLA_TIDYBENCH_DATA}/logs"
cd "${PROJECT_ROOT}"
exec "${OPENPI_ROOT}/.venv/bin/python" "$@"

