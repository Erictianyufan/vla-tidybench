#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-/home/ubuntu/openpi}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -x "${OPENPI_ROOT}/.venv/bin/python" ]]; then
  echo "OpenPI environment not found: ${OPENPI_ROOT}/.venv" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${OPENPI_CUDA_VISIBLE_DEVICES:-1}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/home/ubuntu/data/vla-tidybench/checkpoints}"
export HF_HOME="${HF_HOME:-/home/ubuntu/data/vla-tidybench/cache/huggingface}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONPATH="${PROJECT_ROOT}/source:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"
exec "${OPENPI_ROOT}/.venv/bin/python" "$@"

