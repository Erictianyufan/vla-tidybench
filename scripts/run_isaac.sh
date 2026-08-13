#!/usr/bin/env bash
set -euo pipefail

ISAAC_ROOT="${ISAAC_LAB_ROOT:-/home/ubuntu/IsaacLab}"
ISAAC_ENV="${ISAAC_LAB_VENV:-/home/ubuntu/env_isaaclab}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -x "${ISAAC_ROOT}/isaaclab.sh" ]]; then
  echo "Isaac Lab launcher not found: ${ISAAC_ROOT}/isaaclab.sh" >&2
  exit 2
fi
if [[ ! -f "${ISAAC_ENV}/bin/activate" ]]; then
  echo "Isaac Python environment not found: ${ISAAC_ENV}" >&2
  exit 2
fi

source "${ISAAC_ENV}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/source${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"
exec "${ISAAC_ROOT}/isaaclab.sh" -p "$@" \
  --kit_args="--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false"
