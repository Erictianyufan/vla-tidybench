#!/usr/bin/env bash
set -euo pipefail

ISAAC_ROOT="${ISAAC_LAB_ROOT:-/home/ubuntu/IsaacLab}"
ISAAC_ENV="${ISAAC_LAB_VENV:-/home/ubuntu/env_isaaclab}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 VENDOR_SCRIPT [ARGS...]" >&2
  exit 2
fi

source "${ISAAC_ENV}/bin/activate"
cd "${ISAAC_ROOT}"
exec "${ISAAC_ROOT}/isaaclab.sh" -p "$@" \
  --kit_args="--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false"

