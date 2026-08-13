#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/configs/versions.env"

echo "== hardware =="
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id --format=csv,noheader
echo "logical_cpu=$(nproc)"
free -h
df -hT /

echo "== pinned runtime =="
echo "isaac_lab_root=${ISAAC_LAB_ROOT}"
echo "isaac_lab_vendor_commit=$(git -C "${ISAAC_LAB_ROOT}" rev-parse HEAD)"
"${ISAAC_LAB_PYTHON}" - <<'PY'
from importlib import metadata
for package in ("isaacsim", "isaaclab", "isaaclab_tasks", "torch", "numpy"):
    print(f"{package}={metadata.version(package)}")
PY

echo "== project =="
PYTHONPATH="${PROJECT_ROOT}/source" "${ISAAC_LAB_PYTHON}" - <<'PY'
from vla_tidybench.policy_bridge import ActionAdapter
print("adapter_zero=", ActionAdapter().to_isaac([0, 0, 0, 0, 0, 0, 1]).tolist())
PY

