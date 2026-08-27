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
actual_vendor_commit="$(git -C "${ISAAC_LAB_ROOT}" rev-parse HEAD)"
echo "isaac_lab_vendor_commit=${actual_vendor_commit}"
if [[ "${actual_vendor_commit}" != "${ISAAC_LAB_VENDOR_COMMIT}" ]]; then
  echo "unexpected Isaac Lab vendor commit: ${actual_vendor_commit}" >&2
  exit 2
fi
ISAAC_SIM_VERSION="${ISAAC_SIM_VERSION}" "${ISAAC_LAB_PYTHON}" - <<'PY'
import os
from importlib import metadata
for package in ("isaacsim", "isaaclab", "isaaclab_tasks", "torch", "numpy"):
    print(f"{package}={metadata.version(package)}")
if metadata.version("isaacsim") != os.environ["ISAAC_SIM_VERSION"]:
    raise SystemExit("Isaac Sim package version does not match configs/versions.env")
PY

echo "== project =="
echo "project_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
if [[ -n "$(git -C "${PROJECT_ROOT}" status --porcelain)" ]]; then
  echo "project_worktree=dirty" >&2
  exit 2
fi
echo "project_worktree=clean"
VLA_TIDYBENCH_DATA="${VLA_TIDYBENCH_DATA}" \
PYTHONPATH="${PROJECT_ROOT}/source" "${ISAAC_LAB_PYTHON}" - <<'PY'
from vla_tidybench.policy_bridge import ActionAdapter
print("adapter_zero=", ActionAdapter().to_isaac([0, 0, 0, 0, 0, 0, 1]).tolist())
PY

echo "== formal evaluation contexts =="
VLA_TIDYBENCH_DATA="${VLA_TIDYBENCH_DATA}" \
PYTHONPATH="${PROJECT_ROOT}/source" "${ISAAC_LAB_PYTHON}" - <<'PY'
import os
from pathlib import Path

from vla_tidybench.evaluation_contexts import validate_context_lock

root = Path(os.environ["VLA_TIDYBENCH_DATA"])
lock = validate_context_lock(
    root / "manifests" / "pi05-formal" / "main_validation.lock.json",
    root / "manifests" / "pi05-formal" / "main_validation.json",
    root / "raw",
)
if lock.get("context_count") != 40:
    raise SystemExit(f"expected 40 formal contexts, got {lock.get('context_count')}")
print("formal_contexts=40 content_lock=verified")
PY

for required in \
  scripts/run_pi05_eval_suite.py \
  scripts/run_drawer_pi05_closed_loop.py \
  scripts/run_isaac.sh; do
  test -f "${PROJECT_ROOT}/${required}" || {
    echo "missing evaluation runtime: ${PROJECT_ROOT}/${required}" >&2
    exit 2
  }
done
echo "evaluation_runtime=present"

