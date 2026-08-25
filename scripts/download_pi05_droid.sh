#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)/openpi}"
VLA_TIDYBENCH_DATA="${VLA_TIDYBENCH_DATA:-/data/${USER}/vla-tidybench}"
LOG_DIR="${VLA_TIDYBENCH_DATA}/logs"
mkdir -p "${LOG_DIR}"

export OPENPI_ROOT VLA_TIDYBENCH_DATA

nohup "${PROJECT_ROOT}/scripts/run_openpi.sh" -c \
  "from pathlib import Path; from openpi.shared import download; target=Path('${OPENPI_DATA_HOME:-${VLA_TIDYBENCH_DATA}/checkpoints}')/'openpi-assets/checkpoints/pi05_droid'; marker=target/'params/_CHECKPOINT_METADATA'; result=download.maybe_download('gs://openpi-assets/checkpoints/pi05_droid', force_download=target.exists() and not marker.is_file()); assert (result/'params/_CHECKPOINT_METADATA').is_file(), f'incomplete checkpoint: {result}'; print(result)" \
  > "${LOG_DIR}/pi05_droid_download.log" 2>&1 < /dev/null &
echo "pi05-DROID download started in background: pid=$!"
echo "log=${LOG_DIR}/pi05_droid_download.log"

