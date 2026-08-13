#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="/home/ubuntu/data/vla-tidybench/logs"
mkdir -p "${LOG_DIR}"

nohup "${PROJECT_ROOT}/scripts/run_openpi.sh" -c \
  "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_droid'))" \
  > "${LOG_DIR}/pi05_droid_download.log" 2>&1 < /dev/null &
echo "pi05-DROID download started in background: pid=$!"
echo "log=${LOG_DIR}/pi05_droid_download.log"

