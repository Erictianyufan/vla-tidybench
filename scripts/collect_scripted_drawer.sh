#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL="${SKILL:-open}"
DATASET="${DATASET_FILE:-${DATASET:-/home/ubuntu/data/vla-tidybench/raw/drawer_${SKILL}_scripted.hdf5}}"

cd "${PROJECT_ROOT}"
exec ./scripts/run_isaac.sh scripts/collect_scripted_drawer.py \
  --skill "${SKILL}" \
  --dataset_file "${DATASET}" \
  --num_demos "${NUM_DEMOS:-1}" \
  --max_attempts "${MAX_ATTEMPTS:-5}" \
  --max_steps "${MAX_STEPS:-720}" \
  --seed "${SEED:-101}" \
  --device "${DEVICE:-cuda:0}" \
  --enable_cameras \
  --viz "${VIZ:-none}" \
  "$@"
