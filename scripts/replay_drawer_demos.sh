#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DATASET_FILE:?set DATASET_FILE to the drawer HDF5 path}"
: "${SKILL:?set SKILL to open, pick, place, close, or full}"

cd "${PROJECT_ROOT}"
exec ./scripts/run_isaac.sh scripts/replay_drawer_demos.py \
  --dataset_file "${DATASET_FILE}" \
  --skill "${SKILL}" \
  --reset_sim_buffer_each_episode \
  --device "${DEVICE:-cuda:0}" \
  --enable_cameras \
  --viz "${VIZ:-none}" \
  "$@"
