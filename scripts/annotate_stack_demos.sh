#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:-/home/ubuntu/data/vla-tidybench/raw/stack_human.hdf5}"
OUTPUT="${2:-/home/ubuntu/data/vla-tidybench/raw/stack_annotated.hdf5}"

exec "${PROJECT_ROOT}/scripts/run_vendor_isaac.sh" \
  scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 \
  --input_file "${INPUT}" \
  --output_file "${OUTPUT}" \
  --auto \
  --enable_cameras \
  --device cuda:0 \
  --viz none

