#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLA_TIDYBENCH_DATA="${VLA_TIDYBENCH_DATA:-${HOME}/data/vla-tidybench}"
INPUT="${1:-${VLA_TIDYBENCH_DATA}/raw/stack_train_candidate_10.hdf5}"
OUTPUT="${2:-${VLA_TIDYBENCH_DATA}/raw/stack_annotated.hdf5}"

exec "${PROJECT_ROOT}/scripts/run_vendor_isaac.sh" \
  scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 \
  --input_file "${INPUT}" \
  --output_file "${OUTPUT}" \
  --auto \
  --enable_cameras \
  --device cuda:0 \
  --viz none
