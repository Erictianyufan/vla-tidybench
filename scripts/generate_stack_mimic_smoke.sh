#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLA_TIDYBENCH_DATA="${VLA_TIDYBENCH_DATA:-${HOME}/data/vla-tidybench}"
INPUT="${1:-${VLA_TIDYBENCH_DATA}/raw/stack_annotated.hdf5}"
OUTPUT="${2:-${VLA_TIDYBENCH_DATA}/raw/stack_mimic_smoke.hdf5}"

exec "${PROJECT_ROOT}/scripts/run_vendor_isaac.sh" \
  scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 \
  --input_file "${INPUT}" \
  --output_file "${OUTPUT}" \
  --generation_num_trials 10 \
  --num_envs "${NUM_ENVS:-4}" \
  --enable_cameras \
  --device cuda:0 \
  --viz none

