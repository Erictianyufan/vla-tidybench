#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="/home/ubuntu/data/vla-tidybench/raw/stack_human.hdf5"

echo "This command is interactive. Run it inside the cloud desktop terminal."
echo "Dataset: ${DATASET}"
exec "${PROJECT_ROOT}/scripts/run_vendor_isaac.sh" scripts/tools/record_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 \
  --dataset_file "${DATASET}" \
  --num_demos "${NUM_DEMOS:-10}" \
  --num_success_steps 10 \
  --step_hz 20 \
  --teleop_device keyboard \
  --no-auto_launch_cloudxr \
  --enable_cameras \
  --device cuda:0 \
  --viz kit

