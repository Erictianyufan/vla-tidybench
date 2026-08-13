#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-/home/ubuntu/data/vla-tidybench/raw/stack_train_candidate_10.hdf5}"

exec "${PROJECT_ROOT}/scripts/run_vendor_isaac.sh" scripts/tools/replay_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 \
  --dataset_file "${DATASET}" \
  --num_envs 1 \
  --reset_sim_buffer_each_episode \
  --validate_success_rate \
  --enable_cameras \
  --device cuda:0 \
  --viz none
