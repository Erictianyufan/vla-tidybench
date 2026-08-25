#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NUM_DEMOS=${NUM_DEMOS:-7}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-28}
SEED=${SEED:-41}
VLA_TIDYBENCH_DATA="${VLA_TIDYBENCH_DATA:-${HOME}/data/vla-tidybench}"
DATASET=${DATASET:-${VLA_TIDYBENCH_DATA}/raw/stack_scripted.hdf5}

mkdir -p "$(dirname "$DATASET")" "${VLA_TIDYBENCH_DATA}/logs"

exec "$PROJECT_ROOT/scripts/run_isaac.sh" "$PROJECT_ROOT/scripts/collect_scripted_stack.py" \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 \
  --dataset_file "$DATASET" \
  --num_demos "$NUM_DEMOS" \
  --max_attempts "$MAX_ATTEMPTS" \
  --max_steps 520 \
  --seed "$SEED" \
  --enable_cameras \
  --device cuda:0 \
  --viz none \
  "$@"
