#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLA_TIDYBENCH_DATA="${VLA_TIDYBENCH_DATA:-${HOME}/data/vla-tidybench}"
DATA_ROOT="${DATA_ROOT:-${VLA_TIDYBENCH_DATA}/raw}"
cd "${PROJECT_ROOT}"

for skill in open pick close; do
  echo "===SKILL:${skill}==="
  dataset="${DATA_ROOT}/drawer_${skill}_smoke.hdf5"
  rm -f "${dataset}"
  SKILL="${skill}" NUM_DEMOS=1 MAX_ATTEMPTS=3 MAX_STEPS=500 DATASET_FILE="${dataset}" \
    ./scripts/collect_scripted_drawer.sh --overwrite
done
