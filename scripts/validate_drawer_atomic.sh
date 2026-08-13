#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/data/vla-tidybench/raw}"
cd "${PROJECT_ROOT}"

for skill in open pick close; do
  echo "===SKILL:${skill}==="
  dataset="${DATA_ROOT}/drawer_${skill}_smoke.hdf5"
  rm -f "${dataset}"
  SKILL="${skill}" NUM_DEMOS=1 MAX_ATTEMPTS=3 MAX_STEPS=500 DATASET_FILE="${dataset}" \
    ./scripts/collect_scripted_drawer.sh --overwrite
done
