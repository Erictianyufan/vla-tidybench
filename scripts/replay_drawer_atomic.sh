#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
data_root=${DATA_ROOT:-/home/ubuntu/data/vla-tidybench/raw}
cd "$repo_root"

make drawer-replay SKILL=open DATASET_FILE="$data_root/drawer_open_smoke.hdf5"
make drawer-replay SKILL=pick DATASET_FILE="$data_root/drawer_pick_smoke.hdf5"
make drawer-replay SKILL=place DATASET_FILE="$data_root/drawer_place_smoke.hdf5"
make drawer-replay SKILL=close DATASET_FILE="$data_root/drawer_close_smoke.hdf5"
