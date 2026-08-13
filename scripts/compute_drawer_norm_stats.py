#!/usr/bin/env python3
"""Run OpenPI's norm-stat computation for the drawer config."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from vla_tidybench.openpi.drawer_config import make_config

OPENPI_SCRIPT = Path("/home/ubuntu/openpi/scripts/compute_norm_stats.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("openpi_compute_norm_stats", OPENPI_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {OPENPI_SCRIPT}")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    config = make_config()
    official._config._CONFIGS_DICT[config.name] = config
    official.main(config.name, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
