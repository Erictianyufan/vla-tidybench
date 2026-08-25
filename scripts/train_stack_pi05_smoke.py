#!/usr/bin/env python3
"""Run a bounded pi0.5 LoRA training smoke test with the project config."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from vla_tidybench.openpi.stack_config import make_config


def openpi_train_script() -> Path:
    return Path(os.environ.get("OPENPI_ROOT", Path.home() / "openpi")) / "scripts" / "train.py"


def load_official_train():
    train_script = openpi_train_script()
    spec = importlib.util.spec_from_file_location("openpi_train", train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {train_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--exp-name", default="smoke")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.fsdp_devices < 1:
        parser.error("steps, batch-size and fsdp-devices must be positive")
    if args.batch_size % args.fsdp_devices:
        parser.error("batch-size must be divisible by fsdp-devices")

    config = make_config(
        exp_name=args.exp_name,
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
        overwrite=args.overwrite,
    )
    load_official_train().main(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
