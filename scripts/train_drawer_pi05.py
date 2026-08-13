#!/usr/bin/env python3
"""Run bounded π0.5 LoRA training on the drawer dataset."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from vla_tidybench.openpi.drawer_config import make_config


OPENPI_TRAIN = Path("/home/ubuntu/openpi/scripts/train.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--fsdp-devices", type=int, default=2)
    parser.add_argument("--exp-name", default="smoke")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.fsdp_devices) < 1:
        parser.error("steps, batch-size and fsdp-devices must be positive")
    if args.batch_size % args.fsdp_devices:
        parser.error("batch-size must be divisible by fsdp-devices")
    if args.overwrite and args.resume:
        parser.error("overwrite and resume are mutually exclusive")

    spec = importlib.util.spec_from_file_location("openpi_train", OPENPI_TRAIN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {OPENPI_TRAIN}")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    config = make_config(
        exp_name=args.exp_name,
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    official.main(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
