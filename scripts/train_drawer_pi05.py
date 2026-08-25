#!/usr/bin/env python3
"""Run bounded π0.5 LoRA, action-expert, or full fine-tuning."""

from __future__ import annotations

import argparse
import functools
import importlib.util
import os
from pathlib import Path

from vla_tidybench.openpi.drawer_config import make_config as make_open_config
from vla_tidybench.openpi.drawer_four_skill_config import make_config as make_four_skill_config


def openpi_train_script() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return Path(os.environ.get("OPENPI_ROOT", project_root.parent / "openpi")) / "scripts" / "train.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("lora", "expert", "full"), default="lora")
    parser.add_argument("--optimizer", choices=("adamw", "adafactor"))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--fsdp-devices", type=int, default=3)
    parser.add_argument(
        "--fsdp-min-size-mbytes",
        type=float,
        default=float(os.environ["PI05_FSDP_MIN_SIZE_MBYTES"])
        if "PI05_FSDP_MIN_SIZE_MBYTES" in os.environ
        else None,
        help="override OpenPI's 4-MiB replication threshold; use 0 for shard-all full tuning",
    )
    parser.add_argument("--peak-lr", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--exp-name", default="smoke")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--four-skill", action="store_true")
    parser.add_argument("--dataset-repo", help="override the LeRobot dataset repository")
    parser.add_argument("--init-params", type=Path, help="Orbax params used to initialize this stage")
    parser.add_argument(
        "--synthetic-data",
        action="store_true",
        help="use OpenPI fake tensors for a systems smoke test only",
    )
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.fsdp_devices) < 1:
        parser.error("steps, batch-size and fsdp-devices must be positive")
    if args.batch_size % args.fsdp_devices:
        parser.error("batch-size must be divisible by fsdp-devices")
    if args.fsdp_min_size_mbytes is not None and args.fsdp_min_size_mbytes < 0:
        parser.error("fsdp-min-size-mbytes must be non-negative")
    if args.overwrite and args.resume:
        parser.error("overwrite and resume are mutually exclusive")

    if args.dataset_repo:
        env_name = (
            "VLA_TIDYBENCH_DRAWER_FOUR_SKILL_REPO_ID"
            if args.four_skill
            else "VLA_TIDYBENCH_DRAWER_REPO_ID"
        )
        os.environ[env_name] = args.dataset_repo
    if args.init_params:
        if not args.init_params.is_dir():
            parser.error(f"initial parameter directory does not exist: {args.init_params}")
        os.environ["PI05_CHECKPOINT_PARAMS"] = str(args.init_params.resolve())

    train_script = openpi_train_script()
    spec = importlib.util.spec_from_file_location("openpi_train", train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {train_script}")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    if args.fsdp_min_size_mbytes is not None:
        official.sharding.fsdp_sharding = functools.partial(
            official.sharding.fsdp_sharding,
            min_size_mbytes=args.fsdp_min_size_mbytes,
        )
    make_config = make_four_skill_config if args.four_skill else make_open_config
    config = make_config(
        exp_name=args.exp_name,
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
        finetune_mode=args.mode,
        optimizer_name=args.optimizer,
        peak_lr=args.peak_lr,
        warmup_steps=args.warmup_steps,
        save_interval=args.save_interval,
        synthetic_data=args.synthetic_data,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    official.main(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
