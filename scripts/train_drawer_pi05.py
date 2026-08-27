#!/usr/bin/env python3
"""Run bounded π0.5 LoRA, action-expert, or full fine-tuning."""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
from pathlib import Path

from vla_tidybench.openpi.gpu_preflight import selected_gpu_indices, wait_for_exclusive_gpus
from vla_tidybench.openpi.training_metrics import (
    JsonlTrainingMetrics,
    build_training_provenance,
    source_tree_fingerprint,
    validate_completed_training_run,
)


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

    max_used_mib = int(os.environ.get("PI05_GPU_PREFLIGHT_MAX_USED_MIB", "512"))
    timeout_s = float(os.environ.get("PI05_GPU_PREFLIGHT_TIMEOUT_S", "21600"))
    selected_gpus = selected_gpu_indices()
    if os.environ.get("PI05_SKIP_GPU_PREFLIGHT") != "1":
        wait_for_exclusive_gpus(
            selected_gpus,
            max_used_mib=max_used_mib,
            timeout_s=timeout_s,
        )

    from vla_tidybench.openpi.drawer_config import make_config as make_open_config
    from vla_tidybench.openpi.drawer_four_skill_config import make_config as make_four_skill_config

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

    project_root = Path(__file__).resolve().parents[1]
    train_script = openpi_train_script()
    training_provenance = build_training_provenance(project_root, train_script.parents[1])
    init_params_files = None
    init_params_sha256 = None
    if args.init_params:
        init_params_files, init_params_sha256 = source_tree_fingerprint(
            args.init_params, (Path("."),)
        )
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
    metrics_path = Path(
        os.environ.get("PI05_TRAIN_METRICS_PATH", Path(str(config.checkpoint_dir)) / "train_metrics.jsonl")
    )
    metric_logger = None
    metric_metadata = {
        "config": config.name,
        "experiment": config.exp_name,
        "mode": args.mode,
        "optimizer": args.optimizer or ("adafactor" if args.mode == "full" else "adamw"),
        "dataset_repo": getattr(config.data, "repo_id", "fake"),
        "init_params": str(args.init_params.resolve()) if args.init_params else None,
        "num_train_steps": config.num_train_steps,
        "batch_size": config.batch_size,
        "fsdp_devices": config.fsdp_devices,
        "cuda_visible_devices": list(selected_gpus),
        "peak_lr": args.peak_lr,
        "warmup_steps": args.warmup_steps,
        "save_interval": args.save_interval,
        "fsdp_min_size_mbytes": args.fsdp_min_size_mbytes,
        "synthetic_data": args.synthetic_data,
        "init_params_files": init_params_files,
        "init_params_sha256": init_params_sha256,
        **training_provenance,
    }
    original_wandb_log = official.wandb.log

    def log_locally(payload, *positional, **keywords):
        nonlocal metric_logger
        if metric_logger is None:
            # OpenPI creates, resumes, or wipes checkpoint_dir before its first
            # wandb.log call, so lazy construction observes the final run state.
            metric_logger = JsonlTrainingMetrics(metrics_path, metric_metadata)
        step = keywords.get("step", positional[0] if positional else None)
        metric_logger.log(payload, step=step)
        return original_wandb_log(payload, *positional, **keywords)

    official.wandb.log = log_locally
    official.main(config)
    verified = validate_completed_training_run(
        Path(str(config.checkpoint_dir)),
        num_train_steps=config.num_train_steps,
        dataset_repo=str(metric_metadata["dataset_repo"]),
        metrics_path=metrics_path,
    )
    print("training_artifacts", json.dumps(verified, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
