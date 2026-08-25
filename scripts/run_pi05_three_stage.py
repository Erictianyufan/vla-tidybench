#!/usr/bin/env python3
"""Run the guarded three-stage pi0.5 fine-tuning experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import os
from pathlib import Path
import subprocess


MVP_REPO = "erictianyufan/vla_tidybench_drawer_four_skill_mvp"
FULL_CONFIG_NAME = "pi05_tidybench_drawer_four_skill_full"
LORA_CONFIG_NAME = "pi05_tidybench_drawer_four_skill_lora"


@dataclass(frozen=True)
class Stage:
    number: int
    name: str
    mode: str
    steps: int
    peak_lr: float
    warmup_steps: int
    save_interval: int
    dataset_repo: str
    init_params: Path


def default_data_root() -> Path:
    return Path(os.environ.get("VLA_TIDYBENCH_DATA", f"/data/{os.environ.get('USER', 'user')}/vla-tidybench"))


def parse_args() -> argparse.Namespace:
    data_root = default_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument("--main-dataset-repo", default=MVP_REPO)
    parser.add_argument("--hard-dataset-repo")
    parser.add_argument(
        "--base-params",
        type=Path,
        default=data_root / "checkpoints" / "openpi-assets" / "checkpoints" / "pi05_droid" / "params",
    )
    parser.add_argument("--stage2-params", type=Path)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--fsdp-devices", type=int, default=3)
    parser.add_argument("--smoke", action="store_true", help="run two optimizer steps per selected stage")
    parser.add_argument("--synthetic-data", action="store_true", help="systems smoke only; no task metric is valid")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.fsdp_devices) < 1:
        parser.error("batch-size and fsdp-devices must be positive")
    if args.batch_size % args.fsdp_devices:
        parser.error("batch-size must be divisible by fsdp-devices")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.synthetic_data and not args.smoke:
        parser.error("--synthetic-data is permitted only together with --smoke")
    if not args.smoke and args.main_dataset_repo == MVP_REPO:
        parser.error("formal training is blocked on the four-episode MVP dataset; provide --main-dataset-repo")
    if args.stage in ("3", "all") and not args.smoke and not args.hard_dataset_repo:
        parser.error("stage 3 requires a replay-mixed hard-sample dataset via --hard-dataset-repo")
    return args


def final_stage2_params(data_root: Path, *, exp_name: str, steps: int) -> Path:
    return (
        data_root
        / "checkpoints"
        / "openpi-runs"
        / FULL_CONFIG_NAME
        / exp_name
        / str(steps - 1)
        / "params"
    )


def stage_run_dir(data_root: Path, stage: Stage) -> Path:
    config_name = LORA_CONFIG_NAME if stage.mode == "lora" else FULL_CONFIG_NAME
    return data_root / "checkpoints" / "openpi-runs" / config_name / stage.name


def final_checkpoint(data_root: Path, stage: Stage) -> Path:
    return stage_run_dir(data_root, stage) / str(stage.steps - 1)


def checkpoint_complete(path: Path) -> bool:
    return all(
        required.is_file()
        for required in (
            path / "_CHECKPOINT_METADATA",
            path / "params" / "_METADATA",
            path / "params" / "manifest.ocdbt",
        )
    )


def resumable_checkpoint(run_dir: Path) -> Path | None:
    if not run_dir.is_dir():
        return None
    candidates = sorted(
        (path for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
        reverse=True,
    )
    return next((path for path in candidates if checkpoint_complete(path)), None)


def build_stages(args: argparse.Namespace) -> list[Stage]:
    data_root = default_data_root()
    stage1 = Stage(1, "stage1-lora", "lora", 5_000, 2.5e-5, 500, 500, args.main_dataset_repo, args.base_params)
    stage2 = Stage(2, "stage2-full", "full", 10_000, 1.0e-5, 1_000, 1_000, args.main_dataset_repo, args.base_params)
    inferred_stage2 = final_stage2_params(data_root, exp_name=stage2.name, steps=stage2.steps)
    stage3 = Stage(
        3,
        "stage3-hard-recovery",
        "full",
        3_000,
        2.0e-6,
        300,
        300,
        args.hard_dataset_repo or args.main_dataset_repo,
        args.stage2_params or inferred_stage2,
    )
    stages = [stage1, stage2, stage3]
    if args.smoke:
        stages = [
            replace(
                stage,
                name=f"{stage.name}-smoke",
                steps=2,
                warmup_steps=1,
                save_interval=1,
            )
            for stage in stages
        ]
        smoke_stage2 = final_stage2_params(data_root, exp_name=stages[1].name, steps=stages[1].steps)
        stages[2] = replace(stages[2], init_params=args.stage2_params or smoke_stage2)
    if args.stage != "all":
        stages = [stages[int(args.stage) - 1]]
    return stages


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "scripts" / "train_drawer_pi05.py"
    runner = project_root / "scripts" / "run_openpi.sh"
    stages = build_stages(args)
    data_root = default_data_root()
    for stage in stages:
        run_dir = stage_run_dir(data_root, stage)
        completed = checkpoint_complete(final_checkpoint(data_root, stage))
        if args.resume and completed:
            print(
                f"stage={stage.number} status=complete action=skip "
                f"checkpoint={final_checkpoint(data_root, stage)}",
                flush=True,
            )
            continue
        resume_from = resumable_checkpoint(run_dir) if args.resume else None
        if args.resume and run_dir.exists() and resume_from is None:
            raise RuntimeError(
                f"stage {stage.number} run directory exists but contains no complete checkpoint: {run_dir}; "
                "inspect it or restart explicitly with --overwrite"
            )
        if not args.dry_run and not stage.init_params.is_dir():
            raise FileNotFoundError(f"stage {stage.number} init params not found: {stage.init_params}")
        command = [
            str(runner),
            str(train_script),
            "--four-skill",
            "--mode",
            stage.mode,
            "--steps",
            str(stage.steps),
            "--batch-size",
            str(args.batch_size),
            "--fsdp-devices",
            str(args.fsdp_devices),
            "--peak-lr",
            str(stage.peak_lr),
            "--warmup-steps",
            str(stage.warmup_steps),
            "--save-interval",
            str(stage.save_interval),
            "--exp-name",
            stage.name,
            "--dataset-repo",
            stage.dataset_repo,
            "--init-params",
            str(stage.init_params),
        ]
        if resume_from is not None:
            command.append("--resume")
        elif args.overwrite or args.smoke:
            command.append("--overwrite")
        if args.synthetic_data:
            command.append("--synthetic-data")
        if stage.mode == "full":
            command.extend(("--optimizer", "adafactor", "--fsdp-min-size-mbytes", "0"))
        print(
            f"stage={stage.number} mode={stage.mode} steps={stage.steps} "
            f"dataset={stage.dataset_repo} init={stage.init_params} "
            f"action={'resume' if resume_from is not None else 'start'}"
            + (f" resume_from={resume_from}" if resume_from is not None else ""),
            flush=True,
        )
        print("command:", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
