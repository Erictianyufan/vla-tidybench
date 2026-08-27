#!/usr/bin/env python3
"""Report three-stage checkpoint, log, process, and GPU resource state."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path

from vla_tidybench.openpi.deployment import REQUIRED_CHECKPOINT_FILES
from vla_tidybench.openpi.gpu_preflight import GPUUsage, inspect_gpu_usage

STAGES = (
    ("stage1-lora", "pi05_tidybench_drawer_four_skill_lora", 5_000),
    ("stage2-full", "pi05_tidybench_drawer_four_skill_full", 10_000),
    ("stage3-hard-recovery", "pi05_tidybench_drawer_four_skill_full", 3_000),
)
PROGRESS_PATTERN = re.compile(
    r"Progress on:\s*(?P<step>[0-9]+)it/(?P<total>[0-9.]+[kM]?)it"
    r"\s+rate:(?P<rate>[0-9.]+)s/it"
)
ERROR_PATTERN = re.compile(r"traceback|exception|out of memory|\bnan\b|killed", re.IGNORECASE)


def data_root() -> Path:
    return Path(os.environ.get("VLA_TIDYBENCH_DATA", f"/data/{os.environ.get('USER', 'user')}/vla-tidybench"))


def checkpoint_complete(path: Path) -> bool:
    return all((path / relative).is_file() for relative in REQUIRED_CHECKPOINT_FILES)


def stage_status(run_root: Path, name: str, config: str, steps: int) -> dict[str, object]:
    run_dir = run_root / config / name
    numeric = sorted(
        (path for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit())
        if run_dir.is_dir()
        else (),
        key=lambda path: int(path.name),
    )
    complete = [path for path in numeric if checkpoint_complete(path)]
    final = run_dir / str(steps - 1)
    metrics = run_dir / "train_metrics.jsonl"
    return {
        "name": name,
        "config": config,
        "expected_steps": steps,
        "run_dir": str(run_dir.resolve()),
        "latest_complete_checkpoint": str(complete[-1].resolve()) if complete else None,
        "final_checkpoint_complete": checkpoint_complete(final),
        "metrics_available": metrics.is_file(),
    }


def parse_latest_progress(log_text: str) -> dict[str, object] | None:
    matches = list(PROGRESS_PATTERN.finditer(log_text))
    if not matches:
        return None
    match = matches[-1]
    suffix = match["total"][-1]
    scale = {"k": 1_000, "M": 1_000_000}.get(suffix, 1)
    total_text = match["total"][:-1] if scale != 1 else match["total"]
    return {
        "step": int(match["step"]),
        "total_steps_display": int(float(total_text) * scale),
        "seconds_per_step": float(match["rate"]),
    }


def process_commands(pids: set[int]) -> dict[int, str]:
    if not pids:
        return {}
    result = subprocess.run(
        ["ps", "-p", ",".join(str(pid) for pid in sorted(pids)), "-o", "pid=,cmd="],
        text=True,
        capture_output=True,
        check=False,
    )
    commands: dict[int, str] = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2:
            commands[int(fields[0])] = fields[1]
    return commands


def gpu_report(indices: tuple[int, ...], usage: dict[int, GPUUsage]) -> dict[str, object]:
    pids = {pid for index in indices if index in usage for pid in usage[index].compute_pids}
    commands = process_commands(pids)
    training_pids = sorted(
        pid
        for pid, command in commands.items()
        if "vla-tidybench/scripts/train_drawer_pi05.py" in command
    )
    foreign_pids = sorted(pid for pid in pids if pid not in training_pids)
    return {
        "selected_indices": list(indices),
        "training_pids": training_pids,
        "foreign_compute_pids": foreign_pids,
        "resource_conflict": bool(foreign_pids),
        "devices": [
            {
                "index": index,
                "memory_used_mib": usage[index].memory_used_mib,
                "compute_pids": list(usage[index].compute_pids),
            }
            for index in indices
            if index in usage
        ],
        "process_commands": {str(pid): commands.get(pid, "") for pid in sorted(pids)},
    }


def main() -> int:
    root = data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=root / "checkpoints" / "openpi-runs")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-conflict", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(index < 0 for index in args.gpus):
        parser.error("--gpus must contain unique non-negative indices")

    if args.log is None:
        candidates = sorted((root / "logs").glob("pi05-formal-three-stage-*.log"))
        log_path = candidates[-1] if candidates else None
    else:
        log_path = args.log.expanduser().resolve()
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path and log_path.is_file() else ""
    stages = [stage_status(args.run_root.expanduser().resolve(), *stage) for stage in STAGES]
    active = next((stage["name"] for stage in stages if not stage["final_checkpoint_complete"]), None)
    errors = [line.strip() for line in log_text.splitlines() if ERROR_PATTERN.search(line)][-20:]
    report = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "active_stage": active,
        "all_stages_complete": all(stage["final_checkpoint_complete"] for stage in stages),
        "stages": stages,
        "log": str(log_path.resolve()) if log_path else None,
        "latest_progress": parse_latest_progress(log_text),
        "error_signals": errors,
        "gpu": gpu_report(tuple(args.gpus), inspect_gpu_usage()),
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 2 if args.fail_on_conflict and report["gpu"]["resource_conflict"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
