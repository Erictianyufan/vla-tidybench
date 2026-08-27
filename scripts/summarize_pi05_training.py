#!/usr/bin/env python3
"""Summarize append-only local metrics from the guarded three-stage experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path

STAGES = (
    ("stage1-lora", "pi05_tidybench_drawer_four_skill_lora", 5_000),
    ("stage2-full", "pi05_tidybench_drawer_four_skill_full", 10_000),
    ("stage3-hard-recovery", "pi05_tidybench_drawer_four_skill_full", 3_000),
)
METRIC_KEYS = ("loss", "grad_norm", "param_norm")


def data_root() -> Path:
    return Path(os.environ.get("VLA_TIDYBENCH_DATA", f"/data/{os.environ.get('USER', 'user')}/vla-tidybench"))


def read_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{number}") from error
        if not isinstance(record, dict) or int(record.get("schema_version", -1)) != 1:
            raise ValueError(f"invalid metric schema at {path}:{number}")
        step = int(record.get("step", -1))
        if step < 0 or not isinstance(record.get("session_id"), str):
            raise ValueError(f"invalid metric step/session at {path}:{number}")
        for key in METRIC_KEYS:
            value = float(record.get(key, float("nan")))
            if not math.isfinite(value):
                raise ValueError(f"non-finite {key} at {path}:{number}")
        records.append(record)
    return records


def summarize_stage(name: str, path: Path, expected_steps: int) -> dict[str, object]:
    completion_path = path.parent / "training_completion.json"
    if not path.is_file():
        return {
            "name": name,
            "metrics_path": str(path.resolve()),
            "available": False,
            "expected_steps": expected_steps,
            "final_step_present": False,
            "completion_report": str(completion_path.resolve()),
            "completion_report_available": completion_path.is_file(),
        }
    records = read_records(path)
    latest_by_step = {int(record["step"]): record for record in records}
    effective = [latest_by_step[step] for step in sorted(latest_by_step)]
    losses = [float(record["loss"]) for record in effective]
    grad_norms = [float(record["grad_norm"]) for record in effective]
    param_norms = [float(record["param_norm"]) for record in effective]
    recovered = [record for record in effective if record.get("recovered_from_console") is True]
    native = [record for record in effective if record.get("recovered_from_console") is not True]
    final_step_present = bool(effective and int(effective[-1]["step"]) == expected_steps - 1)
    return {
        "name": name,
        "metrics_path": str(path.resolve()),
        "available": True,
        "expected_steps": expected_steps,
        "raw_records": len(records),
        "unique_steps": len(effective),
        "sessions": len({str(record["session_id"]) for record in records}),
        "recovered_steps": len(recovered),
        "native_steps": len(native),
        "first_step": int(effective[0]["step"]) if effective else None,
        "last_step": int(effective[-1]["step"]) if effective else None,
        "final_step_present": final_step_present,
        "final_step_recovered": bool(
            final_step_present and effective[-1].get("recovered_from_console") is True
        ),
        "completion_report": str(completion_path.resolve()),
        "completion_report_available": completion_path.is_file(),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "grad_norm_max": max(grad_norms) if grad_norms else None,
        "param_norm_last": param_norms[-1] if param_norms else None,
    }


def main() -> int:
    root = data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=root / "checkpoints" / "openpi-runs")
    parser.add_argument("--output", type=Path, default=root / "logs" / "pi05-three-stage-metrics-summary.json")
    parser.add_argument("--require-final-step", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    stages = [
        summarize_stage(name, run_root / config / name / "train_metrics.jsonl", expected_steps)
        for name, config, expected_steps in STAGES
    ]
    report = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "run_root": str(run_root),
        "all_final_steps_present": all(stage["final_step_present"] for stage in stages),
        "all_completion_reports_present": all(
            stage["completion_report_available"] for stage in stages
        ),
        "stages": stages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if args.require_final_step and not report["all_final_steps_present"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
