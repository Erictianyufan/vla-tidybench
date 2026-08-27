#!/usr/bin/env python3
"""Recover numeric OpenPI metrics from a captured console log."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import uuid
from pathlib import Path

NUMBER = r"[-+]?(?:\d+\.\d{4}|\d+(?:[eE][-+]?\d+))"
METRIC_PATTERN = re.compile(
    rf"Step\s+(?P<step>\d+):\s+grad_norm=(?P<grad>{NUMBER}),\s+"
    rf"loss=(?P<loss>{NUMBER}),\s+param_norm=(?P<param>{NUMBER})"
)


def parse_console_metrics(text: str) -> list[dict[str, float | int]]:
    """Return the last complete four-decimal metric tuple for every console step."""

    latest: dict[int, dict[str, float | int]] = {}
    for match in METRIC_PATTERN.finditer(text):
        step = int(match.group("step"))
        latest[step] = {
            "step": step,
            "loss": float(match.group("loss")),
            "grad_norm": float(match.group("grad")),
            "param_norm": float(match.group("param")),
        }
    return [latest[step] for step in sorted(latest)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("lora", "expert", "full"), required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--num-train-steps", type=int, required=True)
    parser.add_argument("--project-commit")
    args = parser.parse_args()
    if args.num_train_steps < 1:
        parser.error("num-train-steps must be positive")

    source = args.log.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to replace existing metrics: {output}")
    raw = source.read_bytes()
    parsed = parse_console_metrics(raw.decode("utf-8", errors="replace"))
    if not parsed:
        parser.error(f"no complete console metrics found in {source}")
    if int(parsed[-1]["step"]) >= args.num_train_steps:
        parser.error("recovered metric step exceeds configured training length")

    recovered_at = dt.datetime.now(dt.UTC).isoformat()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    session_id = f"console-recovery-{uuid.uuid4().hex}"
    records = []
    previous = -1
    for metric in parsed:
        record = {
            "schema_version": 1,
            "created_at_utc": recovered_at,
            "session_id": session_id,
            "previous_metrics_step": previous,
            "experiment": args.experiment,
            "config": args.config,
            "mode": args.mode,
            "dataset_repo": args.dataset_repo,
            "num_train_steps": args.num_train_steps,
            "project_commit": args.project_commit,
            "project_dirty": None,
            "recovered_from_console": True,
            "source_log": str(source),
            "source_log_bytes": len(raw),
            "source_log_sha256": source_sha256,
            **metric,
        }
        records.append(record)
        previous = int(metric["step"])

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "records": len(records),
                "first_step": records[0]["step"],
                "last_step": records[-1]["step"],
                "source_log_sha256": source_sha256,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
