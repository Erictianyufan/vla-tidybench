"""Append-only local metrics for OpenPI runs with external tracking disabled."""

from __future__ import annotations

import datetime as dt
import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vla_tidybench.openpi.deployment import REQUIRED_CHECKPOINT_FILES, checkpoint_asset_id

METRIC_KEYS = ("loss", "grad_norm", "param_norm")


@dataclass
class JsonlTrainingMetrics:
    path: Path
    run_metadata: Mapping[str, Any]
    _previous_last_step: int = field(init=False, default=-1)
    _session_last_step: int = field(init=False, default=-1)
    _session_id: str = field(init=False, default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        self.path = self.path.expanduser().resolve()
        if not self.path.is_file():
            return
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                record = json.loads(line)
                self._previous_last_step = int(record["step"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid training metric at {self.path}:{number}") from error

    def log(self, payload: Mapping[str, Any], *, step: object | None) -> bool:
        """Append one numeric training record; ignore non-metric WandB payloads."""

        if step is None or not all(key in payload for key in METRIC_KEYS):
            return False
        numeric_step = int(step)
        if numeric_step <= self._session_last_step:
            raise ValueError(
                f"training metric step {numeric_step} is not newer than current-session step "
                f"{self._session_last_step}"
            )
        metrics = {key: float(payload[key]) for key in METRIC_KEYS}
        nonfinite = [key for key, value in metrics.items() if not math.isfinite(value)]
        if nonfinite:
            raise FloatingPointError(f"non-finite training metrics at step {numeric_step}: {nonfinite}")
        record = {
            "schema_version": 1,
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "session_id": self._session_id,
            "previous_metrics_step": self._previous_last_step,
            "step": numeric_step,
            **dict(self.run_metadata),
            **metrics,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._session_last_step = numeric_step
        return True


def validate_completed_training_run(
    run_dir: Path,
    *,
    num_train_steps: int,
    dataset_repo: str,
    metrics_path: Path,
) -> dict[str, object]:
    """Verify the durable artifacts required before a training stage may exit."""

    if num_train_steps < 1:
        raise ValueError("num_train_steps must be positive")
    final_step = num_train_steps - 1
    checkpoint = run_dir.expanduser().resolve() / str(final_step)
    missing = [
        str(checkpoint / relative)
        for relative in REQUIRED_CHECKPOINT_FILES
        if not (checkpoint / relative).is_file()
    ]
    if missing:
        raise ValueError("final checkpoint is incomplete; missing: " + ", ".join(missing))
    asset_id = None if dataset_repo == "fake" else checkpoint_asset_id(checkpoint)
    if asset_id is not None and asset_id != dataset_repo:
        raise ValueError(
            f"final checkpoint normalization asset ID {asset_id!r} does not match dataset {dataset_repo!r}"
        )

    metrics_path = metrics_path.expanduser().resolve()
    if not metrics_path.is_file():
        raise ValueError(f"training metrics are missing: {metrics_path}")
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"training metrics are empty: {metrics_path}")
    try:
        final_metrics = json.loads(lines[-1])
        metric_step = int(final_metrics["step"])
        metric_steps = int(final_metrics["num_train_steps"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid final training metric in {metrics_path}") from error
    if metric_step != final_step or metric_steps != num_train_steps:
        raise ValueError(
            f"final training metric step mismatch: got {metric_step}/{metric_steps}, "
            f"expected {final_step}/{num_train_steps}"
        )
    if final_metrics.get("dataset_repo") != dataset_repo:
        raise ValueError("final training metric dataset does not match the completed stage")
    nonfinite = [
        key for key in METRIC_KEYS if not math.isfinite(float(final_metrics.get(key, float("nan"))))
    ]
    if nonfinite:
        raise ValueError(f"final training metrics are non-finite: {nonfinite}")
    return {
        "checkpoint": str(checkpoint),
        "final_step": final_step,
        "dataset_repo": dataset_repo,
        "checkpoint_asset_id": asset_id,
        "metrics_path": str(metrics_path),
        "loss": float(final_metrics["loss"]),
        "grad_norm": float(final_metrics["grad_norm"]),
        "param_norm": float(final_metrics["param_norm"]),
        "verified": True,
    }
