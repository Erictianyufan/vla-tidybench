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
