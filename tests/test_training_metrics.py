from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from vla_tidybench.openpi.training_metrics import JsonlTrainingMetrics


def payload(*, loss: float = 1.5) -> dict[str, np.float32]:
    return {
        "loss": np.float32(loss),
        "grad_norm": np.float32(2.5),
        "param_norm": np.float32(3.5),
    }


def records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_numeric_metrics_are_appended_with_run_provenance(tmp_path: Path) -> None:
    path = tmp_path / "train_metrics.jsonl"
    logger = JsonlTrainingMetrics(path, {"experiment": "stage2-full", "mode": "full"})
    assert logger.log({"camera_views": object()}, step=0) is False
    assert logger.log(payload(), step=0) is True

    [record] = records(path)
    assert record["schema_version"] == 1
    assert record["step"] == 0
    assert record["experiment"] == "stage2-full"
    assert record["mode"] == "full"
    assert record["loss"] == pytest.approx(1.5)
    assert isinstance(record["session_id"], str)


def test_resume_session_can_replay_steps_after_latest_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "train_metrics.jsonl"
    first = JsonlTrainingMetrics(path, {"experiment": "stage"})
    first.log(payload(loss=2.0), step=1000)
    first.log(payload(loss=1.9), step=1001)

    resumed = JsonlTrainingMetrics(path, {"experiment": "stage"})
    resumed.log(payload(loss=1.8), step=1000)
    history = records(path)
    assert [record["step"] for record in history] == [1000, 1001, 1000]
    assert history[-1]["previous_metrics_step"] == 1001
    assert history[-1]["session_id"] != history[0]["session_id"]


def test_nonfinite_or_nonmonotonic_session_metrics_fail_fast(tmp_path: Path) -> None:
    logger = JsonlTrainingMetrics(tmp_path / "metrics.jsonl", {"experiment": "stage"})
    with pytest.raises(FloatingPointError, match="non-finite"):
        logger.log(payload(loss=float("nan")), step=0)
    logger.log(payload(), step=1)
    with pytest.raises(ValueError, match="current-session"):
        logger.log(payload(), step=1)


def test_corrupt_existing_metrics_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid training metric"):
        JsonlTrainingMetrics(path, {"experiment": "stage"})
