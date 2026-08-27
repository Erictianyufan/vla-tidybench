from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from vla_tidybench.openpi.training_metrics import JsonlTrainingMetrics

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_pi05_training", ROOT / "scripts" / "summarize_pi05_training.py"
)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary
SPEC.loader.exec_module(summary)


def metrics(loss: float) -> dict[str, float]:
    return {"loss": loss, "grad_norm": loss + 1.0, "param_norm": loss + 2.0}


def test_summary_uses_latest_record_for_replayed_resume_steps(tmp_path: Path) -> None:
    path = tmp_path / "train_metrics.jsonl"
    first = JsonlTrainingMetrics(path, {"experiment": "stage"})
    first.log(metrics(3.0), step=0)
    first.log(metrics(2.0), step=1)
    resumed = JsonlTrainingMetrics(path, {"experiment": "stage"})
    resumed.log(metrics(1.5), step=1)
    resumed.log(metrics(1.0), step=2)

    report = summary.summarize_stage("stage", path, expected_steps=3)

    assert report["available"] is True
    assert report["raw_records"] == 4
    assert report["unique_steps"] == 3
    assert report["sessions"] == 2
    assert report["first_step"] == 0
    assert report["last_step"] == 2
    assert report["final_step_present"] is True
    assert report["recovered_steps"] == 0
    assert report["native_steps"] == 3
    assert report["completion_report_available"] is False
    assert report["loss_last"] == 1.0
    assert report["loss_min"] == 1.0
    assert report["grad_norm_max"] == 4.0
    assert report["param_norm_last"] == 3.0


def test_missing_metrics_are_reported_without_claiming_completion(tmp_path: Path) -> None:
    report = summary.summarize_stage("stage", tmp_path / "missing.jsonl", expected_steps=5_000)

    assert report["available"] is False
    assert report["expected_steps"] == 5_000
    assert report["final_step_present"] is False


def test_recovered_console_steps_are_disclosed(tmp_path: Path) -> None:
    path = tmp_path / "train_metrics.jsonl"
    logger = JsonlTrainingMetrics(
        path,
        {"experiment": "stage1-lora", "recovered_from_console": True},
    )
    logger.log(metrics(1.5), step=0)

    report = summary.summarize_stage("stage1-lora", path, expected_steps=5_000)

    assert report["recovered_steps"] == 1
    assert report["native_steps"] == 0
    assert report["final_step_recovered"] is False
