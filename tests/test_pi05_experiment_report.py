from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from vla_tidybench.openpi.deployment import checkpoint_fingerprint

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "report_pi05_experiment", ROOT / "scripts" / "report_pi05_experiment.py"
)
assert SPEC is not None and SPEC.loader is not None
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


def write_checkpoint(path: Path, *, complete: bool = True) -> None:
    (path / "params").mkdir(parents=True)
    (path / "_CHECKPOINT_METADATA").touch()
    (path / "params" / "_METADATA").touch()
    if complete:
        (path / "params" / "manifest.ocdbt").touch()


def test_stage_status_reports_only_complete_numeric_checkpoints(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "config" / "stage"
    write_checkpoint(run_dir / "499")
    write_checkpoint(run_dir / "999", complete=False)
    (run_dir / "train_metrics.jsonl").touch()

    status = reporter.stage_status(run_root, "stage", "config", 1_000)

    assert status["latest_complete_checkpoint"].endswith("499")
    assert status["final_checkpoint_complete"] is False
    assert status["metrics_available"] is True
    assert status["metrics_records"] == 0
    assert status["training_completion_available"] is False
    assert status["training_completion_verified"] is False


def test_stage_status_verifies_content_bound_completion(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "config" / "stage"
    checkpoint = run_dir / "2"
    write_checkpoint(checkpoint)
    file_count, byte_count, checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    (run_dir / "training_completion.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "verified": True,
                "checkpoint": str(checkpoint.resolve()),
                "final_step": 2,
                "num_train_steps": 3,
                "dataset_repo": "fake",
                "checkpoint_digest_algorithm": "sha256-tree-v1",
                "checkpoint_file_count": file_count,
                "checkpoint_byte_count": byte_count,
                "checkpoint_sha256": checkpoint_sha256,
                "loss": 1.0,
                "grad_norm": 2.0,
                "param_norm": 3.0,
            }
        ),
        encoding="utf-8",
    )

    status = reporter.stage_status(run_root, "stage", "config", 3, expected_dataset_repo="fake")

    assert status["final_checkpoint_complete"] is True
    assert status["training_completion_available"] is True
    assert status["training_completion_verified"] is True
    assert "training_completion_error" not in status


def test_stage_status_rejects_tampered_completed_checkpoint(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "config" / "stage"
    checkpoint = run_dir / "2"
    write_checkpoint(checkpoint)
    file_count, byte_count, checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    (run_dir / "training_completion.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "verified": True,
                "checkpoint": str(checkpoint.resolve()),
                "final_step": 2,
                "num_train_steps": 3,
                "dataset_repo": "fake",
                "checkpoint_digest_algorithm": "sha256-tree-v1",
                "checkpoint_file_count": file_count,
                "checkpoint_byte_count": byte_count,
                "checkpoint_sha256": checkpoint_sha256,
                "loss": 1.0,
                "grad_norm": 2.0,
                "param_norm": 3.0,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "params" / "manifest.ocdbt").write_text("tampered", encoding="utf-8")

    status = reporter.stage_status(run_root, "stage", "config", 3, expected_dataset_repo="fake")

    assert status["training_completion_available"] is True
    assert status["training_completion_verified"] is False
    assert "training_completion_error" in status


def test_progress_parser_uses_latest_tqdm_record() -> None:
    text = (
        "Progress on: 160it/5.00kit rate:117.3s/it remaining:x\n"
        "Progress on: 161it/5.00kit rate:128.9s/it remaining:y\n"
    )

    progress = reporter.parse_latest_progress(text)

    assert progress == {
        "step": 161,
        "total_steps_display": 5_000,
        "seconds_per_step": 128.9,
        "estimated_remaining_seconds": 623_747,
        "estimated_remaining_hours": 173.3,
    }


def test_metric_coverage_discloses_recovered_and_native_records(tmp_path: Path) -> None:
    path = tmp_path / "train_metrics.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"step": 0, "recovered_from_console": True},
                {"step": 1, "recovered_from_console": True},
                {"step": 499, "recovered_from_console": False},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    coverage = reporter.metric_coverage(path)

    assert coverage["metrics_highest_step"] == 499
    assert coverage["recovered_metrics_records"] == 2
    assert coverage["native_metrics_records"] == 1


def test_metric_coverage_ignores_one_inflight_partial_line(tmp_path: Path) -> None:
    path = tmp_path / "train_metrics.jsonl"
    path.write_text('{"step": 2}\n{"step":', encoding="utf-8")

    coverage = reporter.metric_coverage(path)

    assert "metrics_error" not in coverage
    assert coverage["metrics_records"] == 1


def test_gpu_report_separates_training_and_foreign_pids(monkeypatch) -> None:
    usage = {
        0: reporter.GPUUsage(0, "GPU-a", 13_000, (100, 200)),
        1: reporter.GPUUsage(1, "GPU-b", 10_000, (100,)),
    }
    monkeypatch.setattr(
        reporter,
        "process_commands",
        lambda _pids: {
            100: "python /repo/vla-tidybench/scripts/train_drawer_pi05.py",
            200: "python unrelated.py",
        },
    )

    report = reporter.gpu_report((0, 1), usage)

    assert report["training_pids"] == [100]
    assert report["foreign_compute_pids"] == [200]
    assert report["resource_conflict"] is True
