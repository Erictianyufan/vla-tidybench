from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


def test_progress_parser_uses_latest_tqdm_record() -> None:
    text = (
        "Progress on: 160it/5.00kit rate:117.3s/it remaining:x\n"
        "Progress on: 161it/5.00kit rate:128.9s/it remaining:y\n"
    )

    progress = reporter.parse_latest_progress(text)

    assert progress == {"step": 161, "total_steps_display": 5_000, "seconds_per_step": 128.9}


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
