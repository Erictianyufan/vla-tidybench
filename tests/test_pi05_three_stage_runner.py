from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_pi05_three_stage", ROOT / "scripts/run_pi05_three_stage.py")
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def write_checkpoint(path: Path, *, complete: bool = True) -> None:
    (path / "params").mkdir(parents=True)
    (path / "_CHECKPOINT_METADATA").touch()
    (path / "params" / "_METADATA").touch()
    if complete:
        (path / "params" / "manifest.ocdbt").touch()


def test_resume_selects_latest_complete_numeric_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "stage2-full"
    write_checkpoint(run_dir / "999")
    write_checkpoint(run_dir / "1999")
    write_checkpoint(run_dir / "2999", complete=False)
    assert runner.resumable_checkpoint(run_dir) == run_dir / "1999"


def test_checkpoint_completion_requires_orbax_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "2999"
    write_checkpoint(checkpoint, complete=False)
    assert runner.checkpoint_complete(checkpoint) is False
    (checkpoint / "params" / "manifest.ocdbt").touch()
    assert runner.checkpoint_complete(checkpoint) is True


def test_train_wrapper_installs_local_metric_logging() -> None:
    source = (ROOT / "scripts" / "train_drawer_pi05.py").read_text(encoding="utf-8")
    assert "JsonlTrainingMetrics" in source
    assert '"train_metrics.jsonl"' in source
    assert '"num_train_steps": config.num_train_steps' in source
    assert "official.wandb.log = log_locally" in source
