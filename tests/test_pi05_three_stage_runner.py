from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from vla_tidybench.openpi.deployment import checkpoint_fingerprint

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


def test_stage_completion_requires_content_bound_report(tmp_path: Path) -> None:
    stage = runner.Stage(1, "stage", "lora", 3, 1e-5, 1, 1, "fake", tmp_path / "base")
    checkpoint = runner.final_checkpoint(tmp_path, stage)
    write_checkpoint(checkpoint)
    assert runner.stage_completion_verified(tmp_path, stage) is False

    file_count, byte_count, checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    report = {
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
    runner.training_completion_path(tmp_path, stage).write_text(json.dumps(report), encoding="utf-8")
    assert runner.stage_completion_verified(tmp_path, stage) is True


def test_stage_completion_rejects_checkpoint_tampering(tmp_path: Path) -> None:
    stage = runner.Stage(1, "stage", "lora", 3, 1e-5, 1, 1, "fake", tmp_path / "base")
    checkpoint = runner.final_checkpoint(tmp_path, stage)
    write_checkpoint(checkpoint)
    file_count, byte_count, checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    report = {
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
    runner.training_completion_path(tmp_path, stage).write_text(json.dumps(report), encoding="utf-8")
    (checkpoint / "params" / "manifest.ocdbt").write_text("tampered", encoding="utf-8")
    try:
        runner.stage_completion_verified(tmp_path, stage)
    except ValueError as error:
        assert "checkpoint SHA-256" in str(error) or "inventory" in str(error)
    else:
        raise AssertionError("tampered final checkpoint was accepted")


def test_train_wrapper_installs_local_metric_logging() -> None:
    source = (ROOT / "scripts" / "train_drawer_pi05.py").read_text(encoding="utf-8")
    assert '"train_metrics.jsonl"' in source
    assert '"num_train_steps": config.num_train_steps' in source
    assert "install_wandb_metrics_after_init(" in source
    assert "validate_completed_training_run(" in source
    assert source.index("checkpoint_fingerprint(dataset_path)") < source.index(
        "wait_for_exclusive_gpus("
    )
    assert source.index("wait_for_exclusive_gpus(") < source.index(
        "from vla_tidybench.openpi.drawer_config import"
    )
