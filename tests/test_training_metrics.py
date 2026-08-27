from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from vla_tidybench.openpi.deployment import checkpoint_fingerprint
from vla_tidybench.openpi.training_metrics import (
    JsonlTrainingMetrics,
    lerobot_dataset_path,
    source_tree_fingerprint,
    validate_completed_training_run,
    validate_dataset_fingerprint,
    write_training_completion,
)

PROJECT_COMMIT = "d" * 40


def formal_metadata(dataset_repo: str, steps: int) -> dict[str, object]:
    return {
        "dataset_repo": dataset_repo,
        "num_train_steps": steps,
        "project_commit": PROJECT_COMMIT,
        "project_dirty": False,
        "openpi_source_files": 3,
        "openpi_source_sha256": "e" * 64,
        "init_params": "/checkpoints/base/params",
        "init_params_files": 2,
        "init_params_sha256": "f" * 64,
        "dataset_path": f"/datasets/{dataset_repo}",
        "dataset_digest_algorithm": "sha256-tree-v1",
        "dataset_files": 360,
        "dataset_bytes": 1_000_000,
        "dataset_sha256": "a" * 64,
    }


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


def write_complete_checkpoint(path: Path, dataset_repo: str) -> None:
    (path / "params").mkdir(parents=True)
    (path / "assets" / dataset_repo).mkdir(parents=True)
    (path / "_CHECKPOINT_METADATA").touch()
    (path / "params" / "_METADATA").touch()
    (path / "params" / "manifest.ocdbt").touch()
    (path / "assets" / dataset_repo / "norm_stats.json").write_text("{}", encoding="utf-8")


def test_completed_training_run_binds_checkpoint_metrics_and_assets(tmp_path: Path) -> None:
    dataset_repo = "owner/hard_mix"
    run_dir = tmp_path / "stage3"
    write_complete_checkpoint(run_dir / "2", dataset_repo)
    metrics_path = run_dir / "train_metrics.jsonl"
    logger = JsonlTrainingMetrics(
        metrics_path,
        formal_metadata(dataset_repo, 3),
    )
    for step in range(3):
        logger.log(payload(loss=3.0 - step), step=step)

    report = validate_completed_training_run(
        run_dir,
        num_train_steps=3,
        dataset_repo=dataset_repo,
        metrics_path=metrics_path,
    )

    assert report["verified"] is True
    assert report["final_step"] == 2
    assert report["checkpoint_asset_id"] == dataset_repo
    assert report["dataset_sha256"] == "a" * 64
    assert report["loss"] == pytest.approx(1.0)
    assert report["checkpoint_sha256"]
    completion_path = write_training_completion(run_dir, report)
    assert json.loads(completion_path.read_text(encoding="utf-8"))["verified"] is True


def test_completed_training_run_rejects_wrong_checkpoint_assets(tmp_path: Path) -> None:
    run_dir = tmp_path / "stage3"
    write_complete_checkpoint(run_dir / "0", "owner/wrong")
    metrics_path = run_dir / "train_metrics.jsonl"
    JsonlTrainingMetrics(
        metrics_path,
        formal_metadata("owner/expected", 1),
    ).log(payload(), step=0)

    with pytest.raises(ValueError, match="does not match dataset"):
        validate_completed_training_run(
            run_dir,
            num_train_steps=1,
            dataset_repo="owner/expected",
            metrics_path=metrics_path,
        )


def test_formal_training_requires_source_provenance(tmp_path: Path) -> None:
    dataset_repo = "owner/data"
    run_dir = tmp_path / "stage"
    write_complete_checkpoint(run_dir / "0", dataset_repo)
    metrics_path = run_dir / "train_metrics.jsonl"
    JsonlTrainingMetrics(
        metrics_path,
        {"dataset_repo": dataset_repo, "num_train_steps": 1},
    ).log(payload(), step=0)

    with pytest.raises(ValueError, match="clean project commit"):
        validate_completed_training_run(
            run_dir,
            num_train_steps=1,
            dataset_repo=dataset_repo,
            metrics_path=metrics_path,
        )


def test_formal_training_requires_dataset_content_fingerprint(tmp_path: Path) -> None:
    dataset_repo = "owner/data"
    run_dir = tmp_path / "stage"
    write_complete_checkpoint(run_dir / "0", dataset_repo)
    metrics_path = run_dir / "train_metrics.jsonl"
    metadata = formal_metadata(dataset_repo, 1)
    metadata.pop("dataset_sha256")
    JsonlTrainingMetrics(metrics_path, metadata).log(payload(), step=0)

    with pytest.raises(ValueError, match="dataset content fingerprint"):
        validate_completed_training_run(
            run_dir,
            num_train_steps=1,
            dataset_repo=dataset_repo,
            metrics_path=metrics_path,
        )


def test_synthetic_smoke_completion_does_not_require_clean_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "smoke"
    write_complete_checkpoint(run_dir / "0", "fake")
    metrics_path = run_dir / "train_metrics.jsonl"
    JsonlTrainingMetrics(
        metrics_path,
        {"dataset_repo": "fake", "num_train_steps": 1},
    ).log(payload(), step=0)

    report = validate_completed_training_run(
        run_dir,
        num_train_steps=1,
        dataset_repo="fake",
        metrics_path=metrics_path,
    )

    assert report["verified"] is True


def test_source_tree_fingerprint_is_content_bound(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "config.toml").write_text("name = 'test'\n", encoding="utf-8")

    count, before = source_tree_fingerprint(tmp_path, (Path("src"), Path("config.toml")))
    (tmp_path / "src" / "a.py").write_text("value = 2\n", encoding="utf-8")
    _, after = source_tree_fingerprint(tmp_path, (Path("src"), Path("config.toml")))

    assert count == 2
    assert len(before) == 64
    assert before != after


def test_lerobot_dataset_path_uses_huggingface_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "lerobot" / "owner" / "data"
    dataset.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    assert lerobot_dataset_path("owner/data") == dataset.resolve()
    with pytest.raises(ValueError, match="invalid"):
        lerobot_dataset_path("../escape")


def test_dataset_fingerprint_detects_changes(tmp_path: Path) -> None:
    (tmp_path / "episode.parquet").write_bytes(b"before")
    expected = checkpoint_fingerprint(tmp_path)
    validate_dataset_fingerprint(tmp_path, expected)

    (tmp_path / "episode.parquet").write_bytes(b"after")
    with pytest.raises(ValueError, match="changed during training"):
        validate_dataset_fingerprint(tmp_path, expected)
