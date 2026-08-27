"""Append-only local metrics for OpenPI runs with external tracking disabled."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vla_tidybench.openpi.deployment import (
    CHECKPOINT_DIGEST_ALGORITHM,
    REQUIRED_CHECKPOINT_FILES,
    checkpoint_asset_id,
    checkpoint_fingerprint,
    validate_training_completion,
)

METRIC_KEYS = ("loss", "grad_norm", "param_norm")
TRAINING_COMPLETION_FILENAME = "training_completion.json"
OPENPI_PROVENANCE_PATHS = (
    Path("scripts/train.py"),
    Path("src/openpi"),
    Path("packages/openpi-client/src"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)


def lerobot_dataset_path(repo_id: str) -> Path:
    """Resolve one local LeRobot repository without importing LeRobot."""

    if not repo_id or repo_id == "fake":
        raise ValueError("a real LeRobot repository ID is required")
    relative = Path(repo_id)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid LeRobot repository ID: {repo_id!r}")
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    root = Path(os.environ.get("HF_LEROBOT_HOME", hf_home / "lerobot")).expanduser().resolve()
    dataset = (root / relative).resolve()
    if not dataset.is_relative_to(root):
        raise ValueError(f"LeRobot dataset escapes its cache root: {dataset}")
    if not dataset.is_dir():
        raise FileNotFoundError(f"LeRobot dataset is missing: {dataset}")
    return dataset


def validate_dataset_fingerprint(
    dataset: Path,
    expected: tuple[int, int, str],
) -> None:
    """Fail if a training dataset changed after its initial fingerprint."""

    observed = checkpoint_fingerprint(dataset)
    if observed != expected:
        raise ValueError(
            "LeRobot dataset content changed during training: "
            f"expected={expected}, got={observed}"
        )


def git_state(project_root: Path) -> tuple[str, bool]:
    """Return a repository revision and whether its worktree is unclean/unreadable."""

    root = project_root.expanduser().resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=False
    )
    revision = commit.stdout.strip() if commit.returncode == 0 else ""
    dirty = status.returncode != 0 or bool(status.stdout.strip())
    return revision, dirty


def source_tree_fingerprint(root: Path, relative_paths: tuple[Path, ...]) -> tuple[int, str]:
    """Hash path identities and bytes for the exact source roots used by training."""

    root = root.expanduser().resolve()
    files: set[Path] = set()
    for relative in relative_paths:
        target = root / relative
        if target.is_file():
            files.add(target)
        elif target.is_dir():
            files.update(
                path
                for path in target.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
            )
        else:
            raise FileNotFoundError(f"training source path is missing: {target}")
    digest = hashlib.sha256()
    ordered = sorted(files, key=lambda path: path.relative_to(root).as_posix())
    for path in ordered:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return len(ordered), digest.hexdigest()


def build_training_provenance(project_root: Path, openpi_root: Path) -> dict[str, object]:
    """Capture immutable identities for the project checkout and OpenPI source tree."""

    project_commit, project_dirty = git_state(project_root)
    openpi_files, openpi_sha256 = source_tree_fingerprint(openpi_root, OPENPI_PROVENANCE_PATHS)
    return {
        "project_commit": project_commit,
        "project_dirty": project_dirty,
        "openpi_source_files": openpi_files,
        "openpi_source_sha256": openpi_sha256,
    }


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
    if dataset_repo != "fake":
        project_commit = str(final_metrics.get("project_commit", ""))
        valid_commit = len(project_commit) in (40, 64) and all(
            char in "0123456789abcdef" for char in project_commit
        )
        if not valid_commit or final_metrics.get("project_dirty") is not False:
            raise ValueError("formal training metrics require an exact clean project commit")
        openpi_sha256 = str(final_metrics.get("openpi_source_sha256", ""))
        openpi_files = int(final_metrics.get("openpi_source_files", 0))
        if (
            len(openpi_sha256) != 64
            or any(char not in "0123456789abcdef" for char in openpi_sha256)
            or openpi_files < 1
        ):
            raise ValueError("formal training metrics require a valid OpenPI source fingerprint")
        init_sha256 = str(final_metrics.get("init_params_sha256", ""))
        init_files = int(final_metrics.get("init_params_files", 0))
        if (
            not str(final_metrics.get("init_params", "")).strip()
            or len(init_sha256) != 64
            or any(char not in "0123456789abcdef" for char in init_sha256)
            or init_files < 1
        ):
            raise ValueError("formal training metrics require a valid initialization fingerprint")
        dataset_path = str(final_metrics.get("dataset_path", ""))
        dataset_sha256 = str(final_metrics.get("dataset_sha256", ""))
        dataset_files = int(final_metrics.get("dataset_files", 0))
        dataset_bytes = int(final_metrics.get("dataset_bytes", 0))
        if (
            not dataset_path
            or final_metrics.get("dataset_digest_algorithm") != CHECKPOINT_DIGEST_ALGORITHM
            or len(dataset_sha256) != 64
            or any(char not in "0123456789abcdef" for char in dataset_sha256)
            or dataset_files < 1
            or dataset_bytes < 1
        ):
            raise ValueError("formal training metrics require a valid dataset content fingerprint")
    nonfinite = [
        key for key in METRIC_KEYS if not math.isfinite(float(final_metrics.get(key, float("nan"))))
    ]
    if nonfinite:
        raise ValueError(f"final training metrics are non-finite: {nonfinite}")
    file_count, byte_count, checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    report = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "checkpoint": str(checkpoint),
        "final_step": final_step,
        "num_train_steps": num_train_steps,
        "dataset_repo": dataset_repo,
        "checkpoint_asset_id": asset_id,
        "checkpoint_digest_algorithm": CHECKPOINT_DIGEST_ALGORITHM,
        "checkpoint_file_count": file_count,
        "checkpoint_byte_count": byte_count,
        "checkpoint_sha256": checkpoint_sha256,
        "metrics_path": str(metrics_path),
        "loss": float(final_metrics["loss"]),
        "grad_norm": float(final_metrics["grad_norm"]),
        "param_norm": float(final_metrics["param_norm"]),
        "project_commit": final_metrics.get("project_commit"),
        "project_dirty": final_metrics.get("project_dirty"),
        "openpi_source_files": final_metrics.get("openpi_source_files"),
        "openpi_source_sha256": final_metrics.get("openpi_source_sha256"),
        "init_params": final_metrics.get("init_params"),
        "init_params_files": final_metrics.get("init_params_files"),
        "init_params_sha256": final_metrics.get("init_params_sha256"),
        "dataset_path": final_metrics.get("dataset_path"),
        "dataset_digest_algorithm": final_metrics.get("dataset_digest_algorithm"),
        "dataset_files": final_metrics.get("dataset_files"),
        "dataset_bytes": final_metrics.get("dataset_bytes"),
        "dataset_sha256": final_metrics.get("dataset_sha256"),
        "verified": True,
    }
    validate_training_completion(
        report,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        file_count=file_count,
        byte_count=byte_count,
        dataset_repo=dataset_repo,
        require_clean_provenance=dataset_repo != "fake",
    )
    return report


def write_training_completion(run_dir: Path, report: Mapping[str, Any]) -> Path:
    """Atomically publish the durable completion report beside stage checkpoints."""

    output = run_dir.expanduser().resolve() / TRAINING_COMPLETION_FILENAME
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(dict(report), indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output
