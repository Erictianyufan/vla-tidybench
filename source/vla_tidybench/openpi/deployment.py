"""Validation helpers for a simulation-ready pi0.5 deployment bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_CHECKPOINT_FILES = (
    "_CHECKPOINT_METADATA",
    "params/_METADATA",
    "params/manifest.ocdbt",
)
POLICY_MODES = frozenset(("lora", "expert", "full"))


@dataclass(frozen=True)
class Deployment:
    root: Path
    checkpoint: Path
    policy_mode: str
    manifest: dict[str, Any]
    evaluation: dict[str, Any] | None


def checkpoint_inventory(checkpoint: Path) -> tuple[int, int]:
    files = [path for path in checkpoint.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} JSON at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def load_deployment(path: Path, *, require_validated: bool = True) -> Deployment:
    """Load and verify an exported deployment before policy construction.

    Formal bundles must be clean-code exports with a checksum-bound autonomous
    evaluation report. Systems-smoke bundles can be inspected by opting out of
    the formal evaluation requirement explicitly.
    """

    root = path.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path, "deployment manifest")
    if int(manifest.get("format_version", -1)) != 1:
        raise ValueError(f"unsupported deployment format_version in {manifest_path}")

    policy_mode = str(manifest.get("policy_mode", ""))
    if policy_mode not in POLICY_MODES:
        raise ValueError(f"unsupported deployment policy_mode: {policy_mode!r}")
    if require_validated and bool(manifest.get("project_dirty", True)):
        raise ValueError("formal deployment was exported from a dirty project checkout")

    link = root / "checkpoint"
    if not link.is_symlink():
        raise ValueError(f"deployment checkpoint must be a symbolic link: {link}")
    checkpoint = link.resolve(strict=True)
    recorded_checkpoint = Path(str(manifest.get("checkpoint", ""))).expanduser().resolve()
    if checkpoint != recorded_checkpoint:
        raise ValueError(f"checkpoint link resolves to {checkpoint}, manifest records {recorded_checkpoint}")
    missing = [str(checkpoint / relative) for relative in REQUIRED_CHECKPOINT_FILES if not (checkpoint / relative).is_file()]
    if missing:
        raise ValueError("deployment checkpoint is incomplete; missing: " + ", ".join(missing))

    file_count, byte_count = checkpoint_inventory(checkpoint)
    if file_count != int(manifest.get("file_count", -1)):
        raise ValueError(f"checkpoint file count changed: expected {manifest.get('file_count')}, got {file_count}")
    if byte_count != int(manifest.get("byte_count", -1)):
        raise ValueError(f"checkpoint byte count changed: expected {manifest.get('byte_count')}, got {byte_count}")

    evaluation_manifest = manifest.get("evaluation")
    evaluation: dict[str, Any] | None = None
    if evaluation_manifest is None:
        if require_validated:
            raise ValueError("formal deployment has no evaluation record")
    else:
        if not isinstance(evaluation_manifest, dict):
            raise ValueError("deployment evaluation entry must be an object")
        evaluation_path = root / "evaluation.json"
        evaluation = _load_json(evaluation_path, "evaluation")
        digest = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
        if digest != evaluation_manifest.get("sha256"):
            raise ValueError("evaluation checksum does not match deployment manifest")
        if not bool(evaluation.get("gate_passed", False)):
            raise ValueError("deployment evaluation gate did not pass")
        if not bool(evaluation.get("autonomous_only", False)):
            raise ValueError("deployment evaluation contains assisted rollouts")
        evaluated_checkpoint = Path(str(evaluation.get("checkpoint", ""))).expanduser().resolve()
        if evaluated_checkpoint != checkpoint:
            raise ValueError(
                f"evaluation checkpoint {evaluated_checkpoint} does not match deployment checkpoint {checkpoint}"
            )

    return Deployment(root, checkpoint, policy_mode, manifest, evaluation)
