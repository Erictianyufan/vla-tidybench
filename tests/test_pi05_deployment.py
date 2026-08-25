from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vla_tidybench.openpi.deployment import checkpoint_inventory, load_deployment


def write_checkpoint(path: Path) -> None:
    (path / "params").mkdir(parents=True)
    (path / "_CHECKPOINT_METADATA").write_text("checkpoint", encoding="utf-8")
    (path / "params" / "_METADATA").write_text("params", encoding="utf-8")
    (path / "params" / "manifest.ocdbt").write_text("manifest", encoding="utf-8")


def make_deployment(tmp_path: Path, *, validated: bool = True) -> Path:
    checkpoint = tmp_path / "runs" / "stage3" / "2999"
    write_checkpoint(checkpoint)
    deployment = tmp_path / "deploy" / "pi05-final"
    deployment.mkdir(parents=True)
    try:
        (deployment / "checkpoint").symlink_to(checkpoint, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symbolic links unavailable: {error}")

    evaluation_manifest = None
    if validated:
        evaluation = {
            "schema_version": 1,
            "gate_passed": True,
            "autonomous_only": True,
            "checkpoint": str(checkpoint.resolve()),
        }
        evaluation_path = deployment / "evaluation.json"
        evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
        evaluation_manifest = {"sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest()}
    file_count, byte_count = checkpoint_inventory(checkpoint)
    manifest = {
        "format_version": 1,
        "policy_mode": "full",
        "policy_config": "drawer_four_skill",
        "checkpoint": str(checkpoint.resolve()),
        "file_count": file_count,
        "byte_count": byte_count,
        "project_dirty": False,
        "evaluation": evaluation_manifest,
    }
    (deployment / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return deployment


def test_validated_deployment_is_accepted(tmp_path: Path) -> None:
    deployment = load_deployment(make_deployment(tmp_path))
    assert deployment.policy_mode == "full"
    assert deployment.evaluation is not None
    assert deployment.checkpoint.name == "2999"


def test_modified_evaluation_is_rejected(tmp_path: Path) -> None:
    deployment = make_deployment(tmp_path)
    (deployment / "evaluation.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_deployment(deployment)


def test_unvalidated_bundle_requires_explicit_opt_out(tmp_path: Path) -> None:
    deployment = make_deployment(tmp_path, validated=False)
    with pytest.raises(ValueError, match="no evaluation"):
        load_deployment(deployment)
    assert load_deployment(deployment, require_validated=False).evaluation is None
