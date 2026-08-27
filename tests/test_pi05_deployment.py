from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
from vla_tidybench.openpi.deployment import (
    CHECKPOINT_DIGEST_ALGORITHM,
    checkpoint_asset_id,
    checkpoint_fingerprint,
    load_deployment,
)
from vla_tidybench.task_metrics import FORMAL_SUCCESS_HOLD_STEPS, SUCCESS_PREDICATE_VERSION

ROOT = Path(__file__).resolve().parents[1]
PROJECT_COMMIT = "d" * 40
EXPORT_SPEC = importlib.util.spec_from_file_location(
    "export_pi05_checkpoint", ROOT / "scripts" / "export_pi05_checkpoint.py"
)
assert EXPORT_SPEC is not None and EXPORT_SPEC.loader is not None
exporter = importlib.util.module_from_spec(EXPORT_SPEC)
sys.modules[EXPORT_SPEC.name] = exporter
EXPORT_SPEC.loader.exec_module(exporter)


def write_checkpoint(path: Path) -> None:
    (path / "params").mkdir(parents=True)
    (path / "assets" / "owner" / "data").mkdir(parents=True)
    (path / "_CHECKPOINT_METADATA").write_text("checkpoint", encoding="utf-8")
    (path / "params" / "_METADATA").write_text("params", encoding="utf-8")
    (path / "params" / "manifest.ocdbt").write_text("manifest", encoding="utf-8")
    (path / "assets" / "owner" / "data" / "norm_stats.json").write_text("{}", encoding="utf-8")


def formal_evaluation(checkpoint: Path, checkpoint_sha256: str) -> dict[str, object]:
    skills = ("open", "pick", "place", "close")
    context_sources = [
        {
            "file": f"drawer_{skill}_formal.hdf5",
            "bytes": 100,
            "sha256": "c" * 64,
            "episode_indices": list(range(5)),
            "episode_names": [f"demo_{seed}" for seed in range(5)],
        }
        for skill in skills
    ]
    context_lock = {
        "schema_version": 1,
        "context_manifest": "main_validation.json",
        "context_manifest_sha256": "f" * 64,
        "context_count": 20,
        "total_bytes": 400,
        "sources": context_sources,
    }
    episodes = [
        {
            "skill": skill,
            "seed": seed,
            "context": f"drawer_{skill}_formal.hdf5::demo_{seed}",
            "checkpoint_sha256": checkpoint_sha256,
            "project_commit": PROJECT_COMMIT,
            "success_predicate": SUCCESS_PREDICATE_VERSION,
            "success_hold_steps": FORMAL_SUCCESS_HOLD_STEPS,
            "success": True,
        }
        for skill in skills
        for seed in range(5)
    ]
    return {
        "schema_version": 1,
        "gate_passed": True,
        "autonomous_only": True,
        "policy": "pi0.5-drawer-full",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "project_commit": PROJECT_COMMIT,
        "context_lock": context_lock,
        "context_lock_sha256": hashlib.sha256(
            (json.dumps(context_lock, indent=2) + "\n").encode()
        ).hexdigest(),
        "required_skills": list(skills),
        "episode_count": len(episodes),
        "successes": len(episodes),
        "overall_success_rate": 1.0,
        "p95_infer_ms": 100.0,
        "thresholds": {
            "min_episodes_per_skill": 5,
            "min_success_rate": 0.6,
            "max_p95_infer_ms": 250.0,
        },
        "per_skill": {
            skill: {"episodes": 5, "successes": 5, "success_rate": 1.0} for skill in skills
        },
        "episodes": episodes,
    }


def training_completion(
    checkpoint: Path, checkpoint_sha256: str, file_count: int, byte_count: int
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "verified": True,
        "checkpoint": str(checkpoint.resolve()),
        "final_step": int(checkpoint.name),
        "num_train_steps": int(checkpoint.name) + 1,
        "dataset_repo": "owner/data",
        "checkpoint_digest_algorithm": CHECKPOINT_DIGEST_ALGORITHM,
        "checkpoint_file_count": file_count,
        "checkpoint_byte_count": byte_count,
        "checkpoint_sha256": checkpoint_sha256,
        "project_commit": PROJECT_COMMIT,
        "project_dirty": False,
        "openpi_source_files": 72,
        "openpi_source_sha256": "a" * 64,
        "init_params": "/checkpoints/stage2/9999/params",
        "init_params_files": 10,
        "init_params_sha256": "b" * 64,
        "loss": 0.1,
        "grad_norm": 1.0,
        "param_norm": 2.0,
    }


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
    training_manifest = None
    file_count, byte_count, checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    if validated:
        training = training_completion(checkpoint, checkpoint_sha256, file_count, byte_count)
        training_path = deployment / "training_completion.json"
        training_path.write_text(json.dumps(training), encoding="utf-8")
        training_manifest = {
            "path": "training_completion.json",
            "sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
            "project_commit": training["project_commit"],
            "openpi_source_sha256": training["openpi_source_sha256"],
            "init_params_sha256": training["init_params_sha256"],
        }
        evaluation = formal_evaluation(checkpoint, checkpoint_sha256)
        evaluation_path = deployment / "evaluation.json"
        evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
        evaluation_manifest = {"sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest()}
    manifest = {
        "format_version": 3,
        "stage": "stage3-hard-recovery",
        "dataset_repo": "owner/data",
        "policy_mode": "full",
        "policy_config": "drawer_four_skill",
        "checkpoint": str(checkpoint.resolve()),
        "file_count": file_count,
        "byte_count": byte_count,
        "checkpoint_digest": {
            "algorithm": CHECKPOINT_DIGEST_ALGORITHM,
            "sha256": checkpoint_sha256,
        },
        "project_dirty": False,
        "project_commit": PROJECT_COMMIT,
        "training": training_manifest,
        "evaluation": evaluation_manifest,
    }
    (deployment / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return deployment


def test_validated_deployment_is_accepted(tmp_path: Path) -> None:
    deployment = load_deployment(make_deployment(tmp_path))
    assert deployment.policy_mode == "full"
    assert deployment.training is not None
    assert deployment.evaluation is not None
    assert deployment.checkpoint.name == "2999"
    assert checkpoint_asset_id(deployment.checkpoint) == "owner/data"


def test_dataset_repo_must_match_embedded_normalization_assets(tmp_path: Path) -> None:
    deployment = make_deployment(tmp_path)
    manifest_path = deployment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_repo"] = "owner/wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="normalization asset ID"):
        load_deployment(deployment)


def test_checkpoint_requires_one_unambiguous_normalization_asset(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint(checkpoint)
    (checkpoint / "assets" / "owner" / "other").mkdir(parents=True)
    (checkpoint / "assets" / "owner" / "other" / "norm_stats.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        checkpoint_asset_id(checkpoint)


def test_modified_evaluation_is_rejected(tmp_path: Path) -> None:
    deployment = make_deployment(tmp_path)
    (deployment / "evaluation.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_deployment(deployment)


def test_modified_training_completion_is_rejected(tmp_path: Path) -> None:
    deployment = make_deployment(tmp_path)
    (deployment / "training_completion.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="training completion checksum"):
        load_deployment(deployment)


def test_training_identity_must_match_embedded_completion(tmp_path: Path) -> None:
    deployment = make_deployment(tmp_path)
    manifest_path = deployment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training"]["project_commit"] = "e" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="training identity disagrees"):
        load_deployment(deployment)


def test_same_size_checkpoint_mutation_is_rejected(tmp_path: Path) -> None:
    deployment = make_deployment(tmp_path)
    checkpoint = (deployment / "checkpoint").resolve()
    metadata = checkpoint / "params" / "_METADATA"
    metadata.write_text("broken", encoding="utf-8")
    assert metadata.stat().st_size == len("params")
    with pytest.raises(ValueError, match="content SHA-256"):
        load_deployment(deployment)


def test_weakened_evaluation_threshold_is_rejected(tmp_path: Path) -> None:
    deployment = make_deployment(tmp_path)
    evaluation_path = deployment / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["thresholds"]["min_success_rate"] = 0.5
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    manifest_path = deployment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation"]["sha256"] = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="too weak"):
        load_deployment(deployment)


def test_unvalidated_bundle_requires_explicit_opt_out(tmp_path: Path) -> None:
    deployment = make_deployment(tmp_path, validated=False)
    with pytest.raises(ValueError, match="no training completion"):
        load_deployment(deployment)
    assert load_deployment(deployment, require_validated=False).evaluation is None


def test_legacy_bundle_is_never_a_formal_deployment(tmp_path: Path) -> None:
    deployment = make_deployment(tmp_path, validated=False)
    manifest_path = deployment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 1
    manifest.pop("checkpoint_digest")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="format_version 3"):
        load_deployment(deployment)
    assert load_deployment(deployment, require_validated=False).checkpoint.name == "2999"


def test_export_binds_evaluated_checkpoint_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "runs" / "stage3" / "2999"
    write_checkpoint(checkpoint)
    _, _, checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    file_count, byte_count, _ = checkpoint_fingerprint(checkpoint)
    training_path = checkpoint.parent / "training_completion.json"
    training_path.write_text(
        json.dumps(training_completion(checkpoint, checkpoint_sha256, file_count, byte_count)),
        encoding="utf-8",
    )
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(formal_evaluation(checkpoint, checkpoint_sha256)), encoding="utf-8")

    def clean_git(_root: Path, *arguments: str) -> str:
        return PROJECT_COMMIT if arguments == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr(exporter, "git_value", clean_git)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_pi05_checkpoint.py",
            "--checkpoint",
            str(checkpoint),
            "--dataset-repo",
            "owner/data",
            "--evaluation-report",
            str(evaluation_path),
            "--output-root",
            str(tmp_path / "deploy"),
        ],
    )
    exporter.main()

    deployment = load_deployment(tmp_path / "deploy" / "pi05-tidybench-final")
    assert deployment.manifest["format_version"] == 3
    assert deployment.training is not None
    assert deployment.manifest["checkpoint_storage"] == "copy"
    assert not (deployment.root / "checkpoint").is_symlink()
    assert deployment.checkpoint_sha256 == checkpoint_sha256

    shutil.rmtree(checkpoint)
    relocated = load_deployment(tmp_path / "deploy" / "pi05-tidybench-final")
    assert relocated.checkpoint_sha256 == checkpoint_sha256


def test_copy_manifest_rejects_a_checkpoint_symlink(tmp_path: Path) -> None:
    deployment = make_deployment(tmp_path)
    manifest_path = deployment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_storage"] = "copy"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="real directory"):
        load_deployment(deployment)


def test_install_directory_replaces_complete_bundle(tmp_path: Path) -> None:
    output = tmp_path / "deploy"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")

    exporter.install_directory(staging, output, replace=True)

    assert not staging.exists()
    assert not (output / "old.txt").exists()
    assert (output / "new.txt").read_text(encoding="utf-8") == "new"
    assert not (tmp_path / ".deploy.backup").exists()


def test_install_directory_restores_prior_bundle_on_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "deploy"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()
    original_rename = Path.rename

    def fail_staging_publish(path: Path, target: Path) -> Path:
        if path == staging:
            raise OSError("simulated publish failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staging_publish)
    with pytest.raises(OSError, match="simulated"):
        exporter.install_directory(staging, output, replace=True)

    assert (output / "old.txt").read_text(encoding="utf-8") == "old"
    assert staging.is_dir()
    assert not (tmp_path / ".deploy.backup").exists()


def test_policy_probe_checks_action_shape_and_checkpoint_metadata() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "probe_drawer_policy.py").read_text(
        encoding="utf-8"
    )
    assert 'actions.shape != (16, 7)' in source
    assert 'metadata.get("checkpoint")' in source
    assert 'metadata.get("checkpoint_sha256", "")' in source
    assert 'metadata.get("evaluation_gate_passed"' in source


def test_policy_server_binds_config_to_checkpoint_dataset_assets() -> None:
    source = (ROOT / "scripts" / "serve_drawer_policy.py").read_text(encoding="utf-8")
    assert 'dataset_repo = str(deployment.manifest["dataset_repo"])' in source
    assert "args.dataset_repo or checkpoint_asset_id(checkpoint)" in source
    assert "make_config(finetune_mode=mode, dataset_repo=dataset_repo)" in source
    assert '"project_commit": project_commit' in source
    assert "validated deployment must be served by its exact clean project commit" in source


def test_evaluation_from_another_project_commit_is_rejected(tmp_path: Path) -> None:
    deployment = make_deployment(tmp_path)
    evaluation_path = deployment / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["project_commit"] = "e" * 40
    for episode in evaluation["episodes"]:
        episode["project_commit"] = "e" * 40
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    manifest_path = deployment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation"]["sha256"] = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match deployment code"):
        load_deployment(deployment)


def test_evaluation_context_outside_locked_set_is_rejected(tmp_path: Path) -> None:
    deployment = make_deployment(tmp_path)
    evaluation_path = deployment / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["episodes"][0]["context"] = "drawer_open_formal.hdf5::demo_99"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    manifest_path = deployment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation"]["sha256"] = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="not content-locked"):
        load_deployment(deployment)
