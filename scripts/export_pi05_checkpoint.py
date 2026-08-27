#!/usr/bin/env python3
"""Publish a trained pi0.5 checkpoint as a stable simulation deployment bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from vla_tidybench.openpi.deployment import (
    CHECKPOINT_DIGEST_ALGORITHM,
    checkpoint_fingerprint,
    load_deployment,
    validate_formal_evaluation,
    validate_training_completion,
)


def data_root() -> Path:
    return Path(os.environ.get("VLA_TIDYBENCH_DATA", f"/data/{os.environ.get('USER', 'user')}/vla-tidybench"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="OpenPI numeric step directory")
    parser.add_argument("--name", default="pi05-tidybench-final")
    parser.add_argument("--stage", default="stage3-hard-recovery")
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--mode", choices=("lora", "expert", "full"), default="full")
    parser.add_argument(
        "--checkpoint-storage",
        choices=("copy", "symlink"),
        default="copy",
        help="copy creates a portable bundle; symlink avoids duplicating weights on one host",
    )
    parser.add_argument("--evaluation-report", type=Path)
    parser.add_argument(
        "--training-report",
        type=Path,
        help="defaults to training_completion.json beside the numeric checkpoint",
    )
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="systems-smoke exports only; formal deployment requires a passing evaluation report",
    )
    parser.add_argument("--output-root", type=Path, default=data_root() / "checkpoints" / "deploy")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def git_value(project_root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments], cwd=project_root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def install_directory(staging: Path, output: Path, *, replace: bool) -> None:
    """Install a fully built bundle and preserve an existing bundle on failure."""

    if not output.exists() and not output.is_symlink():
        staging.rename(output)
        return
    if not replace:
        raise FileExistsError(f"deployment already exists: {output}; pass --replace")
    if output.is_symlink() or not output.is_dir():
        raise ValueError(f"refusing to replace a non-directory deployment: {output}")
    backup = output.with_name(f".{output.name}.backup")
    if backup.exists() or backup.is_symlink():
        raise FileExistsError(f"stale deployment backup requires inspection: {backup}")
    output.rename(backup)
    try:
        staging.rename(output)
    except BaseException:
        backup.rename(output)
        raise
    shutil.rmtree(backup)


def main() -> int:
    args = parse_args()
    if not args.name or Path(args.name).name != args.name:
        raise ValueError("--name must be one directory name without path separators")
    if args.evaluation_report is None and not args.allow_unvalidated:
        raise ValueError("formal export requires --evaluation-report")
    if args.allow_unvalidated and "smoke" not in args.stage:
        raise ValueError("--allow-unvalidated is restricted to a smoke stage label")
    if args.evaluation_report is not None and (args.mode != "full" or args.stage != "stage3-hard-recovery"):
        raise ValueError("formal export requires full mode and the stage3-hard-recovery label")
    checkpoint = args.checkpoint.expanduser().resolve()
    required = [
        checkpoint / "_CHECKPOINT_METADATA",
        checkpoint / "params" / "_METADATA",
        checkpoint / "params" / "manifest.ocdbt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("checkpoint is incomplete; missing: " + ", ".join(missing))

    project_root = Path(__file__).resolve().parents[1]
    project_commit = git_value(project_root, "rev-parse", "HEAD")
    project_status = git_value(project_root, "status", "--porcelain")
    project_dirty = project_status is None or bool(project_status)

    evaluation: dict[str, object] | None = None
    evaluation_path: Path | None = None
    file_count, byte_count, checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    training: dict[str, object] | None = None
    training_path = (args.training_report or checkpoint.parent / "training_completion.json").expanduser().resolve()
    if training_path.is_file():
        training = json.loads(training_path.read_text(encoding="utf-8"))
        if not isinstance(training, dict):
            raise ValueError("training completion report must be a JSON object")
        validate_training_completion(
            training,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            file_count=file_count,
            byte_count=byte_count,
            dataset_repo=args.dataset_repo,
            require_clean_provenance=args.evaluation_report is not None,
        )
    elif args.evaluation_report is not None:
        raise FileNotFoundError(f"formal export requires training completion report: {training_path}")
    if args.evaluation_report is not None:
        evaluation_path = args.evaluation_report.expanduser().resolve()
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        validate_formal_evaluation(
            evaluation,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            project_commit=project_commit,
        )

    if evaluation is not None:
        valid_commit = project_commit is not None and len(project_commit) in (40, 64) and all(
            char in "0123456789abcdef" for char in project_commit
        )
        if not valid_commit:
            raise ValueError("formal export requires a valid Git commit")
        if project_dirty:
            raise ValueError("formal export requires a clean project checkout")

    output_root = args.output_root.expanduser().resolve()
    output = output_root / args.name
    if (output.exists() or output.is_symlink()) and not args.replace:
        raise FileExistsError(f"deployment already exists: {output}; pass --replace")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.name}.staging-", dir=output_root))
    try:
        checkpoint_entry = staging / "checkpoint"
        if args.checkpoint_storage == "copy":
            shutil.copytree(checkpoint, checkpoint_entry)
            copied_fingerprint = checkpoint_fingerprint(checkpoint_entry)
            if copied_fingerprint != (file_count, byte_count, checkpoint_sha256):
                raise ValueError("copied checkpoint fingerprint differs from its source")
        else:
            checkpoint_entry.symlink_to(checkpoint, target_is_directory=True)

        evaluation_manifest: dict[str, object] | None = None
        training_manifest: dict[str, object] | None = None
        if training_path.is_file() and training is not None:
            deployed_training = staging / "training_completion.json"
            shutil.copyfile(training_path, deployed_training)
            training_manifest = {
                "path": "training_completion.json",
                "sha256": hashlib.sha256(deployed_training.read_bytes()).hexdigest(),
                "project_commit": training.get("project_commit"),
                "openpi_source_sha256": training.get("openpi_source_sha256"),
                "init_params_sha256": training.get("init_params_sha256"),
                "dataset_digest_algorithm": training.get("dataset_digest_algorithm"),
                "dataset_sha256": training.get("dataset_sha256"),
            }
        if evaluation_path is not None and evaluation is not None:
            deployed_evaluation = staging / "evaluation.json"
            shutil.copyfile(evaluation_path, deployed_evaluation)
            evaluation_manifest = {
                "path": "evaluation.json",
                "sha256": hashlib.sha256(deployed_evaluation.read_bytes()).hexdigest(),
                "episode_count": evaluation.get("episode_count"),
                "overall_success_rate": evaluation.get("overall_success_rate"),
                "p95_infer_ms": evaluation.get("p95_infer_ms"),
                "gate_passed": True,
            }

        manifest = {
            "format_version": 3,
            "name": args.name,
            "stage": args.stage,
            "dataset_repo": args.dataset_repo,
            "policy_mode": args.mode,
            "policy_config": "drawer_four_skill",
            "checkpoint": str(checkpoint),
            "checkpoint_storage": args.checkpoint_storage,
            "deployment_checkpoint": "checkpoint",
            "file_count": file_count,
            "byte_count": byte_count,
            "checkpoint_digest": {
                "algorithm": CHECKPOINT_DIGEST_ALGORITHM,
                "sha256": checkpoint_sha256,
            },
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "project_commit": project_commit,
            "project_dirty": project_dirty,
            "training": training_manifest,
            "evaluation": evaluation_manifest,
            "serve_command": f"make pi05-deployment-serve DEPLOYMENT={output}",
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        load_deployment(staging, require_validated=evaluation is not None)
        install_directory(staging, output, replace=args.replace)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
