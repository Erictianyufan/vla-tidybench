#!/usr/bin/env python3
"""Publish a trained pi0.5 checkpoint as a stable simulation deployment bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def data_root() -> Path:
    return Path(os.environ.get("VLA_TIDYBENCH_DATA", f"/data/{os.environ.get('USER', 'user')}/vla-tidybench"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="OpenPI numeric step directory")
    parser.add_argument("--name", default="pi05-tidybench-final")
    parser.add_argument("--stage", default="stage3-hard-recovery")
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--mode", choices=("lora", "expert", "full"), default="full")
    parser.add_argument("--evaluation-report", type=Path)
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


def main() -> int:
    args = parse_args()
    if args.evaluation_report is None and not args.allow_unvalidated:
        raise ValueError("formal export requires --evaluation-report")
    if args.allow_unvalidated and "smoke" not in args.stage:
        raise ValueError("--allow-unvalidated is restricted to a smoke stage label")
    checkpoint = args.checkpoint.expanduser().resolve()
    required = [
        checkpoint / "_CHECKPOINT_METADATA",
        checkpoint / "params" / "_METADATA",
        checkpoint / "params" / "manifest.ocdbt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("checkpoint is incomplete; missing: " + ", ".join(missing))

    evaluation: dict[str, object] | None = None
    evaluation_path: Path | None = None
    if args.evaluation_report is not None:
        evaluation_path = args.evaluation_report.expanduser().resolve()
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if int(evaluation.get("schema_version", -1)) != 1:
            raise ValueError("unsupported evaluation report schema")
        if not bool(evaluation.get("gate_passed", False)):
            raise ValueError("refusing to export a checkpoint that failed its evaluation gate")
        if not bool(evaluation.get("autonomous_only", False)):
            raise ValueError("formal deployment evaluation must contain autonomous rollouts only")
        evaluated_checkpoint = Path(str(evaluation.get("checkpoint", ""))).expanduser().resolve()
        if evaluated_checkpoint != checkpoint:
            raise ValueError(f"evaluation checkpoint {evaluated_checkpoint} does not match {checkpoint}")

    output = args.output_root.expanduser().resolve() / args.name
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_link = output / "checkpoint"
    if checkpoint_link.exists() or checkpoint_link.is_symlink():
        if not args.replace:
            raise FileExistsError(f"deployment link already exists: {checkpoint_link}")
        if checkpoint_link.is_dir() and not checkpoint_link.is_symlink():
            raise IsADirectoryError(f"refusing to replace a real directory: {checkpoint_link}")
        checkpoint_link.unlink()
    checkpoint_link.symlink_to(checkpoint, target_is_directory=True)

    evaluation_manifest: dict[str, object] | None = None
    if evaluation_path is not None and evaluation is not None:
        deployed_evaluation = output / "evaluation.json"
        if evaluation_path != deployed_evaluation:
            shutil.copyfile(evaluation_path, deployed_evaluation)
        evaluation_manifest = {
            "path": str(deployed_evaluation),
            "sha256": hashlib.sha256(deployed_evaluation.read_bytes()).hexdigest(),
            "episode_count": evaluation.get("episode_count"),
            "overall_success_rate": evaluation.get("overall_success_rate"),
            "p95_infer_ms": evaluation.get("p95_infer_ms"),
            "gate_passed": True,
        }

    project_root = Path(__file__).resolve().parents[1]
    file_count = sum(1 for path in checkpoint.rglob("*") if path.is_file())
    byte_count = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
    manifest = {
        "format_version": 1,
        "name": args.name,
        "stage": args.stage,
        "dataset_repo": args.dataset_repo,
        "policy_mode": args.mode,
        "checkpoint": str(checkpoint),
        "deployment_checkpoint": str(checkpoint_link),
        "file_count": file_count,
        "byte_count": byte_count,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_commit": git_value(project_root, "rev-parse", "HEAD"),
        "project_dirty": bool(git_value(project_root, "status", "--porcelain")),
        "evaluation": evaluation_manifest,
        "serve_command": (
            f"make drawer-policy-serve CHECKPOINT={checkpoint_link} "
            f"POLICY_MODE={args.mode} POLICY_CONFIG_FLAG=--four-skill"
        ),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
