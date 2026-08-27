#!/usr/bin/env python3
"""Serve a trained VLA-TidyBench pi0.5 drawer checkpoint over WebSocket."""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

import numpy as np
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import normalize
from vla_tidybench.openpi.deployment import (
    checkpoint_asset_id,
    checkpoint_fingerprint,
    load_deployment,
)
from vla_tidybench.openpi.drawer_config import make_config as make_open_config
from vla_tidybench.openpi.drawer_four_skill_config import make_config as make_four_skill_config


def identity_norm_stats() -> dict[str, normalize.NormStats]:
    zeros = np.zeros(32, dtype=np.float32)
    ones = np.ones(32, dtype=np.float32)
    stats = normalize.NormStats(mean=zeros, std=ones, q01=-ones, q99=ones)
    return {"state": stats, "actions": stats}


def git_state(project_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    revision = commit.stdout.strip() if commit.returncode == 0 else ""
    dirty = status.returncode != 0 or bool(status.stdout.strip())
    return revision, dirty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="direct numeric OpenPI checkpoint")
    source.add_argument("--deployment", type=Path, help="exported deployment directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt", default="open the top drawer")
    parser.add_argument("--four-skill", action="store_true")
    parser.add_argument("--mode", choices=("lora", "expert", "full"))
    parser.add_argument(
        "--dataset-repo",
        help="normalization asset ID; direct checkpoints auto-discover it when omitted",
    )
    parser.add_argument(
        "--allow-unvalidated-deployment",
        action="store_true",
        help="permit a systems-smoke deployment without a passing evaluation report",
    )
    parser.add_argument(
        "--synthetic-identity-norm",
        action="store_true",
        help="systems smoke only: serve without dataset-derived normalization statistics",
    )
    args = parser.parse_args()
    if args.synthetic_identity_norm and not args.allow_unvalidated_deployment:
        parser.error("--synthetic-identity-norm requires --allow-unvalidated-deployment")

    project_root = Path(__file__).resolve().parents[1]
    project_commit, project_dirty = git_state(project_root)
    deployment = None
    if args.deployment is not None:
        deployment = load_deployment(args.deployment, require_validated=not args.allow_unvalidated_deployment)
        if args.mode is not None and args.mode != deployment.policy_mode:
            parser.error(f"--mode {args.mode} disagrees with deployment mode {deployment.policy_mode}")
        if deployment.manifest.get("policy_config") != "drawer_four_skill":
            parser.error("deployment is not marked as the drawer_four_skill policy configuration")
        checkpoint = deployment.checkpoint
        checkpoint_sha256 = deployment.checkpoint_sha256
        mode = deployment.policy_mode
        four_skill = True
        dataset_repo = str(deployment.manifest["dataset_repo"])
        if not args.allow_unvalidated_deployment and (
            project_dirty or project_commit != deployment.manifest.get("project_commit")
        ):
            parser.error("validated deployment must be served by its exact clean project commit")
    else:
        assert args.checkpoint is not None
        checkpoint = args.checkpoint.expanduser().resolve()
        _, _, checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
        mode = args.mode or "expert"
        four_skill = args.four_skill
        dataset_repo = args.dataset_repo or checkpoint_asset_id(checkpoint)

    make_config = make_four_skill_config if four_skill else make_open_config
    policy = policy_config.create_trained_policy(
        make_config(finetune_mode=mode, dataset_repo=dataset_repo),
        checkpoint,
        default_prompt=args.default_prompt,
        norm_stats=identity_norm_stats() if args.synthetic_identity_norm else None,
    )
    metadata = dict(policy.metadata)
    metadata.update(
        {
            "project": "VLA-TidyBench",
            "policy": f"pi0.5-drawer-{mode}",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_repo": dataset_repo,
            "project_commit": project_commit,
            "project_dirty": project_dirty,
            "deployment": str(deployment.root) if deployment is not None else None,
            "evaluation_gate_passed": bool(deployment and deployment.evaluation),
            "synthetic_identity_norm": args.synthetic_identity_norm,
        }
    )
    WebsocketPolicyServer(policy, args.host, args.port, metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
