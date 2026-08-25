#!/usr/bin/env python3
"""Serve a trained VLA-TidyBench pi0.5 drawer checkpoint over WebSocket."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from vla_tidybench.openpi.drawer_config import make_config as make_open_config
from vla_tidybench.openpi.drawer_four_skill_config import make_config as make_four_skill_config
from vla_tidybench.openpi.deployment import load_deployment


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
        "--allow-unvalidated-deployment",
        action="store_true",
        help="permit a systems-smoke deployment without a passing evaluation report",
    )
    args = parser.parse_args()

    deployment = None
    if args.deployment is not None:
        deployment = load_deployment(args.deployment, require_validated=not args.allow_unvalidated_deployment)
        if args.mode is not None and args.mode != deployment.policy_mode:
            parser.error(f"--mode {args.mode} disagrees with deployment mode {deployment.policy_mode}")
        if deployment.manifest.get("policy_config") != "drawer_four_skill":
            parser.error("deployment is not marked as the drawer_four_skill policy configuration")
        checkpoint = deployment.checkpoint
        mode = deployment.policy_mode
        four_skill = True
    else:
        assert args.checkpoint is not None
        checkpoint = args.checkpoint.expanduser().resolve()
        mode = args.mode or "expert"
        four_skill = args.four_skill

    make_config = make_four_skill_config if four_skill else make_open_config
    policy = policy_config.create_trained_policy(
        make_config(finetune_mode=mode), checkpoint, default_prompt=args.default_prompt
    )
    metadata = dict(policy.metadata)
    metadata.update(
        {
            "project": "VLA-TidyBench",
            "policy": f"pi0.5-drawer-{mode}",
            "checkpoint": str(checkpoint),
            "deployment": str(deployment.root) if deployment is not None else None,
            "evaluation_gate_passed": bool(deployment and deployment.evaluation),
        }
    )
    WebsocketPolicyServer(policy, args.host, args.port, metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
