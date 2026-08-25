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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt", default="open the top drawer")
    parser.add_argument("--four-skill", action="store_true")
    parser.add_argument("--mode", choices=("lora", "expert", "full"), default="expert")
    args = parser.parse_args()
    make_config = make_four_skill_config if args.four_skill else make_open_config
    policy = policy_config.create_trained_policy(
        make_config(finetune_mode=args.mode), args.checkpoint, default_prompt=args.default_prompt
    )
    metadata = dict(policy.metadata)
    metadata.update(
        {
            "project": "VLA-TidyBench",
            "policy": f"pi0.5-drawer-{args.mode}",
            "checkpoint": str(args.checkpoint),
        }
    )
    WebsocketPolicyServer(policy, args.host, args.port, metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
