"""Protocol-compatible policy server used before downloading model weights."""

from __future__ import annotations

import argparse
import logging

import numpy as np
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi_client.base_policy import BasePolicy


class ZeroPolicy(BasePolicy):
    """Return a short, safe no-motion chunk with an open gripper."""

    def __init__(self, horizon: int = 10) -> None:
        self.horizon = horizon

    def infer(self, obs: dict) -> dict:
        required = {"table_rgb", "wrist_rgb", "robot_state", "prompt", "episode_id", "step_id"}
        missing = required.difference(obs)
        if missing:
            raise ValueError(f"Missing observation fields: {sorted(missing)}")
        actions = np.zeros((self.horizon, 7), dtype=np.float32)
        actions[:, 6] = 1.0
        return {
            "actions": actions,
            "episode_id": str(obs["episode_id"]),
            "observation_step": int(obs["step_id"]),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()
    metadata = {"policy": "dummy-zero", "action_dim": 7, "action_horizon": args.horizon}
    WebsocketPolicyServer(ZeroPolicy(args.horizon), args.host, args.port, metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

