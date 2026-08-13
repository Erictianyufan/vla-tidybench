"""Test an OpenPI policy service from the unchanged Isaac Python runtime."""

from __future__ import annotations

import argparse

import numpy as np

from vla_tidybench.policy_bridge.websocket_client import PolicyClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    with PolicyClient(args.host, args.port) as client:
        response = client.infer(
            {
                "table_rgb": image,
                "wrist_rgb": image,
                "robot_state": np.zeros(18, dtype=np.float32),
                "prompt": "pick up the red cube",
                "episode_id": "isaac-client-smoke",
                "step_id": 3,
            }
        )
        actions = np.asarray(response["actions"])
        if actions.shape != (10, 7):
            raise RuntimeError(f"Expected (10, 7), got {actions.shape}")
        print("VLA_TIDYBENCH_ISAAC_CLIENT_PASSED")
        print(f"metadata={client.metadata} action_shape={actions.shape}")


if __name__ == "__main__":
    main()
