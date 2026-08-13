"""Round-trip test for the OpenPI WebSocket protocol and action contract."""

from __future__ import annotations

import argparse

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    client = WebsocketClientPolicy(args.host, args.port)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    response = client.infer(
        {
            "table_rgb": image,
            "wrist_rgb": image,
            "robot_state": np.zeros(18, dtype=np.float32),
            "prompt": "pick up the red cube",
            "episode_id": "smoke-episode",
            "step_id": 12,
        }
    )
    actions = np.asarray(response["actions"])
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise RuntimeError(f"Invalid policy response action shape: {actions.shape}")
    if response["episode_id"] != "smoke-episode" or response["observation_step"] != 12:
        raise RuntimeError("Policy response lost episode/step identity")
    print("VLA_TIDYBENCH_PROTOCOL_PASSED")
    print(f"metadata={client.get_server_metadata()}")
    print(f"action_shape={actions.shape} infer_ms={response['server_timing']['infer_ms']:.3f}")


if __name__ == "__main__":
    main()

