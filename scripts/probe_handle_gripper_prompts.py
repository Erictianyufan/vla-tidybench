#!/usr/bin/env python3
"""Probe whether language alone makes pi0.5 close its gripper at a drawer handle.

Reads deployable observations from one successful expert recording. This is an
offline diagnostic, not a closed-loop success metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from vla_tidybench.data.isaac_hdf5 import load_episode, sorted_episode_names
from vla_tidybench.policy_bridge.websocket_client import PolicyClient


PROMPTS = (
    "open the top drawer",
    "grasp the top drawer handle",
    "close the gripper on the top drawer handle and pull it open",
    "pull the top drawer open",
)
PHASES = {"HANDLE_CLOSE", "DRAWER_PULL"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("stride must be positive")

    with h5py.File(args.recording, "r") as recording:
        name = sorted_episode_names(recording["data"])[0]
        phases = recording["data"][name]["teacher_phase"].asstr()[:]
    episode = load_episode(args.recording, name)
    selected = [index for index, phase in enumerate(phases) if phase in PHASES][:: args.stride]
    if not selected:
        raise ValueError("recording contains no handle-close or drawer-pull frames")

    results = {}
    with PolicyClient(args.host, args.port, timeout_s=120.0) as client:
        for prompt in PROMPTS:
            predicted = []
            for index in selected:
                response = client.infer(
                    {
                        "observation/image": episode.table_image[index],
                        "observation/wrist_image": episode.wrist_image[index],
                        "observation/state": episode.state[index],
                        "prompt": prompt,
                    }
                )
                predicted.append(np.asarray(response["actions"], dtype=np.float32)[0])
            model = np.stack(predicted)
            expert = episode.actions[selected]
            results[prompt] = {
                "samples": len(selected),
                "model_close_fraction": round(float(np.mean(model[:, 6] < 0)), 3),
                "expert_close_fraction": round(float(np.mean(expert[:, 6] < 0)), 3),
                "mean_arm_action_error": round(
                    float(np.mean(np.linalg.norm(model[:, :6] - expert[:, :6], axis=1))), 4
                ),
            }
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
