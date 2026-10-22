#!/usr/bin/env python3
"""Probe a live policy on saved RGB/proprioception at selected teacher phases.

Phase labels select diagnostic frames only; they are never sent to the policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from vla_tidybench.policy_bridge.action_adapter import ActionAdapter
from vla_tidybench.policy_bridge.websocket_client import PolicyClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="open the top drawer")
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--phases", nargs="+", help="probe only these teacher phases")
    args = parser.parse_args()
    output = []
    adapter = ActionAdapter()
    with h5py.File(args.recording, "r") as recording, PolicyClient(args.host, args.port, timeout_s=120) as policy:
        episode = recording["data"][args.episode]
        phases = np.asarray(episode["teacher_phase"]).astype(str)
        for phase in args.phases or (
            "HANDLE_APPROACH",
            "HANDLE_CLOSE",
            "DRAWER_PULL",
            "PICK_ABOVE",
            "PICK_DESCEND",
            "PICK_CLOSE",
            "PLACE_ABOVE",
            "PLACE_INSERT",
            "PLACE_DESCEND",
            "PLACE_RELEASE",
            "PLACE_RETREAT",
            "CLOSE_GRASP",
        ):
            indices = np.flatnonzero(phases == phase)
            if len(indices) == 0:
                continue
            positions = (0, min(20, len(indices) - 1), min(60, len(indices) - 1), len(indices) // 2, len(indices) - 1)
            for position in sorted(set(positions)):
                index = int(indices[position])
                obs = episode["obs"]
                state = np.concatenate((obs["joint_pos"][index], obs["joint_vel"][index]), dtype=np.float32)
                response = policy.infer(
                    {
                        "observation/image": np.asarray(obs["table_cam"][index], dtype=np.uint8),
                        "observation/wrist_image": np.asarray(obs["wrist_cam"][index], dtype=np.uint8),
                        "observation/state": state,
                        "prompt": args.prompt,
                    }
                )
                actions = np.asarray(response["actions"], dtype=np.float32)
                label_physical = adapter.from_isaac(episode["actions"][index])
                model_physical = actions[0]
                label_motion = label_physical[:6]
                model_motion = model_physical[:6]
                cosine = float(
                    np.dot(label_motion, model_motion)
                    / max(np.linalg.norm(label_motion) * np.linalg.norm(model_motion), 1e-8)
                )
                output.append(
                    {
                        "phase": phase,
                        "frame": index,
                        "label_gripper": float(episode["actions"][index, 6]),
                        "model_gripper_first4": actions[:4, 6].round(3).tolist(),
                        "model_gripper_mean16": float(actions[:, 6].mean()),
                        "label_motion": label_motion.round(4).tolist(),
                        "model_motion_first": model_motion.round(4).tolist(),
                        "motion_cosine": cosine,
                    }
                )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
