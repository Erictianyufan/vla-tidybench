#!/usr/bin/env python3
"""Measure model action agreement on a successful recorded expert episode.

This is a diagnostic, not a closed-loop success metric. It reads only the
deployable RGB/state fields and compares one-step action signs and errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from vla_tidybench.data.isaac_hdf5 import load_episode, sorted_episode_names
from vla_tidybench.policy_bridge.websocket_client import PolicyClient

PROMPTS = {
    "open": "open the top drawer",
    "pick": "pick up the medicine bottle from the home table",
    "place": "put the medicine bottle into the top drawer",
    "close": "close the top drawer",
}


def phase_skill(phase: str) -> str:
    if phase == "SETTLE" or phase.startswith("HANDLE_") or phase == "DRAWER_PULL":
        return "open"
    if phase.startswith("PICK_"):
        return "pick"
    if phase.startswith("PLACE_"):
        return "place"
    return "close"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")

    with h5py.File(args.recording, "r") as recording:
        name = sorted_episode_names(recording["data"])[0]
        phases = recording["data"][name]["teacher_phase"].asstr()[:]
    episode = load_episode(args.recording, name)
    if len(phases) != episode.length:
        raise ValueError("phase/action length mismatch")

    results: dict[str, list[tuple[bool, bool, float]]] = {}
    with PolicyClient(args.host, args.port, timeout_s=120.0) as client:
        for index in range(0, episode.length, args.stride):
            skill = phase_skill(phases[index])
            response = client.infer(
                {
                    "observation/image": episode.table_image[index],
                    "observation/wrist_image": episode.wrist_image[index],
                    "observation/state": episode.state[index],
                    "prompt": PROMPTS[skill],
                }
            )
            proposed = np.asarray(response["actions"], dtype=np.float32)[0]
            expert = episode.actions[index]
            results.setdefault(skill, []).append(
                (
                    bool(proposed[6] < 0),
                    bool(expert[6] < 0),
                    float(np.linalg.norm(proposed[:6] - expert[:6])),
                )
            )

    summary = {}
    for skill, rows in results.items():
        values = np.asarray(rows, dtype=np.float32)
        summary[skill] = {
            "samples": len(rows),
            "model_close_fraction": round(float(values[:, 0].mean()), 3),
            "expert_close_fraction": round(float(values[:, 1].mean()), 3),
            "gripper_sign_accuracy": round(float((values[:, 0] == values[:, 1]).mean()), 3),
            "mean_arm_action_error": round(float(values[:, 2].mean()), 4),
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
