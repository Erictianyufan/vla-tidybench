#!/usr/bin/env python3
"""Split successful home-tabletop full-task episodes into four skill episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from vla_tidybench.data.isaac_hdf5 import ACTION_DIM, STATE_DIM, load_episode, sorted_episode_names

PROMPTS = {
    "open": "open the top drawer",
    "pick": "pick up the medicine bottle from the home table",
    "place": "put the medicine bottle into the top drawer",
    "close": "close the top drawer",
}


def skill_for_phase(phase: str) -> str:
    if phase == "SETTLE" or phase.startswith("HANDLE_") or phase == "DRAWER_PULL":
        return "open"
    if phase.startswith("PICK_"):
        return "pick"
    if phase.startswith("PLACE_"):
        return "place"
    return "close"


def skill_segments(phases: np.ndarray) -> list[tuple[str, int, int]]:
    if len(phases) == 0:
        raise ValueError("empty phase trace")
    segments: list[tuple[str, int, int]] = []
    start = 0
    previous = skill_for_phase(str(phases[0]))
    for index in range(1, len(phases)):
        current = skill_for_phase(str(phases[index]))
        if current != previous:
            segments.append((previous, start, index))
            start, previous = index, current
    segments.append((previous, start, len(phases)))
    if [name for name, _, _ in segments] != ["open", "pick", "place", "close"]:
        raise ValueError(f"expected four ordered skill segments, got {segments}")
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("fps must be positive")
    output = HF_LEROBOT_HOME / args.repo_id
    if output.exists():
        parser.error(f"dataset exists; choose a new repo id: {output}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=args.fps,
        features={
            "image": {"dtype": "image", "shape": (200, 200, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (200, 200, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
        },
        image_writer_threads=8,
        image_writer_processes=0,
    )
    counts = {skill: 0 for skill in PROMPTS}
    for path in args.recordings:
        with h5py.File(path, "r") as recording:
            episode_names = sorted_episode_names(recording["data"])
        for name in episode_names:
            with h5py.File(path, "r") as recording:
                phases = recording["data"][name]["teacher_phase"].asstr()[:]
            episode = load_episode(path, name)
            if len(phases) != episode.length:
                raise ValueError(f"phase/action length mismatch in {path}::{name}")
            for skill, start, stop in skill_segments(phases):
                for index in range(start, stop):
                    dataset.add_frame(
                        {
                            "image": episode.table_image[index],
                            "wrist_image": episode.wrist_image[index],
                            "state": episode.state[index],
                            "actions": episode.actions[index],
                            "task": PROMPTS[skill],
                        }
                    )
                dataset.save_episode()
                counts[skill] += 1
                print(f"{path.name}::{name} {skill} {stop - start} frames", flush=True)
    print(f"saved {counts} to {output}")


if __name__ == "__main__":
    main()
