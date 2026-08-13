#!/usr/bin/env python3
"""Convert a checked-in VLA-TidyBench data manifest to a local LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from vla_tidybench.data.isaac_hdf5 import ACTION_DIM
from vla_tidybench.data.isaac_hdf5 import STATE_DIM
from vla_tidybench.data.isaac_hdf5 import load_episode
from vla_tidybench.data.isaac_hdf5 import sorted_episode_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if int(config.get("schema_version", -1)) != 1:
        parser.error("unsupported conversion config schema_version")
    repo_id = str(config["repo_id"])
    prompt = str(config["prompt"]).strip()
    if not prompt:
        parser.error("prompt must be non-empty")

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if not args.overwrite:
            parser.error(f"output exists: {output_path}; pass --overwrite")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=int(config["fps"]),
        features={
            "image": {
                "dtype": "image",
                "shape": (200, 200, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (200, 200, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
        },
        image_writer_threads=int(config.get("image_writer_threads", 8)),
        image_writer_processes=0,
    )

    episode_count = 0
    frame_count = 0
    for source in config["sources"]:
        source_path = (args.data_root / source["file"]).resolve()
        with h5py.File(source_path, "r") as source_file:
            names = sorted_episode_names(source_file["data"])
        indices = source.get("episode_indices", list(range(len(names))))
        for index in indices:
            if index < 0 or index >= len(names):
                raise IndexError(f"episode index {index} out of range for {source_path}")
            name = names[index]
            episode = load_episode(source_path, name)
            for step in range(episode.length):
                dataset.add_frame(
                    {
                        "image": episode.table_image[step],
                        "wrist_image": episode.wrist_image[step],
                        "state": episode.state[step],
                        "actions": episode.actions[step],
                        "task": prompt,
                    }
                )
            dataset.save_episode()
            episode_count += 1
            frame_count += episode.length
            print(f"saved {source_path.name}::{name} ({episode.length} frames)", flush=True)

    print(f"converted {episode_count} episodes and {frame_count} frames into {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
