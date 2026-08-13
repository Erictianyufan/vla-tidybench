"""Merge selected human and scripted HDF5 episodes without mutating raw files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def parse_source(value: str) -> tuple[Path, list[int] | None, str]:
    parts = value.split("::")
    path = Path(parts[0]).expanduser().resolve()
    indices = None if len(parts) < 2 or parts[1] in {"", "all"} else [int(item) for item in parts[1].split(",")]
    default_source = "unknown" if len(parts) < 3 else parts[2]
    return path, indices, default_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="PATH::all|0,1,2::DEFAULT_SOURCE; may be repeated",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            parser.error(f"output exists: {output}")
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    copied = 0
    total_steps = 0
    source_counts: dict[str, int] = {}
    env_args: str | None = None
    format_version: int | None = None

    with h5py.File(output, "w") as destination:
        destination_data = destination.create_group("data")
        for source_spec in args.source:
            path, requested_indices, default_source = parse_source(source_spec)
            with h5py.File(path, "r") as source_file:
                candidate_format_version = int(source_file.attrs.get("format_version", 0))
                if format_version is None:
                    format_version = candidate_format_version
                    destination.attrs["format_version"] = candidate_format_version
                elif candidate_format_version != format_version:
                    raise ValueError(f"dataset format-version mismatch in {path}")
                source_data = source_file["data"]
                candidate_env_args = source_data.attrs.get("env_args")
                if env_args is None:
                    env_args = candidate_env_args
                elif candidate_env_args != env_args:
                    raise ValueError(f"environment metadata mismatch in {path}")

                episode_names = sorted(source_data.keys(), key=lambda name: int(name.split("_")[-1]))
                indices = range(len(episode_names)) if requested_indices is None else requested_indices
                for index in indices:
                    if index < 0 or index >= len(episode_names):
                        raise IndexError(f"episode {index} is out of range for {path}")
                    source_episode = source_data[episode_names[index]]
                    if not bool(source_episode.attrs.get("success", False)):
                        raise ValueError(f"refusing to merge unsuccessful episode {path}::{index}")

                    target_name = f"demo_{copied}"
                    source_file.copy(source_episode, destination_data, name=target_name)
                    target_episode = destination_data[target_name]
                    source_label = str(target_episode.attrs.get("source", default_source))
                    target_episode.attrs["source"] = source_label
                    target_episode.attrs["source_file"] = path.name
                    target_episode.attrs["source_episode"] = episode_names[index]
                    source_counts[source_label] = source_counts.get(source_label, 0) + 1
                    total_steps += int(target_episode.attrs["num_samples"])
                    copied += 1

        destination_data.attrs["env_args"] = env_args or ""
        destination_data.attrs["total"] = total_steps
        destination_data.attrs["successful_episodes"] = copied
        destination_data.attrs["source_counts"] = json.dumps(source_counts, sort_keys=True)
        destination_data.attrs["merge_policy"] = "explicit replay-validated episode selection"

    print(f"merged {copied} successful episodes ({total_steps} steps) into {output}")
    print("source counts: " + json.dumps(source_counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
