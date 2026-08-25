#!/usr/bin/env python3
"""Create deterministic nominal train/validation and replay-mixed hard-data manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random

import h5py


@dataclass(frozen=True)
class Episode:
    file: str
    path: Path
    index: int
    name: str
    prompt: str
    frames: int
    role: str

    @property
    def key(self) -> tuple[Path, str]:
        return self.path, self.name


def sorted_episode_names(data: h5py.Group) -> list[str]:
    try:
        return sorted(data.keys(), key=lambda name: int(name.removeprefix("demo_")))
    except ValueError as error:
        raise ValueError("episodes must use demo_<integer> names") from error


def inspect_episode(path: Path, data: h5py.Group, name: str) -> int:
    episode = data[name]
    if not bool(episode.attrs.get("success", False)):
        raise ValueError(f"supervised manifests accept successful/recovered episodes only: {path}::{name}")
    if "actions" not in episode or "obs" not in episode:
        raise ValueError(f"missing actions/obs: {path}::{name}")
    frames = int(episode["actions"].shape[0])
    expected = {
        "table_cam": (frames, 200, 200, 3),
        "wrist_cam": (frames, 200, 200, 3),
        "joint_pos": (frames, 9),
        "joint_vel": (frames, 9),
    }
    observations = episode["obs"]
    for key, shape in expected.items():
        if key not in observations or tuple(observations[key].shape) != shape:
            actual = None if key not in observations else tuple(observations[key].shape)
            raise ValueError(f"{path}::{name} obs/{key} has shape {actual}, expected {shape}")
    if tuple(episode["actions"].shape) != (frames, 7) or frames < 1:
        raise ValueError(f"{path}::{name} actions must have shape (T, 7) with T > 0")
    return frames


def load_source_config(config_path: Path, data_root: Path, *, role: str) -> tuple[dict[str, object], list[Episode]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported config schema: {config_path}")
    default_prompt = str(config.get("prompt", "")).strip()
    data_root = data_root.expanduser().resolve()
    episodes: list[Episode] = []
    seen: set[tuple[Path, str]] = set()
    for source in config.get("sources", []):
        source_file = str(source["file"])
        path = (data_root / source_file).resolve()
        if not path.is_relative_to(data_root):
            raise ValueError(f"source escapes data root: {source_file}")
        if not path.is_file():
            raise FileNotFoundError(path)
        prompt = str(source.get("prompt", default_prompt)).strip()
        if not prompt:
            raise ValueError(f"missing prompt for {source_file}")
        with h5py.File(path, "r") as dataset:
            if int(dataset.attrs.get("format_version", -1)) != 1 or "data" not in dataset:
                raise ValueError(f"unsupported HDF5 format: {path}")
            names = sorted_episode_names(dataset["data"])
            indices = source.get("episode_indices", list(range(len(names))))
            for raw_index in indices:
                index = int(raw_index)
                if index < 0 or index >= len(names):
                    raise IndexError(f"episode index {index} out of range for {path}")
                name = names[index]
                key = (path, name)
                if key in seen:
                    raise ValueError(f"duplicate episode in {role} config: {path}::{name}")
                seen.add(key)
                episodes.append(Episode(source_file, path, index, name, prompt, inspect_episode(path, dataset["data"], name), role))
    if not episodes:
        raise ValueError(f"no episodes selected by {config_path}")
    return config, episodes


def split_nominal(
    episodes: list[Episode],
    *,
    seed: int,
    validation_fraction: float,
    min_train_per_prompt: int,
    min_validation_per_prompt: int,
) -> tuple[list[Episode], list[Episode]]:
    groups: dict[str, list[Episode]] = defaultdict(list)
    for episode in episodes:
        groups[episode.prompt].append(episode)
    train: list[Episode] = []
    validation: list[Episode] = []
    for prompt, selected in sorted(groups.items()):
        if len(selected) < min_train_per_prompt + min_validation_per_prompt:
            raise ValueError(
                f"prompt {prompt!r} has {len(selected)} episodes; needs at least "
                f"{min_train_per_prompt + min_validation_per_prompt}"
            )
        shuffled = sorted(selected, key=lambda episode: (episode.file, episode.index))
        prompt_seed = int.from_bytes(hashlib.sha256(f"{seed}\0{prompt}".encode()).digest()[:8], "big")
        random.Random(prompt_seed).shuffle(shuffled)
        validation_count = max(min_validation_per_prompt, round(len(shuffled) * validation_fraction))
        validation_count = min(validation_count, len(shuffled) - min_train_per_prompt)
        validation.extend(shuffled[:validation_count])
        train.extend(shuffled[validation_count:])
    return train, validation


def hard_mix(
    nominal_train: list[Episode],
    hard: list[Episode],
    *,
    seed: int,
    nominal_replay_ratio: float,
    min_hard_per_prompt: int,
) -> list[Episode]:
    if {episode.key for episode in nominal_train} & {episode.key for episode in hard}:
        raise ValueError("hard-recovery inputs overlap nominal training episodes")
    nominal_groups: dict[str, list[Episode]] = defaultdict(list)
    hard_groups: dict[str, list[Episode]] = defaultdict(list)
    for episode in nominal_train:
        nominal_groups[episode.prompt].append(episode)
    for episode in hard:
        hard_groups[episode.prompt].append(episode)
    missing_prompts = set(nominal_groups) ^ set(hard_groups)
    if missing_prompts:
        raise ValueError(f"nominal/hard prompt sets differ: {sorted(missing_prompts)}")

    mixed = list(hard)
    for prompt, hard_episodes in sorted(hard_groups.items()):
        if len(hard_episodes) < min_hard_per_prompt:
            raise ValueError(f"prompt {prompt!r} has only {len(hard_episodes)} hard episodes")
        replay_count = math.ceil(len(hard_episodes) * nominal_replay_ratio)
        candidates = sorted(nominal_groups[prompt], key=lambda episode: (episode.file, episode.index))
        if replay_count > len(candidates):
            raise ValueError(f"prompt {prompt!r} needs {replay_count} nominal replay episodes, has {len(candidates)}")
        prompt_seed = int.from_bytes(hashlib.sha256(f"hard\0{seed}\0{prompt}".encode()).digest()[:8], "big")
        random.Random(prompt_seed).shuffle(candidates)
        mixed.extend(candidates[:replay_count])
    return mixed


def manifest(repo_id: str, config: dict[str, object], episodes: list[Episode], *, split: str) -> dict[str, object]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for episode in episodes:
        groups[(episode.file, episode.prompt, episode.role)].append(episode.index)
    return {
        "schema_version": 1,
        "repo_id": repo_id,
        "fps": int(config["fps"]),
        "image_writer_threads": int(config.get("image_writer_threads", 8)),
        "split": split,
        "sources": [
            {
                "file": file,
                "episode_indices": sorted(indices),
                "prompt": prompt,
                "role": role,
            }
            for (file, prompt, role), indices in sorted(groups.items())
        ],
    }


def count_by_prompt(episodes: list[Episode]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for prompt in sorted({episode.prompt for episode in episodes}):
        selected = [episode for episode in episodes if episode.prompt == prompt]
        result[prompt] = {"episodes": len(selected), "frames": sum(episode.frames for episode in selected)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-config", type=Path, required=True)
    parser.add_argument("--hard-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-prefix", required=True, help="e.g. owner/vla_tidybench_drawer_v1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--min-train-per-prompt", type=int, default=8)
    parser.add_argument("--min-validation-per-prompt", type=int, default=2)
    parser.add_argument("--min-hard-per-prompt", type=int, default=2)
    parser.add_argument("--nominal-replay-ratio", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in (0, 1)")
    if min(args.min_train_per_prompt, args.min_validation_per_prompt, args.min_hard_per_prompt) < 1:
        parser.error("minimum episode counts must be positive")
    if args.nominal_replay_ratio <= 0:
        parser.error("--nominal-replay-ratio must be positive")

    main_config, nominal = load_source_config(args.main_config, args.data_root, role="nominal")
    hard_config, hard = load_source_config(args.hard_config, args.data_root, role="hard_recovery")
    if int(main_config["fps"]) != int(hard_config["fps"]):
        parser.error("main and hard configs must use the same fps")
    overlap = {episode.key for episode in nominal} & {episode.key for episode in hard}
    if overlap:
        parser.error(f"main and hard configs overlap by {len(overlap)} episodes")
    train, validation = split_nominal(
        nominal,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        min_train_per_prompt=args.min_train_per_prompt,
        min_validation_per_prompt=args.min_validation_per_prompt,
    )
    mixed = hard_mix(
        train,
        hard,
        seed=args.seed,
        nominal_replay_ratio=args.nominal_replay_ratio,
        min_hard_per_prompt=args.min_hard_per_prompt,
    )
    assignments = {
        "main_train.json": manifest(f"{args.repo_prefix}_train", main_config, train, split="train"),
        "main_validation.json": manifest(f"{args.repo_prefix}_validation", main_config, validation, split="validation"),
        "hard_mix_train.json": manifest(f"{args.repo_prefix}_hard_mix", main_config, mixed, split="hard_mix_train"),
    }
    canonical = json.dumps(assignments, sort_keys=True, separators=(",", ":")).encode()
    audit = {
        "schema_version": 1,
        "split_id": hashlib.sha256(canonical).hexdigest(),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "nominal_replay_ratio": args.nominal_replay_ratio,
        "main_train": count_by_prompt(train),
        "main_validation": count_by_prompt(validation),
        "hard_recovery": count_by_prompt(hard),
        "hard_mix_train": count_by_prompt(mixed),
        "leakage": {
            "train_validation_overlap": len({episode.key for episode in train} & {episode.key for episode in validation}),
            "nominal_hard_overlap": len({episode.key for episode in nominal} & {episode.key for episode in hard}),
        },
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {**assignments, "split_audit.json": audit}
    existing = [output_dir / name for name in outputs if (output_dir / name).exists()]
    if existing and not args.overwrite:
        parser.error("outputs exist; pass --overwrite: " + ", ".join(map(str, existing)))
    for name, payload in outputs.items():
        (output_dir / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
