#!/usr/bin/env python3
"""Split successful home full-task recordings into four training-only skills.

This derives language/action pairs from a continuous recorded episode. The
segments are not standalone physical rollouts and must never be used as
evaluation or replay evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

PHASE_TO_SKILL = {
    **dict.fromkeys(
        (
            "SETTLE",
            "HANDLE_APPROACH",
            "HANDLE_ALIGN",
            "HANDLE_CLOSE",
            "DRAWER_PULL",
            "HANDLE_RELEASE",
            "HANDLE_RETREAT",
        ),
        "open",
    ),
    **dict.fromkeys(("PICK_ABOVE", "PICK_DESCEND", "PICK_CLOSE", "PICK_LIFT", "PICK_HOLD"), "pick"),
    **dict.fromkeys(
        ("PLACE_CLEARANCE", "PLACE_ABOVE", "PLACE_DESCEND", "PLACE_INSERT", "PLACE_RELEASE", "PLACE_RETREAT"),
        "place",
    ),
    **dict.fromkeys(
        ("CLOSE_APPROACH", "CLOSE_ALIGN", "CLOSE_GRASP", "DRAWER_PUSH", "CLOSE_RELEASE", "FINAL_RETREAT", "DONE"),
        "close",
    ),
}
PROMPTS = {
    "open": "open the top drawer",
    "pick": "pick up the medicine bottle from the home table",
    "place": "put the medicine bottle into the top drawer",
    "close": "close the top drawer",
}
SKILLS = tuple(PROMPTS)
OBSERVATIONS = ("table_cam", "wrist_cam", "joint_pos", "joint_vel")


def skill_slices(phases: np.ndarray) -> dict[str, slice]:
    names = [phase.decode() if isinstance(phase, bytes) else str(phase) for phase in phases]
    try:
        skills = [PHASE_TO_SKILL[name] for name in names]
    except KeyError as exc:
        raise ValueError(f"unexpected full-task phase: {exc}") from exc
    boundaries = [0] + [i for i in range(1, len(skills)) if skills[i] != skills[i - 1]] + [len(skills)]
    sequence = [skills[start] for start in boundaries[:-1]]
    if sequence != list(SKILLS):
        raise ValueError(f"expected open/pick/place/close in order, got {sequence}")
    return {skill: slice(start, end) for skill, start, end in zip(SKILLS, boundaries[:-1], boundaries[1:], strict=True)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        parser.error("output or manifest already exists; refusing to overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    indices: dict[str, list[int]] = {skill: [] for skill in SKILLS}
    output_index = 0
    with h5py.File(args.output, "x") as destination:
        destination.attrs["format_version"] = 1
        destination.attrs["derived_training_only"] = True
        destination.attrs["source_files"] = json.dumps([str(path.resolve()) for path in args.source])
        dest_data = destination.create_group("data")
        for source_path in args.source:
            with h5py.File(source_path, "r") as source:
                if source["data"].attrs.get("skill") != "full":
                    raise ValueError(f"not a full-task source: {source_path}")
                names = sorted(source["data"], key=lambda name: int(name.removeprefix("demo_")))
                for name in names:
                    parent = source["data"][name]
                    if not bool(parent.attrs.get("success", False)):
                        raise ValueError(f"unsuccessful source episode: {source_path}::{name}")
                    slices = skill_slices(np.asarray(parent["teacher_phase"]))
                    length = len(parent["actions"])
                    if len(parent["teacher_phase"]) != length:
                        raise ValueError(f"phase/action length mismatch: {source_path}::{name}")
                    for skill in SKILLS:
                        span = slices[skill]
                        demo = dest_data.create_group(f"demo_{output_index}")
                        demo.attrs["success"] = True
                        demo.attrs["skill"] = skill
                        demo.attrs["language_instruction"] = PROMPTS[skill]
                        demo.attrs["source"] = "derived_home_teacher_training_segment"
                        demo.attrs["parent_file"] = str(source_path.resolve())
                        demo.attrs["parent_episode"] = name
                        demo.attrs["start_step"] = span.start
                        demo.attrs["end_step"] = span.stop
                        demo.create_dataset("actions", data=parent["actions"][span], compression="gzip")
                        obs = demo.create_group("obs")
                        for key in OBSERVATIONS:
                            obs.create_dataset(key, data=parent["obs"][key][span], compression="gzip")
                        indices[skill].append(output_index)
                        output_index += 1
    manifest = {
        "schema_version": 1,
        "repo_id": args.repo_id,
        "fps": 20,
        "image_writer_threads": 8,
        "training_only": True,
        "sources": [
            {"file": args.output.name, "prompt": PROMPTS[skill], "episode_indices": indices[skill]} for skill in SKILLS
        ],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"segments": output_index, "per_skill": {key: len(value) for key, value in indices.items()}}, indent=2
        )
    )


if __name__ == "__main__":
    main()
