#!/usr/bin/env python3
"""Repair metadata attached in lexicographic instead of numeric episode order.

Only move HDF5 links for ``teacher_phase`` and ``hero_cam``. Original actions,
observations, and video bytes are never copied or rewritten. This script is
specific to an audited 24-episode teacher file produced by the old collector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from split_home_full_for_training import skill_slices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    args = parser.parse_args()
    path = args.recording.resolve()
    if path.name != "home_full_teacher_024.hdf5":
        parser.error("repair is restricted to home_full_teacher_024.hdf5")
    audit_path = path.with_suffix(".metadata_repair.json")
    if audit_path.exists():
        parser.error(f"audit already exists: {audit_path}")

    with h5py.File(path, "r+") as recording:
        data = recording["data"]
        if len(data) != 24 or int(data.attrs.get("successful_episodes", -1)) != 24:
            raise ValueError("expected 24 successful episodes")
        if "phase_camera_metadata_repair" in data.attrs or "_metadata_repair_tmp" in recording:
            raise ValueError("already repaired or incomplete prior repair; inspect manually")
        source_names = list(data.keys())  # Actual old data.values() iteration order.
        target_names = [f"demo_{index}" for index in range(24)]
        if set(source_names) != set(target_names):
            raise ValueError("unexpected episode names")
        mapping = []
        for index, (source_name, target_name) in enumerate(zip(source_names, target_names, strict=True)):
            source, target = data[source_name], data[target_name]
            if source.attrs.get("source") != "scripted_truth_teacher":
                raise ValueError(f"not a teacher episode: {source_name}")
            actions = len(target["actions"])
            phases = len(source["teacher_phase"])
            hero = len(source["hero_cam"])
            if phases != actions or hero != actions:
                raise ValueError(f"mapping {source_name} -> {target_name} has lengths {phases}/{hero}/{actions}")
            skill_slices(np.asarray(source["teacher_phase"]))
            mapping.append({"index": index, "source": source_name, "target": target_name, "frames": actions})

        audit = {
            "schema_version": 1,
            "recording": str(path),
            "reason": "old collector zipped lexicographic HDF5 data.values() with numeric-order episode payloads",
            "moved_links_only": ["teacher_phase", "hero_cam"],
            "mapping": mapping,
        }
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

        temporary = recording.create_group("_metadata_repair_tmp")
        for item in mapping:
            holder = temporary.create_group(str(item["index"]))
            source_name = item["source"]
            for key in ("teacher_phase", "hero_cam"):
                recording.move(f"/data/{source_name}/{key}", f"{holder.name}/{key}")
        recording.flush()
        for item in mapping:
            target_name = item["target"]
            for key in ("teacher_phase", "hero_cam"):
                recording.move(f"{temporary.name}/{item['index']}/{key}", f"/data/{target_name}/{key}")
        del recording["_metadata_repair_tmp"]
        for target_name in target_names:
            episode = data[target_name]
            actions = len(episode["actions"])
            if len(episode["teacher_phase"]) != actions or len(episode["hero_cam"]) != actions:
                raise ValueError(f"post-repair length mismatch: {target_name}")
            skill_slices(np.asarray(episode["teacher_phase"]))
        data.attrs["phase_camera_metadata_repair"] = audit_path.name
        recording.flush()
    print(json.dumps({"repaired_episodes": 24, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
