#!/usr/bin/env python3
"""Reject a home full-task video unless one policy-first episode is auditable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _string(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def validate(path: Path) -> dict[str, float | int | str]:
    with h5py.File(path, "r") as recording:
        data = recording["data"]
        if int(data.attrs.get("successful_episodes", 0)) != 1 or len(data) != 1:
            raise ValueError("expected exactly one successful, uncut episode")
        if _string(data.attrs.get("scene_variant", "")) != "home_tabletop_full":
            raise ValueError("recording is not the home tabletop full task")
        episode = next(iter(data.values()))
        if _string(episode.attrs.get("control_mode", "")) != "policy_first":
            raise ValueError("recording did not use policy-first control")
        if _string(episode.attrs.get("source", "")) != "pi0.5_policy_with_bounded_truth_guided_dls":
            raise ValueError("missing policy-first provenance")
        names = (
            "hero_cam", "obs/table_cam", "obs/wrist_cam", "policy_actions", "policy_raw_actions",
            "dls_reference_actions", "correction_actions", "executed_actions", "policy_skills",
            "policy_prompts", "teacher_phase",
        )
        frame_count = len(episode["hero_cam"])
        if frame_count < 4 or any(len(episode[name]) != frame_count for name in names):
            raise ValueError("three camera streams and action/phase traces must align frame by frame")
        stages = [_string(value) for value in episode["policy_skills"][:]]
        position = 0
        for required in ("open", "pick", "place", "close"):
            try:
                position = stages.index(required, position) + 1
            except ValueError as exc:
                raise ValueError(f"missing or out-of-order stage: {required}") from exc
        model = np.asarray(episode["policy_raw_actions"], dtype=np.float32)
        correction = np.asarray(episode["correction_actions"], dtype=np.float32)
        executed = np.asarray(episode["executed_actions"], dtype=np.float32)
        if model.shape != (frame_count, 7) or correction.shape != model.shape or executed.shape != model.shape:
            raise ValueError("action traces must be (frames, 7)")
        if not all(np.isfinite(array).all() for array in (model, correction, executed)):
            raise ValueError("action traces contain non-finite values")
        limit = float(episode.attrs["dls_max_correction_norm"])
        correction_norms = np.linalg.norm(correction[:, :6], axis=1)
        if np.max(correction_norms) > limit + 1e-5:
            raise ValueError("DLS correction exceeded its logged limit")
        if not np.allclose(executed[:, :6], np.clip(model[:, :6] + correction[:, :6], -1.0, 1.0), atol=1e-5):
            raise ValueError("executed motion is not policy action plus the bounded correction")
        if not np.array_equal(executed[:, 6], model[:, 6]):
            raise ValueError("policy did not control the gripper throughout the episode")
        model_motion = float(np.linalg.norm(model[:, :6], axis=1).sum())
        correction_motion = float(correction_norms.sum())
        model_fraction = model_motion / max(model_motion + correction_motion, 1e-8)
        if model_fraction < 0.60:
            raise ValueError(f"policy motion fraction {model_fraction:.3f} is below the 0.60 audit gate")
        inference_calls = int(episode.attrs["inference_calls"])
        if inference_calls < 4:
            raise ValueError("too few pi0.5 inference calls for the four-stage task")
        return {
            "frames": frame_count,
            "inference_calls": inference_calls,
            "scene_variant": "home_tabletop_full",
            "control_mode": "policy_first",
            "max_dls_correction_norm": float(np.max(correction_norms)),
            "model_motion_fraction": model_fraction,
            "mean_inference_ms": float(episode.attrs["mean_infer_ms"]),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.recording), indent=2))


if __name__ == "__main__":
    main()
