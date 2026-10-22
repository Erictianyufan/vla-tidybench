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


def _debounced_gripper(model: np.ndarray, *, close_after: int, open_after: int) -> np.ndarray:
    if min(close_after, open_after) < 1:
        raise ValueError("invalid gripper confirmation steps")
    state = 1.0
    pending = 0
    result = np.empty(len(model), dtype=np.float32)
    for index, value in enumerate(model):
        requested = 1.0 if value >= 0 else -1.0
        if requested == state:
            pending = 0
        else:
            pending += 1
            if pending >= (close_after if requested < 0 else open_after):
                state = requested
                pending = 0
        result[index] = state
    return result


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
        if _string(episode.attrs.get("state_convention", "")) != "joint_pos_rel_joint_vel_rel":
            raise ValueError("policy state is not confirmed to match the relative-joint training convention")
        policy_name = _string(episode.attrs.get("policy", ""))
        checkpoint_sha256 = _string(episode.attrs.get("policy_checkpoint_sha256", ""))
        if not policy_name.startswith("pi0.5-drawer-"):
            raise ValueError("missing pi0.5 policy identity")
        if len(checkpoint_sha256) != 64 or any(char not in "0123456789abcdef" for char in checkpoint_sha256):
            raise ValueError("missing valid pi0.5 checkpoint SHA-256")
        names = (
            "hero_cam",
            "obs/table_cam",
            "obs/wrist_cam",
            "policy_actions",
            "policy_raw_actions",
            "dls_reference_actions",
            "correction_actions",
            "executed_actions",
            "policy_skills",
            "policy_prompts",
            "teacher_phase",
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
        mode = _string(episode.attrs.get("dls_correction_mode", "isotropic"))
        reference = np.asarray(episode["dls_reference_actions"], dtype=np.float32)
        if reference.shape != model.shape or not np.isfinite(reference).all():
            raise ValueError("DLS reference trace is malformed")
        delta = reference[:, :6] - np.clip(model[:, :6], -1.0, 1.0)
        expected_correction = np.zeros_like(delta)
        if mode == "skill_router":
            isotropic_mask = np.isin(stages, ("open", "close"))
            translation_mask = np.isin(stages, ("pick", "place"))
            if not np.all(isotropic_mask | translation_mask):
                raise ValueError("unknown skill in DLS correction router")
        elif mode == "isotropic":
            isotropic_mask = np.ones(frame_count, dtype=bool)
            translation_mask = np.zeros(frame_count, dtype=bool)
        elif mode == "translation_first":
            isotropic_mask = np.zeros(frame_count, dtype=bool)
            translation_mask = np.ones(frame_count, dtype=bool)
        else:
            raise ValueError("unknown DLS correction mode")
        norms = np.linalg.norm(delta, axis=1)
        scale = np.minimum(1.0, limit / np.maximum(norms, 1e-8))
        expected_correction[isotropic_mask] = (delta * scale[:, None])[isotropic_mask]
        translation_norms = np.linalg.norm(delta[:, :3], axis=1)
        translation_scale = np.minimum(1.0, 0.65 * limit / np.maximum(translation_norms, 1e-8))
        translation_correction = delta[:, :3] * translation_scale[:, None]
        remaining = np.sqrt(np.maximum(0.0, limit**2 - np.sum(translation_correction**2, axis=1)))
        rotation_norms = np.linalg.norm(delta[:, 3:], axis=1)
        rotation_scale = np.minimum(1.0, remaining / np.maximum(rotation_norms, 1e-8))
        expected_correction[translation_mask, :3] = translation_correction[translation_mask]
        expected_correction[translation_mask, 3:] = (delta[:, 3:] * rotation_scale[:, None])[translation_mask]
        if not np.allclose(correction[:, :6], expected_correction, atol=1e-5):
            raise ValueError("logged DLS correction does not match the declared bounded rule")
        if not np.allclose(executed[:, :6], np.clip(model[:, :6] + correction[:, :6], -1.0, 1.0), atol=1e-5):
            raise ValueError("executed motion is not policy action plus the bounded correction")
        if not np.array_equal(executed[:, 6], model[:, 6]):
            raise ValueError("policy did not control the gripper throughout the episode")
        gripper_filter = "none"
        filtered_frames = 0
        if "policy_model_raw_actions" in episode:
            model_unfiltered = np.asarray(episode["policy_model_raw_actions"], dtype=np.float32)
            if model_unfiltered.shape != model.shape or not np.isfinite(model_unfiltered).all():
                raise ValueError("raw model action trace is missing or malformed")
            if not np.allclose(model_unfiltered[:, :6], model[:, :6], atol=1e-6):
                raise ValueError("gripper filter changed the model motion axes")
            close_after = int(episode.attrs["policy_gripper_close_confirm_steps"])
            open_after = int(episode.attrs["policy_gripper_open_confirm_steps"])
            expected_gripper = _debounced_gripper(
                model_unfiltered[:, 6], close_after=close_after, open_after=open_after
            )
            if not np.array_equal(model[:, 6], expected_gripper):
                raise ValueError("gripper command cannot be reproduced from model-only debouncing")
            filtered_frames = int(
                np.count_nonzero(expected_gripper != np.where(model_unfiltered[:, 6] >= 0, 1.0, -1.0))
            )
            gripper_filter = f"model-only temporal confirmation close={close_after} open={open_after}"
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
            "policy": policy_name,
            "policy_checkpoint_sha256": checkpoint_sha256,
            "max_dls_correction_norm": float(np.max(correction_norms)),
            "dls_correction_mode": mode,
            "model_motion_fraction": model_fraction,
            "gripper_filter": gripper_filter,
            "gripper_filtered_frames": filtered_frames,
            "mean_inference_ms": float(episode.attrs["mean_infer_ms"]),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.recording), indent=2))


if __name__ == "__main__":
    main()
