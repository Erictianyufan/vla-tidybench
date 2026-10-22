from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py", exc_type=ImportError)
validator_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_home_full_pi05.py"
spec = importlib.util.spec_from_file_location("home_full_validator", validator_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
validate = module.validate


def _recording(path, *, correction_size: float = 0.04) -> None:
    with h5py.File(path, "w") as output:
        data = output.create_group("data")
        data.attrs["successful_episodes"] = 1
        data.attrs["scene_variant"] = "home_tabletop_full"
        episode = data.create_group("demo_0")
        episode.attrs["control_mode"] = "policy_first"
        episode.attrs["source"] = "pi0.5_policy_with_bounded_truth_guided_dls"
        episode.attrs["state_convention"] = "joint_pos_rel_joint_vel_rel"
        episode.attrs["policy"] = "pi0.5-drawer-full-primary+bounded-dls"
        episode.attrs["policy_checkpoint_sha256"] = "a" * 64
        episode.attrs["dls_max_correction_norm"] = 0.1
        episode.attrs["inference_calls"] = 4
        episode.attrs["mean_infer_ms"] = 25.0
        episode.create_dataset("hero_cam", data=np.zeros((4, 2, 2, 3), dtype=np.uint8))
        obs = episode.create_group("obs")
        obs.create_dataset("table_cam", data=np.zeros((4, 2, 2, 3), dtype=np.uint8))
        obs.create_dataset("wrist_cam", data=np.zeros((4, 2, 2, 3), dtype=np.uint8))
        model = np.tile(np.array([0.2, 0, 0, 0, 0, 0, -1], dtype=np.float32), (4, 1))
        correction = np.tile(np.array([correction_size, 0, 0, 0, 0, 0, 0], dtype=np.float32), (4, 1))
        for name, values in (
            ("policy_actions", model),
            ("policy_raw_actions", model),
            ("dls_reference_actions", model + correction),
            ("correction_actions", correction),
            ("executed_actions", model + correction),
        ):
            episode.create_dataset(name, data=values)
        strings = np.asarray(["open", "pick", "place", "close"], dtype=h5py.string_dtype())
        for name in ("policy_skills", "policy_prompts", "teacher_phase"):
            episode.create_dataset(name, data=strings)


def test_validates_one_policy_first_full_episode(tmp_path) -> None:
    path = tmp_path / "success.hdf5"
    _recording(path)
    summary = validate(path)
    assert summary["frames"] == 4
    assert summary["model_motion_fraction"] > 0.6


def test_rejects_excessive_dls_correction(tmp_path) -> None:
    path = tmp_path / "bad.hdf5"
    _recording(path, correction_size=0.12)
    with pytest.raises(ValueError, match="exceeded"):
        validate(path)


def test_rejects_missing_checkpoint_identity(tmp_path) -> None:
    path = tmp_path / "missing-checkpoint.hdf5"
    _recording(path)
    with h5py.File(path, "r+") as recording:
        del recording["data/demo_0"].attrs["policy_checkpoint_sha256"]
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        validate(path)


def test_rejects_unverified_joint_state_convention(tmp_path) -> None:
    path = tmp_path / "wrong-state.hdf5"
    _recording(path)
    with h5py.File(path, "r+") as recording:
        del recording["data/demo_0"].attrs["state_convention"]
    with pytest.raises(ValueError, match="relative-joint"):
        validate(path)


def test_audits_model_only_gripper_confirmation(tmp_path) -> None:
    path = tmp_path / "debounced.hdf5"
    _recording(path)
    with h5py.File(path, "r+") as recording:
        episode = recording["data/demo_0"]
        episode.attrs["policy_gripper_close_confirm_steps"] = 2
        episode.attrs["policy_gripper_open_confirm_steps"] = 2
        episode.create_dataset("policy_model_raw_actions", data=episode["policy_raw_actions"][:])
        episode["policy_raw_actions"][0, 6] = 1.0
        episode["executed_actions"][0, 6] = 1.0
    assert validate(path)["gripper_filtered_frames"] == 1
    with h5py.File(path, "r+") as recording:
        recording["data/demo_0/policy_raw_actions"][0, 6] = -1.0
        recording["data/demo_0/executed_actions"][0, 6] = -1.0
    with pytest.raises(ValueError, match="cannot be reproduced"):
        validate(path)


def test_rejects_forged_bounded_dls_direction(tmp_path) -> None:
    path = tmp_path / "forged-dls.hdf5"
    _recording(path)
    with h5py.File(path, "r+") as recording:
        episode = recording["data/demo_0"]
        episode["correction_actions"][0, 0] = -0.04
        episode["executed_actions"][0, 0] = 0.16
    with pytest.raises(ValueError, match="does not match"):
        validate(path)


def test_validates_skill_routed_bounded_correction(tmp_path) -> None:
    path = tmp_path / "routed.hdf5"
    _recording(path)
    with h5py.File(path, "r+") as recording:
        recording["data/demo_0"].attrs["dls_correction_mode"] = "skill_router"
    assert validate(path)["dls_correction_mode"] == "skill_router"
