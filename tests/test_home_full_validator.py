from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py", exc_type=ImportError)
from scripts.validate_home_full_pi05 import validate  # noqa: E402


def _recording(path, *, correction_size: float = 0.04) -> None:
    with h5py.File(path, "w") as output:
        data = output.create_group("data")
        data.attrs["successful_episodes"] = 1
        data.attrs["scene_variant"] = "home_tabletop_full"
        episode = data.create_group("demo_0")
        episode.attrs["control_mode"] = "policy_first"
        episode.attrs["source"] = "pi0.5_policy_with_bounded_truth_guided_dls"
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
