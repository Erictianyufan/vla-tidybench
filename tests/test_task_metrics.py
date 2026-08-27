from __future__ import annotations

import numpy as np
import pytest
from vla_tidybench.task_metrics import drawer_skill_success

BASE = {
    "initial_drawer_m": 0.36,
    "initial_object_xyz": np.asarray([0.20, -0.20, 0.066]),
    "drawer_m": 0.36,
    "object_xyz": np.asarray([0.20, -0.20, 0.066]),
    "handle_x": 0.30,
    "finger_positions": np.asarray([0.04, 0.04]),
}


def metric(skill: str, **updates) -> bool:
    values = {**BASE, **updates}
    return drawer_skill_success(skill, **values)


def test_pick_requires_relative_lift_and_closed_gripper() -> None:
    lifted = np.asarray([0.20, -0.20, 0.185])
    assert metric("pick", object_xyz=lifted) is False
    assert metric("pick", object_xyz=lifted, finger_positions=np.asarray([0.024, 0.024])) is True
    assert metric(
        "pick",
        initial_object_xyz=np.asarray([0.20, -0.20, 0.70]),
        object_xyz=np.asarray([0.20, -0.20, 0.72]),
        finger_positions=np.asarray([0.024, 0.024]),
    ) is False


def test_place_requires_in_drawer_bottle_and_open_gripper() -> None:
    placed = np.asarray([0.325, -0.20, 0.712])
    assert metric("place", object_xyz=placed, finger_positions=np.asarray([0.04, 0.04])) is True
    assert metric("place", object_xyz=placed, finger_positions=np.asarray([0.024, 0.024])) is False
    assert metric("place", object_xyz=np.asarray([0.28, -0.20, 0.712])) is False


def test_close_requires_bottle_to_remain_in_drawer_frame() -> None:
    close_state = {
        "initial_drawer_m": 0.36,
        "initial_object_xyz": np.asarray([0.61, -0.107, 0.731]),
        "drawer_m": 0.001,
        "object_xyz": np.asarray([0.960, -0.107, 0.712]),
    }
    assert metric("close", **close_state) is True
    assert metric("close", **{**close_state, "object_xyz": np.asarray([0.70, -0.40, 0.05])}) is False


def test_open_and_full_scene_predicates() -> None:
    assert metric("open", drawer_m=0.31) is True
    assert metric("open", drawer_m=0.29) is False
    assert metric(
        "full",
        drawer_m=0.01,
        object_xyz=np.asarray([0.96, -0.10, 0.712]),
        finger_positions=np.asarray([0.04, 0.04]),
    ) is True


def test_nonfinite_metric_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        metric("pick", object_xyz=np.asarray([0.20, -0.20, np.nan]))
