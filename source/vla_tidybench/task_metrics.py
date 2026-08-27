"""Auditable simulator-truth success predicates for drawer skills.

These metrics consume privileged state only after an action is executed. They
must never be included in the policy observation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

SUCCESS_PREDICATE_VERSION = "drawer_skill_v2_relative_stable"
FORMAL_SUCCESS_HOLD_STEPS = 5
OPEN_DRAWER_THRESHOLD_M = 0.30
CLOSE_DRAWER_THRESHOLD_M = 0.04
PICK_MIN_LIFT_M = 0.08
GRIPPER_CLOSED_MAX_WIDTH_M = 0.06
GRIPPER_OPEN_MIN_WIDTH_M = 0.06


def _xyz(value: ArrayLike, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite xyz vector")
    return result


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def object_in_open_drawer(object_xyz: ArrayLike, handle_x: float) -> bool:
    """Return whether the bottle center is inside the open top drawer."""

    obj = _xyz(object_xyz, "object_xyz")
    front_x = _finite(handle_x, "handle_x")
    return bool(0.68 < obj[2] < 0.86 and obj[0] > front_x + 0.023 and abs(obj[1]) < 0.26)


def object_in_closed_drawer(object_xyz: ArrayLike) -> bool:
    """Conservative scene bounds for a bottle retained by the closed drawer."""

    obj = _xyz(object_xyz, "object_xyz")
    return bool(0.68 < obj[2] < 0.86 and 0.85 < obj[0] < 1.08 and abs(obj[1]) < 0.26)


def drawer_skill_success(
    skill: str,
    *,
    initial_drawer_m: float,
    initial_object_xyz: ArrayLike,
    drawer_m: float,
    object_xyz: ArrayLike,
    handle_x: float,
    finger_positions: ArrayLike,
) -> bool:
    """Evaluate one instantaneous task state against a frozen initial state."""

    initial_drawer = _finite(initial_drawer_m, "initial_drawer_m")
    current_drawer = _finite(drawer_m, "drawer_m")
    initial_object = _xyz(initial_object_xyz, "initial_object_xyz")
    current_object = _xyz(object_xyz, "object_xyz")
    fingers = np.asarray(finger_positions, dtype=np.float64)
    if fingers.shape != (2,) or not np.isfinite(fingers).all():
        raise ValueError("finger_positions must contain two finite joint positions")
    gripper_width = float(fingers.sum())

    if skill == "open":
        return current_drawer >= OPEN_DRAWER_THRESHOLD_M
    if skill == "pick":
        return bool(
            current_object[2] - initial_object[2] >= PICK_MIN_LIFT_M
            and gripper_width <= GRIPPER_CLOSED_MAX_WIDTH_M
        )
    if skill == "place":
        return object_in_open_drawer(current_object, handle_x) and gripper_width >= GRIPPER_OPEN_MIN_WIDTH_M
    if skill == "close":
        # A retained bottle moves with the drawer: object_x + drawer_q stays
        # nearly invariant as the drawer travels toward the cabinet.
        retained_error = abs(
            (current_object[0] + current_drawer) - (initial_object[0] + initial_drawer)
        )
        return bool(
            current_drawer <= CLOSE_DRAWER_THRESHOLD_M
            and 0.68 < current_object[2] < 0.86
            and abs(current_object[1]) < 0.26
            and retained_error <= 0.08
        )
    if skill == "full":
        return bool(
            current_drawer <= CLOSE_DRAWER_THRESHOLD_M
            and object_in_closed_drawer(current_object)
            and gripper_width >= GRIPPER_OPEN_MIN_WIDTH_M
        )
    raise ValueError(f"unsupported drawer skill: {skill!r}")
