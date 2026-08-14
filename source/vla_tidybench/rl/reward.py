"""Auditable reward terms for residual specialists."""

from __future__ import annotations


def open_drawer_reward(
    *, previous_q: float, current_q: float, contact: bool, collision: bool, residual_norm: float, success: bool
) -> dict[str, float]:
    terms = {
        "progress": 5.0 * (current_q - previous_q),
        "contact": 0.02 if contact else 0.0,
        "collision": -0.25 if collision else 0.0,
        "residual": -0.01 * residual_norm * residual_norm,
        "success": 2.0 if success else 0.0,
        "time": -0.002,
    }
    terms["total"] = sum(terms.values())
    return terms


def pick_residual_reward(
    *,
    previous_distance: float,
    current_distance: float,
    previous_object_z: float,
    current_object_z: float,
    residual_norm: float,
    success: bool,
    truncated: bool,
) -> dict[str, float]:
    """Reward a bounded residual for correcting a biased PICK controller.

    Object pose is privileged training information: it is used by the reward,
    never by the residual actor observation.
    """

    terms = {
        "reach_progress": 8.0 * (previous_distance - current_distance),
        "reach_distance": -0.2 * current_distance,
        "lift_progress": 80.0 * max(current_object_z - previous_object_z, 0.0),
        "residual": -0.002 * residual_norm * residual_norm,
        "success": 10.0 if success else 0.0,
        "timeout": -2.0 if truncated and not success else 0.0,
        "time": -0.01,
    }
    terms["total"] = sum(terms.values())
    return terms
