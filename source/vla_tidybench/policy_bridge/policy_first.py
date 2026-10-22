"""Bounded DLS correction for a genuinely policy-first Isaac action."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PolicyFirstAction:
    policy_raw: NDArray[np.float32]
    correction_raw: NDArray[np.float32]
    executed_raw: NDArray[np.float32]


class ModelGripperDebouncer:
    """Suppress short gripper sign flips using only successive model outputs."""

    def __init__(self, *, close_after: int, open_after: int) -> None:
        if close_after < 1 or open_after < 1:
            raise ValueError("gripper confirmation steps must be positive")
        self.close_after = close_after
        self.open_after = open_after
        self.state = 1.0
        self.pending = 0

    def update(self, model_gripper: float) -> float:
        if not np.isfinite(model_gripper):
            raise ValueError("model gripper must be finite")
        requested = 1.0 if model_gripper >= 0 else -1.0
        if requested == self.state:
            self.pending = 0
        else:
            self.pending += 1
            required = self.close_after if requested < 0 else self.open_after
            if self.pending >= required:
                self.state = requested
                self.pending = 0
        return self.state


def compose_policy_first(
    policy_raw: ArrayLike,
    dls_reference_raw: ArrayLike,
    *,
    max_correction_norm: float,
    correction_mode: str = "isotropic",
) -> PolicyFirstAction:
    """Use the policy command as the base; limit DLS to a small 6D correction.

    The gripper always follows the policy. There is no hidden teacher fallback:
    if the model cannot finish a phase, the episode must fail its success gate.
    """

    policy = np.asarray(policy_raw, dtype=np.float32)
    reference = np.asarray(dls_reference_raw, dtype=np.float32)
    if policy.shape != (7,) or reference.shape != (7,):
        raise ValueError("policy and DLS reference must both have shape (7,)")
    if not np.isfinite(policy).all() or not np.isfinite(reference).all():
        raise ValueError("actions must be finite")
    if not 0.0 <= max_correction_norm <= 0.25:
        raise ValueError("max_correction_norm must be in [0, 0.25]")
    if correction_mode not in ("isotropic", "translation_first"):
        raise ValueError("unknown correction_mode")

    policy = policy.copy()
    policy[:6] = np.clip(policy[:6], -1.0, 1.0)
    policy[6] = 1.0 if policy[6] >= 0.0 else -1.0
    delta = reference[:6] - policy[:6]
    correction = np.zeros(7, dtype=np.float32)
    if correction_mode == "isotropic":
        norm = float(np.linalg.norm(delta))
        scale = min(1.0, max_correction_norm / norm) if norm > 0.0 else 0.0
        correction[:6] = delta * scale
    else:
        translation_norm = float(np.linalg.norm(delta[:3]))
        translation_budget = 0.65 * max_correction_norm
        translation_scale = min(1.0, translation_budget / translation_norm) if translation_norm > 0.0 else 0.0
        correction[:3] = delta[:3] * translation_scale
        rotation_budget = np.sqrt(max(0.0, max_correction_norm**2 - float(np.dot(correction[:3], correction[:3]))))
        rotation_norm = float(np.linalg.norm(delta[3:]))
        rotation_scale = min(1.0, rotation_budget / rotation_norm) if rotation_norm > 0.0 else 0.0
        correction[3:6] = delta[3:] * rotation_scale
    executed = policy.copy()
    executed[:6] = np.clip(policy[:6] + correction[:6], -1.0, 1.0)
    return PolicyFirstAction(policy, correction, executed)
