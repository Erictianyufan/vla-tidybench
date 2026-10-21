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


def compose_policy_first(
    policy_raw: ArrayLike,
    dls_reference_raw: ArrayLike,
    *,
    max_correction_norm: float,
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

    policy = policy.copy()
    policy[:6] = np.clip(policy[:6], -1.0, 1.0)
    policy[6] = 1.0 if policy[6] >= 0.0 else -1.0
    delta = reference[:6] - policy[:6]
    norm = float(np.linalg.norm(delta))
    scale = min(1.0, max_correction_norm / norm) if norm > 0.0 else 0.0
    correction = np.zeros(7, dtype=np.float32)
    correction[:6] = delta * scale
    executed = policy.copy()
    executed[:6] = np.clip(policy[:6] + correction[:6], -1.0, 1.0)
    return PolicyFirstAction(policy, correction, executed)
