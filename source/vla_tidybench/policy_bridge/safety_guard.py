"""Deployment-time checks applied after policy composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .action_adapter import ActionAdapter


@dataclass(frozen=True)
class SafetyLimits:
    max_translation_norm_m: float = 0.035
    max_rotation_norm_rad: float = 0.18


class SafetyGuard:
    """Reject non-finite commands and bound physical motion per control step."""

    def __init__(self, adapter: ActionAdapter | None = None, limits: SafetyLimits | None = None) -> None:
        self.adapter = adapter or ActionAdapter()
        self.limits = limits or SafetyLimits()

    def apply(self, action: ArrayLike) -> NDArray[np.float32]:
        vector = self.adapter.clip_physical(action)
        translation_norm = float(np.linalg.norm(vector[:3]))
        if translation_norm > self.limits.max_translation_norm_m:
            vector[:3] *= self.limits.max_translation_norm_m / translation_norm
        rotation_norm = float(np.linalg.norm(vector[3:6]))
        if rotation_norm > self.limits.max_rotation_norm_rad:
            vector[3:6] *= self.limits.max_rotation_norm_rad / rotation_norm
        return vector

    def to_isaac(self, action: ArrayLike) -> NDArray[np.float32]:
        return self.adapter.to_isaac(self.apply(action))

