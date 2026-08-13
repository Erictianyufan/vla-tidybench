"""The single authoritative conversion between physical and Isaac actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ActionSpec:
    """Canonical Franka action limits and simulator scaling."""

    translation_limit_m: float = 0.025
    rotation_limit_rad: float = 0.12
    ik_relative_scale: float = 0.5
    gripper_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.translation_limit_m <= 0 or self.rotation_limit_rad <= 0:
            raise ValueError("Action limits must be positive")
        if self.ik_relative_scale <= 0:
            raise ValueError("IK scale must be positive")


class ActionAdapter:
    """Convert canonical physical 7D actions to Isaac IK-relative raw actions.

    Canonical convention:
      - indices 0:3: translation delta [m] in the robot base frame;
      - indices 3:6: rotation-vector delta [rad] in the robot base frame;
      - index 6: gripper, positive=open and negative=close.

    Isaac Lab's action term multiplies the first six raw values by
    ``ik_relative_scale``. Dividing here prevents a second hidden scale.
    """

    def __init__(self, spec: ActionSpec | None = None) -> None:
        self.spec = spec or ActionSpec()

    @staticmethod
    def _vector(action: ArrayLike) -> NDArray[np.float32]:
        vector = np.asarray(action, dtype=np.float32)
        if vector.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError("Action contains NaN or infinity")
        return vector.copy()

    def clip_physical(self, action: ArrayLike) -> NDArray[np.float32]:
        vector = self._vector(action)
        vector[:3] = np.clip(
            vector[:3], -self.spec.translation_limit_m, self.spec.translation_limit_m
        )
        vector[3:6] = np.clip(vector[3:6], -self.spec.rotation_limit_rad, self.spec.rotation_limit_rad)
        vector[6] = 1.0 if vector[6] >= self.spec.gripper_threshold else -1.0
        return vector

    def to_isaac(self, action: ArrayLike) -> NDArray[np.float32]:
        vector = self.clip_physical(action)
        vector[:6] = np.clip(vector[:6] / self.spec.ik_relative_scale, -1.0, 1.0)
        return vector

    def from_isaac(self, action: ArrayLike) -> NDArray[np.float32]:
        vector = self._vector(action)
        vector[:6] *= self.spec.ik_relative_scale
        vector[6] = 1.0 if vector[6] >= 0.0 else -1.0
        return vector

