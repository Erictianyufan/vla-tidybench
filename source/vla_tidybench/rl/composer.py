"""Safety-bounded composition of a frozen VLA action and a residual action."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualComposer:
    translation_limit_m: float = 0.015
    rotation_limit_rad: float = 0.08
    beta: float = 0.25

    def compose(self, nominal, residual) -> np.ndarray:
        base = np.asarray(nominal, dtype=np.float32)
        delta = np.asarray(residual, dtype=np.float32)
        if base.shape != (7,) or delta.shape != (6,):
            raise ValueError("nominal must be 7D and residual must be 6D")
        limits = np.asarray([self.translation_limit_m] * 3 + [self.rotation_limit_rad] * 3, dtype=np.float32)
        output = base.copy()
        output[:6] += self.beta * np.clip(delta, -limits, limits)
        # The RL specialist cannot cross the discrete gripper threshold.
        output[6] = base[6]
        return output
