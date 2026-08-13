"""Validation for deployable pi0.5 observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class DeployableObservation:
    table_rgb: NDArray[np.uint8]
    wrist_rgb: NDArray[np.uint8]
    robot_state: NDArray[np.float32]
    prompt: str


def make_observation(
    table_rgb: ArrayLike, wrist_rgb: ArrayLike, robot_state: ArrayLike, prompt: str
) -> DeployableObservation:
    images = []
    for name, value in (("table_rgb", table_rgb), ("wrist_rgb", wrist_rgb)):
        image = np.asarray(value)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError(f"{name} must be uint8 HWC RGB/RGBA, got {image.dtype} {image.shape}")
        images.append(np.ascontiguousarray(image[..., :3]))
    state = np.asarray(robot_state, dtype=np.float32)
    if state.ndim != 1 or not np.isfinite(state).all():
        raise ValueError("robot_state must be a finite 1D vector")
    if not prompt.strip():
        raise ValueError("prompt must be non-empty")
    return DeployableObservation(images[0], images[1], state.copy(), prompt.strip())

