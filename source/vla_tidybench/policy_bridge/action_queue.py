"""Episode-aware receding-horizon action queue."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ActionChunk:
    episode_id: str
    observation_step: int
    created_monotonic_s: float
    actions: NDArray[np.float32]

    @classmethod
    def create(cls, episode_id: str, observation_step: int, actions: ArrayLike) -> "ActionChunk":
        array = np.asarray(actions, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 7 or array.shape[0] < 1:
            raise ValueError(f"Expected action chunk [H, 7], got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("Action chunk contains NaN or infinity")
        return cls(episode_id, observation_step, time.monotonic(), array.copy())


class ActionQueue:
    """Queue that rejects replies from old episodes, steps or wall-clock time."""

    def __init__(self, max_age_s: float = 0.5) -> None:
        if max_age_s <= 0:
            raise ValueError("max_age_s must be positive")
        self.max_age_s = max_age_s
        self._actions: deque[NDArray[np.float32]] = deque()
        self._episode_id: str | None = None
        self._observation_step = -1

    def reset(self, episode_id: str) -> None:
        self._actions.clear()
        self._episode_id = episode_id
        self._observation_step = -1

    def accept(self, chunk: ActionChunk, current_step: int, now_s: float | None = None) -> bool:
        now_s = time.monotonic() if now_s is None else now_s
        valid = (
            chunk.episode_id == self._episode_id
            and chunk.observation_step >= self._observation_step
            and chunk.observation_step <= current_step
            and now_s - chunk.created_monotonic_s <= self.max_age_s
        )
        if not valid:
            return False
        self._actions = deque(action.copy() for action in chunk.actions)
        self._observation_step = chunk.observation_step
        return True

    def pop(self) -> NDArray[np.float32] | None:
        return self._actions.popleft() if self._actions else None

    def __len__(self) -> int:
        return len(self._actions)

