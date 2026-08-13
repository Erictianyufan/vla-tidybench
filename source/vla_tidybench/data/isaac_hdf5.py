"""Validated conversion from Isaac Lab recordings to deployable VLA samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import h5py


STATE_DIM = 18
ACTION_DIM = 7
DEPLOYABLE_OBSERVATION_KEYS = ("table_cam", "wrist_cam", "joint_pos", "joint_vel")
PRIVILEGED_KEYS = ("object", "cube_positions", "cube_orientations", "states")


@dataclass(frozen=True)
class EpisodeArrays:
    """Arrays allowed to cross the simulator-to-policy data boundary."""

    table_image: NDArray[np.uint8]
    wrist_image: NDArray[np.uint8]
    state: NDArray[np.float32]
    actions: NDArray[np.float32]

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


def sorted_episode_names(data_group: "h5py.Group") -> list[str]:
    """Sort standard Isaac episode names numerically and reject other names."""

    names = list(data_group.keys())
    try:
        return sorted(names, key=lambda name: int(name.removeprefix("demo_")))
    except ValueError as exc:
        raise ValueError("all episode names must use the demo_<integer> convention") from exc


def canonical_actions(raw_actions: NDArray[np.generic]) -> NDArray[np.float32]:
    """Convert recorded Isaac IK-relative actions into canonical physical actions."""

    from vla_tidybench.policy_bridge.action_adapter import ActionAdapter
    from vla_tidybench.policy_bridge.safety_guard import SafetyGuard

    adapter = ActionAdapter()
    guard = SafetyGuard(adapter=adapter)
    actions = adapter.from_isaac_batch(raw_actions)
    guarded = np.stack([guard.apply(action) for action in actions])
    if not np.allclose(actions, guarded, rtol=0.0, atol=1e-6):
        changed = int(np.any(np.abs(actions - guarded) > 1e-6, axis=1).sum())
        raise ValueError(f"{changed}/{len(actions)} expert actions exceed the deployment safety contract")
    return actions


def deployable_state(joint_pos: NDArray[np.generic], joint_vel: NDArray[np.generic]) -> NDArray[np.float32]:
    """Build the 18D deployable state without simulator object truth."""

    position = np.asarray(joint_pos, dtype=np.float32)
    velocity = np.asarray(joint_vel, dtype=np.float32)
    if position.ndim != 2 or position.shape[1] != 9:
        raise ValueError(f"joint_pos must have shape (T, 9), got {position.shape}")
    if velocity.shape != position.shape:
        raise ValueError(f"joint_vel shape {velocity.shape} does not match joint_pos {position.shape}")
    state = np.concatenate((position, velocity), axis=1, dtype=np.float32)
    if state.shape[1] != STATE_DIM or not np.isfinite(state).all():
        raise ValueError("state is malformed or contains NaN/infinity")
    return state


def load_episode(path: Path, episode_name: str) -> EpisodeArrays:
    """Load and validate one successful Isaac episode.

    The loader intentionally names every permitted source field. Privileged
    object state and serialized simulator state are never returned.
    """

    import h5py

    with h5py.File(path, "r") as dataset:
        if int(dataset.attrs.get("format_version", -1)) != 1:
            raise ValueError(f"unsupported or missing format_version in {path}")
        episode = dataset["data"][episode_name]
        if not bool(episode.attrs.get("success", False)):
            raise ValueError(f"refusing unsuccessful episode {path}::{episode_name}")
        observations = episode["obs"]
        missing = [key for key in DEPLOYABLE_OBSERVATION_KEYS if key not in observations]
        if missing:
            raise ValueError(f"missing deployable observations in {episode_name}: {missing}")

        table = np.asarray(observations["table_cam"], dtype=np.uint8)
        wrist = np.asarray(observations["wrist_cam"], dtype=np.uint8)
        state = deployable_state(observations["joint_pos"], observations["joint_vel"])
        actions = canonical_actions(episode["actions"])

    expected_image_shape = (actions.shape[0], 200, 200, 3)
    if table.shape != expected_image_shape or wrist.shape != expected_image_shape:
        raise ValueError(
            f"camera arrays must both have shape {expected_image_shape}, got {table.shape} and {wrist.shape}"
        )
    if state.shape[0] != actions.shape[0]:
        raise ValueError("state/action sequence lengths do not match")
    return EpisodeArrays(table, wrist, state, actions)
