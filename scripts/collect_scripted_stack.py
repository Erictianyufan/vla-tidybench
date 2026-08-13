"""Collect successful Franka cube-stack demonstrations with a privileged scripted teacher.

The teacher reads simulator object poses and the forward-kinematics end-effector
pose, then emits the environment's canonical 7D IK-relative action. Joint-space
commands are still produced by the task's native damped-least-squares IK term.
Privileged poses are used by the teacher only; the recorded policy observation
contract remains unchanged.
"""

from __future__ import annotations

import argparse
import enum
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0")
parser.add_argument("--dataset_file", type=Path, required=True)
parser.add_argument("--num_demos", type=int, default=1)
parser.add_argument("--max_attempts", type=int, default=12)
parser.add_argument("--max_steps", type=int, default=520)
parser.add_argument("--seed", type=int, default=41)
parser.add_argument("--position_gain", type=float, default=1.7)
parser.add_argument("--max_translation_action", type=float, default=0.12)
parser.add_argument("--overwrite", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_demos < 1:
    parser.error("--num_demos must be positive")
if args_cli.max_attempts < args_cli.num_demos:
    parser.error("--max_attempts must be at least --num_demos")
if args_cli.dataset_file.exists() and not args_cli.overwrite:
    parser.error(f"dataset already exists: {args_cli.dataset_file}; pass --overwrite to replace it")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import isaaclab_tasks  # noqa: E402, F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

TEACHER_VERSION = "truth_fsm_dls_v2"
LANGUAGE_INSTRUCTION = "stack the red block on the blue block, then stack the green block on top"


class Phase(enum.Enum):
    START = enum.auto()
    ABOVE_PICK = enum.auto()
    DESCEND_PICK = enum.auto()
    CLOSE = enum.auto()
    LIFT = enum.auto()
    ABOVE_PLACE = enum.auto()
    DESCEND_PLACE = enum.auto()
    OPEN = enum.auto()
    RETREAT = enum.auto()
    SETTLE = enum.auto()
    DONE = enum.auto()
    FAILED = enum.auto()


class StackTeacher:
    """Finite-state task-space teacher for red-on-blue then green-on-red."""

    def __init__(
        self,
        env,
        *,
        position_gain: float,
        max_translation_action: float,
        seed: int,
    ) -> None:
        self.env = env
        self.device = env.device
        self.position_gain = position_gain
        self.max_translation_action = max_translation_action
        self.rng = np.random.default_rng(seed)
        self.pick_order = ("cube_2", "cube_3")
        self.place_order = ("cube_1", "cube_2")
        self.reset_episode()

    def reset_episode(self) -> None:
        self.phase = Phase.START
        self.phase_steps = 0
        self.stable_steps = 0
        self.object_index = 0
        self.pick_xy_jitter = self.rng.uniform(-0.0025, 0.0025, size=2)
        self.place_xy_jitter = self.rng.uniform(-0.003, 0.003, size=2)
        self.lift_target = np.zeros(3, dtype=np.float32)
        self.last_reason = ""

    @property
    def source_name(self) -> str:
        return self.pick_order[self.object_index]

    @property
    def destination_name(self) -> str:
        return self.place_order[self.object_index]

    def _eef_position(self) -> torch.Tensor:
        return self.env.scene["ee_frame"].data.target_pos_w.torch[:, 0, :]

    def _object_position(self, name: str) -> torch.Tensor:
        return self.env.scene[name].data.root_pos_w.torch

    def _set_phase(self, phase: Phase) -> None:
        if phase is not self.phase:
            print(f"  phase {self.phase.name} -> {phase.name}", flush=True)
        self.phase = phase
        self.phase_steps = 0
        self.stable_steps = 0

    def _position_action(self, target: torch.Tensor, gripper: float, threshold: float = 0.010) -> torch.Tensor:
        eef_pos = self._eef_position()
        error = target - eef_pos
        distance = torch.linalg.vector_norm(error, dim=1)
        self.stable_steps = self.stable_steps + 1 if bool((distance < threshold)[0]) else 0

        # Environment action scale is 0.5. The proportional command below is
        # intentionally conservative and is clipped before native DLS IK.
        translation = torch.clamp(
            error * self.position_gain,
            min=-self.max_translation_action,
            max=self.max_translation_action,
        )
        action = torch.zeros((self.env.num_envs, 7), dtype=torch.float32, device=self.device)
        action[:, :3] = translation
        action[:, 6] = gripper
        return action

    def _hold_action(self, gripper: float) -> torch.Tensor:
        action = torch.zeros((self.env.num_envs, 7), dtype=torch.float32, device=self.device)
        action[:, 6] = gripper
        return action

    def _stacked(self, upper_name: str, lower_name: str) -> bool:
        upper = self._object_position(upper_name)
        lower = self._object_position(lower_name)
        delta = upper - lower
        xy_ok = torch.linalg.vector_norm(delta[:, :2], dim=1) < 0.045
        height_ok = torch.abs(delta[:, 2] - 0.0468) < 0.012
        return bool(torch.logical_and(xy_ok, height_ok)[0])

    def _advance_object(self) -> None:
        if self.object_index == 0:
            if not self._stacked("cube_2", "cube_1"):
                self.last_reason = "red block did not settle on blue block"
                self._set_phase(Phase.FAILED)
                return
            self.object_index = 1
            self.pick_xy_jitter = self.rng.uniform(-0.0025, 0.0025, size=2)
            self.place_xy_jitter = self.rng.uniform(-0.003, 0.003, size=2)
            self._set_phase(Phase.ABOVE_PICK)
        else:
            self._set_phase(Phase.DONE)

    def action(self) -> torch.Tensor:
        self.phase_steps += 1
        source_pos = self._object_position(self.source_name).clone()
        destination_pos = self._object_position(self.destination_name).clone()
        pick_jitter = torch.tensor(self.pick_xy_jitter, dtype=torch.float32, device=self.device)
        place_jitter = torch.tensor(self.place_xy_jitter, dtype=torch.float32, device=self.device)

        if self.phase is Phase.START:
            if self.phase_steps >= 8:
                self._set_phase(Phase.ABOVE_PICK)
            return self._hold_action(1.0)

        if self.phase is Phase.ABOVE_PICK:
            target = source_pos.clone()
            target[:, :2] += pick_jitter
            target[:, 2] = 0.145
            action = self._position_action(target, 1.0, threshold=0.012)
            if self.stable_steps >= 4:
                self._set_phase(Phase.DESCEND_PICK)
            return action

        if self.phase is Phase.DESCEND_PICK:
            target = source_pos.clone()
            target[:, :2] += pick_jitter
            target[:, 2] += 0.013
            action = self._position_action(target, 1.0, threshold=0.007)
            if self.stable_steps >= 4:
                self._set_phase(Phase.CLOSE)
            return action

        if self.phase is Phase.CLOSE:
            if self.phase_steps >= 14:
                current = self._eef_position()[0].detach().cpu().numpy()
                self.lift_target = np.array([current[0], current[1], 0.155], dtype=np.float32)
                self._set_phase(Phase.LIFT)
            return self._hold_action(-1.0)

        if self.phase is Phase.LIFT:
            target = torch.tensor(self.lift_target, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self._position_action(target, -1.0, threshold=0.012)
            if self.stable_steps >= 4:
                carried_height = float(source_pos[0, 2])
                if carried_height < 0.065:
                    self.last_reason = f"failed grasp of {self.source_name}: z={carried_height:.3f}"
                    self._set_phase(Phase.FAILED)
                else:
                    self._set_phase(Phase.ABOVE_PLACE)
            return action

        if self.phase is Phase.ABOVE_PLACE:
            target = destination_pos.clone()
            target[:, :2] += place_jitter
            target[:, 2] = torch.clamp(destination_pos[:, 2] + 0.135, min=0.145)
            action = self._position_action(target, -1.0, threshold=0.012)
            if self.stable_steps >= 4:
                self._set_phase(Phase.DESCEND_PLACE)
            return action

        if self.phase is Phase.DESCEND_PLACE:
            target = destination_pos.clone()
            target[:, :2] += place_jitter
            target[:, 2] += 0.059
            action = self._position_action(target, -1.0, threshold=0.008)
            if self.stable_steps >= 5:
                self._set_phase(Phase.OPEN)
            return action

        if self.phase is Phase.OPEN:
            if self.phase_steps >= 16:
                self._set_phase(Phase.RETREAT)
            return self._hold_action(1.0)

        if self.phase is Phase.RETREAT:
            target = destination_pos.clone()
            target[:, :2] += place_jitter
            target[:, 2] = torch.clamp(destination_pos[:, 2] + 0.135, min=0.145)
            action = self._position_action(target, 1.0, threshold=0.012)
            if self.stable_steps >= 4:
                self._set_phase(Phase.SETTLE)
            return action

        if self.phase is Phase.SETTLE:
            # Keep recording a stability margin after release. This makes the
            # contact-rich placement more robust to non-deterministic replay.
            if self.phase_steps >= 28:
                self._advance_object()
            return self._hold_action(1.0)

        return self._hold_action(1.0)


def configure_environment():
    output_path = args_cli.dataset_file.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and args_cli.overwrite:
        output_path.unlink()

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task
    env_cfg.observations.policy.concatenate_terms = False

    # Manual episode boundaries prevent the full sim reset that conflicts with
    # Direct GPU API. Drop/success/time-out are checked by this collector.
    for name in ("time_out", "success", "cube_1_dropping", "cube_2_dropping", "cube_3_dropping"):
        if hasattr(env_cfg.terminations, name):
            setattr(env_cfg.terminations, name, None)

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = str(output_path.parent)
    env_cfg.recorders.dataset_filename = output_path.stem
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.export_in_close = False
    return env_cfg, output_path


def finish_episode(env, succeeded: bool) -> None:
    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    success = torch.tensor([succeeded], dtype=torch.bool, device=env.device)
    env.recorder_manager.set_success_to_episodes([0], success)
    env.recorder_manager.export_episodes([0])


def add_dataset_metadata(path: Path, attempts: int, successes: int) -> None:
    with h5py.File(path, "r+") as dataset:
        data = dataset["data"]
        data.attrs["collector"] = TEACHER_VERSION
        data.attrs["language_instruction"] = LANGUAGE_INSTRUCTION
        data.attrs["seed"] = args_cli.seed
        data.attrs["attempts"] = attempts
        data.attrs["successful_episodes"] = successes
        for demo in data.values():
            demo.attrs["source"] = "scripted_truth_teacher"
            demo.attrs["teacher_version"] = TEACHER_VERSION
            demo.attrs["language_instruction"] = LANGUAGE_INSTRUCTION


def main() -> int:
    env_cfg, output_path = configure_environment()
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    teacher = StackTeacher(
        env,
        position_gain=args_cli.position_gain,
        max_translation_action=args_cli.max_translation_action,
        seed=args_cli.seed,
    )

    successes = 0
    attempts = 0
    run_started = time.monotonic()
    try:
        env.reset(seed=args_cli.seed)
        while successes < args_cli.num_demos and attempts < args_cli.max_attempts and simulation_app.is_running():
            attempts += 1
            teacher.reset_episode()
            episode_started = time.monotonic()
            print(f"attempt {attempts}/{args_cli.max_attempts}", flush=True)

            succeeded = False
            with torch.inference_mode():
                for _step in range(args_cli.max_steps):
                    action = teacher.action()
                    env.step(action)

                    if teacher.phase is Phase.DONE:
                        succeeded = teacher._stacked("cube_2", "cube_1") and teacher._stacked("cube_3", "cube_2")
                        break
                    if teacher.phase is Phase.FAILED:
                        break
                    if any(float(env.scene[name].data.root_pos_w.torch[0, 2]) < -0.03 for name in teacher.pick_order):
                        teacher.last_reason = "a block fell below the table"
                        teacher._set_phase(Phase.FAILED)
                        break
                else:
                    teacher.last_reason = f"episode exceeded {args_cli.max_steps} steps"

            finish_episode(env, succeeded)
            elapsed = time.monotonic() - episode_started
            if succeeded:
                successes += 1
                print(f"SUCCESS {successes}/{args_cli.num_demos}: {_step + 1} steps, {elapsed:.1f}s", flush=True)
            else:
                print(f"FAILED: {teacher.last_reason or teacher.phase.name}, {_step + 1} steps", flush=True)

            if successes < args_cli.num_demos and attempts < args_cli.max_attempts:
                env.reset(seed=args_cli.seed + attempts)

        env.recorder_manager.close()
        if output_path.exists():
            add_dataset_metadata(output_path, attempts, successes)
    finally:
        env.close()

    summary = {
        "teacher": TEACHER_VERSION,
        "dataset": str(output_path),
        "attempts": attempts,
        "successes": successes,
        "target_successes": args_cli.num_demos,
        "elapsed_s": round(time.monotonic() - run_started, 2),
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if successes == args_cli.num_demos else 1


if __name__ == "__main__":
    raise SystemExit(main())
