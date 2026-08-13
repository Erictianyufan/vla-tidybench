"""Collect project drawer demonstrations with truth-guided task-space teachers.

The teacher reads simulator truth only to form task-space waypoints.  The
environment's production DifferentialInverseKinematicsAction term performs
the damped-least-squares joint solve.  Recorded policy observations contain
only two RGB images and Franka proprioception.
"""

from __future__ import annotations

import argparse
import enum
import faulthandler
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--skill", choices=("open", "pick", "place", "close", "full"), default="open")
parser.add_argument("--dataset_file", type=Path, required=True)
parser.add_argument("--num_demos", type=int, default=1)
parser.add_argument("--max_attempts", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=720)
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--overwrite", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.dataset_file.exists() and not args_cli.overwrite:
    parser.error(f"dataset exists: {args_cli.dataset_file}; pass --overwrite")
if args_cli.num_demos < 1 or args_cli.max_attempts < args_cli.num_demos:
    parser.error("invalid demo/attempt count")

faulthandler.enable()
faulthandler.dump_traceback_later(45, repeat=False)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _startup_mark(name: str) -> None:
    print(f"[TIDYBENCH_STARTUP] {name}", file=sys.__stderr__, flush=True)


_startup_mark("app_ready")

import gymnasium as gym  # noqa: E402
_startup_mark("gymnasium")
import h5py  # noqa: E402
_startup_mark("h5py")
import numpy as np  # noqa: E402
_startup_mark("numpy")
import torch  # noqa: E402
_startup_mark("torch")
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
_startup_mark("recorder_cfg")
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab.utils.math import compute_pose_error  # noqa: E402
_startup_mark("isaac_utils")

from vla_tidybench.isaac import TidyBenchDrawerEnvCfg  # noqa: E402
_startup_mark("drawer_cfg")


TEACHER_VERSION = "tidybench_truth_fsm_dls_v1"
PROMPTS = {
    "open": "open the top drawer",
    "pick": "pick up the red object",
    "place": "put the red object into the top drawer",
    "close": "close the top drawer",
    "full": "put the red object into the top drawer and close it",
}


class Phase(enum.Enum):
    SETTLE = enum.auto()
    HANDLE_APPROACH = enum.auto()
    HANDLE_ALIGN = enum.auto()
    HANDLE_CLOSE = enum.auto()
    DRAWER_PULL = enum.auto()
    HANDLE_RELEASE = enum.auto()
    HANDLE_RETREAT = enum.auto()
    PICK_ABOVE = enum.auto()
    PICK_DESCEND = enum.auto()
    PICK_CLOSE = enum.auto()
    PICK_LIFT = enum.auto()
    PLACE_CLEARANCE = enum.auto()
    PLACE_ABOVE = enum.auto()
    PLACE_DESCEND = enum.auto()
    PLACE_INSERT = enum.auto()
    PLACE_RELEASE = enum.auto()
    PLACE_RETREAT = enum.auto()
    CLOSE_APPROACH = enum.auto()
    CLOSE_ALIGN = enum.auto()
    CLOSE_GRASP = enum.auto()
    DRAWER_PUSH = enum.auto()
    CLOSE_RELEASE = enum.auto()
    FINAL_RETREAT = enum.auto()
    DONE = enum.auto()
    FAILED = enum.auto()


class DrawerTeacher:
    """Finite-state teacher for OPEN/PICK/PLACE/CLOSE and their composition."""

    handle_quat = (0.5, 0.5, 0.5, 0.5)  # xyzw, aligned with the horizontal handle
    down_quat = (1.0, 0.0, 0.0, 0.0)  # xyzw, gripper pointing down

    def __init__(self, env, skill: str, seed: int) -> None:
        self.env = env
        self.skill = skill
        self.device = env.device
        self.rng = np.random.default_rng(seed)
        self.drawer_joint_idx = env.scene["cabinet"].find_joints("drawer_top_joint")[0][0]
        self.reset_episode()

    def reset_episode(self) -> None:
        self.phase = Phase.SETTLE
        self.phase_steps = 0
        self.stable_steps = 0
        self.last_reason = ""
        self.fixed_pull_target = None
        self.fixed_target = None
        self.fixed_quat = None
        self.place_handle_anchor = None
        self.pick_jitter = self.rng.uniform(-0.003, 0.003, size=2)
        self.place_jitter = self.rng.uniform(-0.006, 0.006, size=2)

    def _set_phase(self, phase: Phase) -> None:
        if phase is not self.phase:
            print(f"  {self.phase.name} -> {phase.name}", flush=True)
        self.phase = phase
        self.phase_steps = 0
        self.stable_steps = 0
        if phase in (
            Phase.PLACE_CLEARANCE,
            Phase.PLACE_ABOVE,
            Phase.PLACE_DESCEND,
            Phase.PLACE_INSERT,
            Phase.PLACE_RETREAT,
        ):
            self.fixed_target = None

    def _eef(self) -> tuple[torch.Tensor, torch.Tensor]:
        frame = self.env.scene["ee_frame"].data
        return frame.target_pos_w.torch[:, 0, :], frame.target_quat_w.torch[:, 0, :]

    def _handle(self) -> torch.Tensor:
        return self.env.scene["cabinet_frame"].data.target_pos_w.torch[:, 0, :]

    def _object(self) -> torch.Tensor:
        return self.env.scene["target_object"].data.root_pos_w.torch

    def _drawer_pos(self) -> float:
        return float(self.env.scene["cabinet"].data.joint_pos.torch[0, self.drawer_joint_idx])

    def _pose_action(
        self,
        target_pos: torch.Tensor,
        target_quat_xyzw: tuple[float, float, float, float],
        gripper: float,
        *,
        pos_threshold: float = 0.012,
        rot_threshold: float = 0.15,
    ) -> torch.Tensor:
        current_pos, current_quat = self._eef()
        target_quat = torch.tensor(target_quat_xyzw, dtype=torch.float32, device=self.device).unsqueeze(0)
        pos_error, rot_error = compute_pose_error(current_pos, current_quat, target_pos, target_quat)
        pos_norm = torch.linalg.vector_norm(pos_error, dim=1)
        rot_norm = torch.linalg.vector_norm(rot_error, dim=1)
        reached = bool(((pos_norm < pos_threshold) & (rot_norm < rot_threshold))[0])
        self.stable_steps = self.stable_steps + 1 if reached else 0

        action = torch.zeros((1, 7), dtype=torch.float32, device=self.device)
        action[:, :3] = torch.clamp(pos_error * 1.8, -0.12, 0.12)
        action[:, 3:6] = torch.clamp(rot_error * 1.4, -0.55, 0.55)
        action[:, 6] = gripper
        return action

    def _hold(self, gripper: float) -> torch.Tensor:
        action = torch.zeros((1, 7), dtype=torch.float32, device=self.device)
        action[:, 6] = gripper
        return action

    def _start_after_settle(self) -> Phase:
        if self.skill in ("open", "full"):
            return Phase.HANDLE_APPROACH
        if self.skill == "pick":
            return Phase.PICK_ABOVE
        if self.skill == "place":
            return Phase.PICK_ABOVE
        return Phase.CLOSE_APPROACH

    def _after_open(self) -> Phase:
        return Phase.DONE if self.skill == "open" else Phase.PICK_ABOVE

    def _after_pick(self) -> Phase:
        return Phase.DONE if self.skill == "pick" else Phase.PLACE_CLEARANCE

    def _after_place(self) -> Phase:
        return Phase.DONE if self.skill == "place" else Phase.CLOSE_APPROACH

    def action(self) -> torch.Tensor:
        self.phase_steps += 1
        handle = self._handle().clone()
        obj = self._object().clone()

        if self.phase is Phase.SETTLE:
            if self.phase_steps >= 10:
                self._set_phase(self._start_after_settle())
            return self._hold(1.0)

        if self.phase is Phase.HANDLE_APPROACH:
            target = handle.clone()
            target[:, 0] -= 0.10
            action = self._pose_action(target, self.handle_quat, 1.0, pos_threshold=0.015)
            if self.stable_steps >= 4:
                self._set_phase(Phase.HANDLE_ALIGN)
            return action

        if self.phase is Phase.HANDLE_ALIGN:
            target = handle.clone()
            target[:, 0] -= 0.008
            action = self._pose_action(target, self.handle_quat, 1.0, pos_threshold=0.008, rot_threshold=0.10)
            if self.stable_steps >= 4:
                self._set_phase(Phase.HANDLE_CLOSE)
            return action

        if self.phase is Phase.HANDLE_CLOSE:
            if self.phase_steps >= 14:
                self.fixed_pull_target = handle.clone()
                self.fixed_pull_target[:, 0] -= 0.39
                self._set_phase(Phase.DRAWER_PULL)
            return self._hold(-1.0)

        if self.phase is Phase.DRAWER_PULL:
            action = self._pose_action(self.fixed_pull_target, self.handle_quat, -1.0, pos_threshold=0.025)
            if self._drawer_pos() >= 0.30:
                self._set_phase(Phase.HANDLE_RELEASE)
            elif self.phase_steps > 150:
                self.last_reason = f"drawer stuck at {self._drawer_pos():.3f} m"
                self._set_phase(Phase.FAILED)
            return action

        if self.phase is Phase.HANDLE_RELEASE:
            if self.phase_steps >= 12:
                self._set_phase(Phase.HANDLE_RETREAT)
            return self._hold(1.0)

        if self.phase is Phase.HANDLE_RETREAT:
            target = handle.clone()
            target[:, 0] -= 0.06
            action = self._pose_action(target, self.handle_quat, 1.0, pos_threshold=0.018)
            if self.stable_steps >= 3:
                self._set_phase(self._after_open())
            return action

        if self.phase is Phase.PICK_ABOVE:
            target = obj.clone()
            target[:, :2] += torch.tensor(self.pick_jitter, dtype=torch.float32, device=self.device)
            target[:, 2] += 0.14
            action = self._pose_action(target, self.down_quat, 1.0, pos_threshold=0.014)
            if self.stable_steps >= 4:
                self._set_phase(Phase.PICK_DESCEND)
            return action

        if self.phase is Phase.PICK_DESCEND:
            target = obj.clone()
            target[:, :2] += torch.tensor(self.pick_jitter, dtype=torch.float32, device=self.device)
            target[:, 2] += 0.012
            # Contact and the table collision margin can leave the tool about
            # 15-20 mm above this geometric target.  The gripper still spans
            # the 40 mm object, so use a physically meaningful gate instead
            # of waiting forever for an unreachable 8 mm pose residual.
            action = self._pose_action(target, self.down_quat, 1.0, pos_threshold=0.022)
            if self.stable_steps >= 4:
                self._set_phase(Phase.PICK_CLOSE)
            return action

        if self.phase is Phase.PICK_CLOSE:
            if self.phase_steps >= 14:
                self._set_phase(Phase.PICK_LIFT)
            return self._hold(-1.0)

        if self.phase is Phase.PICK_LIFT:
            # Use an object-relative target; recomputing from the current EEF
            # each frame would create a moving target and never converge.
            target = obj.clone()
            target[:, :2] += torch.tensor(self.pick_jitter, dtype=torch.float32, device=self.device)
            target[:, 2] = 0.30
            action = self._pose_action(target, self.down_quat, -1.0, pos_threshold=0.025)
            if float(obj[0, 2]) >= 0.18:
                if float(obj[0, 2]) < 0.12:
                    self.last_reason = f"grasp failed, object z={float(obj[0, 2]):.3f}"
                    self._set_phase(Phase.FAILED)
                else:
                    self._set_phase(self._after_pick())
            return action

        if self.phase is Phase.PLACE_CLEARANCE:
            if self.fixed_target is None:
                self.fixed_target = self._eef()[0].clone()
                self.fixed_target[:, 2] = 0.82
            action = self._pose_action(self.fixed_target, self.down_quat, -1.0, pos_threshold=0.055)
            if self.stable_steps >= 4:
                self._set_phase(Phase.PLACE_ABOVE)
            return action

        if self.phase is Phase.PLACE_ABOVE:
            if self.fixed_target is None:
                self.place_handle_anchor = handle.clone()
                self.fixed_target = self.place_handle_anchor.clone()
                self.fixed_target[:, 0] += 0.04
                self.fixed_target[:, 1] = -0.10 + torch.tensor(
                    self.place_jitter[1], dtype=torch.float32, device=self.device
                )
                self.fixed_target[:, 2] = 0.86
            action = self._pose_action(self.fixed_target, self.down_quat, -1.0, pos_threshold=0.080)
            if self.stable_steps >= 4:
                self._set_phase(Phase.PLACE_INSERT)
            return action

        if self.phase is Phase.PLACE_INSERT:
            if self.fixed_target is None:
                self.fixed_target = self.place_handle_anchor.clone()
                # Cross the front wall at clearance height before descending.
                self.fixed_target[:, 0] += 0.10
                self.fixed_target[:, 1] = -0.10 + torch.tensor(
                    self.place_jitter[1], dtype=torch.float32, device=self.device
                )
                self.fixed_target[:, 2] = 0.86
            action = self._pose_action(self.fixed_target, self.down_quat, -1.0, pos_threshold=0.105)
            if self.stable_steps >= 4:
                self._set_phase(Phase.PLACE_DESCEND)
            return action

        if self.phase is Phase.PLACE_DESCEND:
            if self.fixed_target is None:
                self.fixed_target = self.place_handle_anchor.clone()
                self.fixed_target[:, 0] += 0.10
                self.fixed_target[:, 1] = -0.10 + torch.tensor(
                    self.place_jitter[1], dtype=torch.float32, device=self.device
                )
                self.fixed_target[:, 2] = 0.72
            action = self._pose_action(self.fixed_target, self.down_quat, -1.0, pos_threshold=0.090)
            if self.stable_steps >= 4:
                self._set_phase(Phase.PLACE_RELEASE)
            return action

        if self.phase is Phase.PLACE_RELEASE:
            if self.phase_steps >= 16:
                self._set_phase(Phase.PLACE_RETREAT)
            return self._hold(1.0)

        if self.phase is Phase.PLACE_RETREAT:
            # Decide from settled object truth before spending the remainder
            # of a failed episode on an unreachable cosmetic retreat.
            if self.phase_steps >= 18:
                if not self._object_in_drawer():
                    self.last_reason = f"object not in drawer: {obj[0].detach().cpu().tolist()}"
                    self._set_phase(Phase.FAILED)
                    return self._hold(1.0)
                self._set_phase(self._after_place())
                return self._hold(1.0)
            if self.fixed_target is None:
                self.fixed_target = self._eef()[0].clone()
                self.fixed_target[:, 2] += 0.08
            target = self.fixed_target
            action = self._pose_action(target, self.down_quat, 1.0, pos_threshold=0.030)
            if self.stable_steps >= 3:
                self._set_phase(self._after_place())
            return action

        if self.phase is Phase.CLOSE_APPROACH:
            target = handle.clone()
            target[:, 0] -= 0.10
            action = self._pose_action(target, self.handle_quat, 1.0, pos_threshold=0.018)
            if self._drawer_pos() <= 0.04 and self.phase_steps > 20:
                self._set_phase(Phase.CLOSE_RELEASE)
                return action
            if self.stable_steps >= 3:
                self._set_phase(Phase.CLOSE_ALIGN)
            return action

        if self.phase is Phase.CLOSE_ALIGN:
            target = handle.clone()
            target[:, 0] -= 0.008
            action = self._pose_action(target, self.handle_quat, 1.0, pos_threshold=0.010)
            if self.stable_steps >= 3:
                self._set_phase(Phase.CLOSE_GRASP)
            return action

        if self.phase is Phase.CLOSE_GRASP:
            if self.phase_steps >= 12:
                self.fixed_pull_target = handle.clone()
                self.fixed_pull_target[:, 0] += max(self._drawer_pos() - 0.015, 0.0)
                self._set_phase(Phase.DRAWER_PUSH)
            return self._hold(-1.0)

        if self.phase is Phase.DRAWER_PUSH:
            action = self._pose_action(self.fixed_pull_target, self.handle_quat, -1.0, pos_threshold=0.025)
            if self._drawer_pos() <= 0.04:
                self._set_phase(Phase.CLOSE_RELEASE)
            elif self.phase_steps > 160:
                self.last_reason = f"drawer would not close: {self._drawer_pos():.3f} m"
                self._set_phase(Phase.FAILED)
            return action

        if self.phase is Phase.CLOSE_RELEASE:
            if self.phase_steps >= 12:
                self._set_phase(Phase.FINAL_RETREAT)
            return self._hold(1.0)

        if self.phase is Phase.FINAL_RETREAT:
            target = handle.clone()
            target[:, 0] -= 0.12
            action = self._pose_action(target, self.handle_quat, 1.0, pos_threshold=0.020)
            if self.stable_steps >= 3:
                self._set_phase(Phase.DONE)
            return action

        return self._hold(1.0)

    def _object_in_drawer(self) -> bool:
        obj = self._object()[0]
        handle_x = self._handle()[0, 0]
        # The 4.5 cm cuboid center must be at least one half-width plus a
        # small margin behind the handle/front-plane proxy.
        return bool(obj[2] > 0.68 and obj[2] < 0.86 and obj[0] > handle_x + 0.023 and abs(obj[1]) < 0.26)

    def success(self) -> bool:
        if self.skill == "open":
            return self._drawer_pos() >= 0.30
        if self.skill == "pick":
            return float(self._object()[0, 2]) >= 0.12
        if self.skill == "place":
            return self._object_in_drawer()
        if self.skill == "close":
            return self._drawer_pos() <= 0.04
        return self._drawer_pos() <= 0.04 and self._object_in_drawer()


def make_cfg() -> TidyBenchDrawerEnvCfg:
    output = args_cli.dataset_file.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and args_cli.overwrite:
        output.unlink()
    cfg = TidyBenchDrawerEnvCfg()
    cfg.sim.device = args_cli.device
    if args_cli.skill in ("place", "close"):
        # This is part of the environment's reset state so RecorderManager
        # serializes the exact prerequisite state and physical replay can
        # reproduce the episode without an out-of-band teleport.
        cfg.scene.cabinet.init_state.joint_pos["drawer_top_joint"] = 0.36
    if args_cli.skill == "place":
        cfg.scene.cabinet.actuators["drawers"].stiffness = 10000.0
        cfg.scene.cabinet.actuators["drawers"].damping = 500.0
    cfg.recorders = ActionStateRecorderManagerCfg()
    cfg.recorders.dataset_export_dir_path = str(output.parent)
    cfg.recorders.dataset_filename = output.stem
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    cfg.recorders.export_in_close = False
    return cfg


def finish_episode(env, succeeded: bool) -> None:
    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    success = torch.tensor([succeeded], dtype=torch.bool, device=env.device)
    env.recorder_manager.set_success_to_episodes([0], success)
    env.recorder_manager.export_episodes([0])


def initialize_skill_state(env, skill: str) -> None:
    """Reserved for future prerequisite state validation.

    Skill prerequisites are configured before environment creation so the
    RecorderManager initial-state term remains truthful and replayable.
    """

    del env, skill


def add_metadata(path: Path, attempts: int, successes: int) -> None:
    with h5py.File(path, "r+") as dataset:
        data = dataset["data"]
        data.attrs["collector"] = TEACHER_VERSION
        data.attrs["skill"] = args_cli.skill
        data.attrs["language_instruction"] = PROMPTS[args_cli.skill]
        data.attrs["attempts"] = attempts
        data.attrs["successful_episodes"] = successes
        data.attrs["seed"] = args_cli.seed
        for demo in data.values():
            demo.attrs["source"] = "scripted_truth_teacher"
            demo.attrs["teacher_version"] = TEACHER_VERSION
            demo.attrs["skill"] = args_cli.skill
            demo.attrs["language_instruction"] = PROMPTS[args_cli.skill]


def main() -> int:
    faulthandler.cancel_dump_traceback_later()
    output = args_cli.dataset_file.resolve()
    env = gym.make("Isaac-Open-Drawer-Franka-IK-Rel-v0", cfg=make_cfg()).unwrapped
    teacher = DrawerTeacher(env, args_cli.skill, args_cli.seed)
    attempts = 0
    successes = 0
    started = time.monotonic()
    try:
        while successes < args_cli.num_demos and attempts < args_cli.max_attempts and simulation_app.is_running():
            attempts += 1
            env.reset(seed=args_cli.seed + attempts - 1)
            initialize_skill_state(env, args_cli.skill)
            teacher.reset_episode()
            print(f"attempt {attempts}/{args_cli.max_attempts} skill={args_cli.skill}", flush=True)
            succeeded = False
            with torch.inference_mode():
                for step in range(args_cli.max_steps):
                    env.step(teacher.action())
                    if step % 40 == 0:
                        eef_pos = teacher._eef()[0][0].detach().cpu().tolist()
                        fixed = None if teacher.fixed_target is None else teacher.fixed_target[0].detach().cpu().tolist()
                        print(
                            f"  step={step} phase={teacher.phase.name} drawer={teacher._drawer_pos():.3f} "
                            f"object_z={float(teacher._object()[0, 2]):.3f} eef={eef_pos} target={fixed}",
                            flush=True,
                        )
                    if teacher.phase is Phase.DONE:
                        succeeded = teacher.success()
                        break
                    if teacher.phase is Phase.FAILED:
                        break
                else:
                    teacher.last_reason = f"exceeded {args_cli.max_steps} steps"
            finish_episode(env, succeeded)
            if succeeded:
                successes += 1
                print(f"SUCCESS {successes}/{args_cli.num_demos} in {step + 1} steps", flush=True)
            else:
                print(f"FAILED: {teacher.last_reason or teacher.phase.name}", flush=True)
        env.recorder_manager.close()
        if output.exists():
            add_metadata(output, attempts, successes)
        summary = {
            "teacher": TEACHER_VERSION,
            "skill": args_cli.skill,
            "dataset": str(output),
            "attempts": attempts,
            "successes": successes,
            "elapsed_s": round(time.monotonic() - started, 2),
        }
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if successes == args_cli.num_demos else 1
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
