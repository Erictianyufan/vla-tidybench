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
parser.add_argument("--showcase", action="store_true", help="record the furnished scene and hero camera")
parser.add_argument("--policy-host", help="query a live pi0.5 server and compose its bounded residual")
parser.add_argument("--policy-port", type=int, default=8000)
parser.add_argument("--policy-residual-weight", type=float, default=0.0001)
parser.add_argument("--policy-replan-steps", type=int, default=4)
parser.add_argument(
    "--actions-only",
    action="store_true",
    help="disable RGB sensors when collecting a recovery trajectory rather than a training dataset",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.dataset_file.exists() and not args_cli.overwrite:
    parser.error(f"dataset exists: {args_cli.dataset_file}; pass --overwrite")
if args_cli.num_demos < 1 or args_cli.max_attempts < args_cli.num_demos:
    parser.error("invalid demo/attempt count")
if args_cli.policy_host and not 0.0 < args_cli.policy_residual_weight <= 0.05:
    parser.error("--policy-residual-weight must be in (0, 0.05]")

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

from vla_tidybench.isaac import TidyBenchDrawerEnvCfg, TidyBenchDrawerShowcaseEnvCfg  # noqa: E402
from vla_tidybench.policy_bridge.action_adapter import ActionAdapter  # noqa: E402
from vla_tidybench.policy_bridge.websocket_client import PolicyClient  # noqa: E402
from vla_tidybench.task_metrics import drawer_skill_success  # noqa: E402

_startup_mark("drawer_cfg")


TEACHER_VERSION = "tidybench_truth_fsm_dls_v1"
PROMPTS = {
    "open": "open the top drawer",
    "pick": "pick up the medicine bottle",
    "place": "put the medicine bottle into the top drawer",
    "close": "close the top drawer",
    "full": "put the medicine bottle into the top drawer and close it",
}

PLACE_HELD_JOINT_POS = {
    "panda_joint1": 0.20995605,
    "panda_joint2": -1.4162296,
    "panda_joint3": -0.45338604,
    "panda_joint4": -2.3953204,
    "panda_joint5": -0.24656539,
    "panda_joint6": 2.636889,
    "panda_joint7": -1.0277694,
    "panda_finger_joint1": 0.02456491,
    "panda_finger_joint2": 0.02480536,
}
PLACE_HELD_OBJECT_POSE = (
    (0.24361168, -0.24285677, 0.80261046),
    (-0.73046213, -0.19504146, 0.26132822, 0.60007614),
)
CLOSE_OBJECT_POSE = (
    (0.61, -0.10684, 0.73107),
    (-0.00772, -0.00638, 0.07677, 0.997),
)


def skill_for_phase(phase: Phase) -> str:
    name = phase.name
    if name == "SETTLE" or name.startswith("HANDLE_") or name == "DRAWER_PULL":
        return "open"
    if name.startswith("PICK_"):
        return "pick"
    if name.startswith("PLACE_"):
        return "place"
    return "close"


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
        if hasattr(self, "drawer_joint_idx"):
            cabinet = self.env.scene["cabinet"]
            cabinet.write_joint_stiffness_to_sim(0.0, joint_ids=[self.drawer_joint_idx])
            cabinet.write_joint_damping_to_sim(20.0, joint_ids=[self.drawer_joint_idx])
        self.phase = Phase.SETTLE
        self.phase_steps = 0
        self.stable_steps = 0
        self.last_reason = ""
        self.fixed_pull_target = None
        self.fixed_target = None
        self.fixed_quat = None
        self.place_handle_anchor = None
        self.drawer_hold_target = None
        self.pick_jitter = self.rng.uniform(-0.003, 0.003, size=2)
        self.place_jitter = self.rng.uniform(-0.006, 0.006, size=2)
        self.initial_drawer = self._drawer_pos()
        self.initial_object_xyz = self._object()[0, :3].detach().cpu().numpy().copy()

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
            return Phase.PLACE_ABOVE
        return Phase.CLOSE_APPROACH

    def _after_open(self) -> Phase:
        return Phase.DONE if self.skill == "open" else Phase.PICK_ABOVE

    def _after_pick(self) -> Phase:
        return Phase.DONE if self.skill == "pick" else Phase.PLACE_CLEARANCE

    def _after_place(self) -> Phase:
        return Phase.DONE if self.skill == "place" else Phase.CLOSE_APPROACH

    def action(self) -> torch.Tensor:
        self.phase_steps += 1
        self._update_drawer_hold()
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
            # Grasp the upper body of the medicine bottle. Descending to the
            # old can-center target makes the Franka palm hit the child-proof
            # cap and slide sideways before the fingers close.
            target[:, 2] += 0.055
            action = self._pose_action(target, self.down_quat, 1.0, pos_threshold=0.018)
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
            action = self._pose_action(
                self.fixed_target,
                self.down_quat,
                -1.0,
                pos_threshold=0.055,
                rot_threshold=3.2,
            )
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
            action = self._pose_action(
                self.fixed_target,
                self.down_quat,
                -1.0,
                pos_threshold=0.080,
                rot_threshold=3.2,
            )
            if self.stable_steps >= 4:
                self._set_phase(Phase.PLACE_INSERT)
            return action

        if self.phase is Phase.PLACE_INSERT:
            if self.fixed_target is None:
                self.fixed_target = self.place_handle_anchor.clone()
                # The audited PLACE trajectory releases just behind the
                # drawer front plane. Going deeper makes a tall bottle and
                # the gripper collide with the cabinet top.
                self.fixed_target[:, 0] += 0.04
                self.fixed_target[:, 1] = -0.10 + torch.tensor(
                    self.place_jitter[1], dtype=torch.float32, device=self.device
                )
                self.fixed_target[:, 2] = 0.86
            action = self._pose_action(
                self.fixed_target,
                self.down_quat,
                -1.0,
                pos_threshold=0.105,
                rot_threshold=3.2,
            )
            if self.stable_steps >= 4:
                self._set_phase(Phase.PLACE_DESCEND)
            return action

        if self.phase is Phase.PLACE_DESCEND:
            if self.fixed_target is None:
                self.fixed_target = self.place_handle_anchor.clone()
                self.fixed_target[:, 0] += 0.04
                self.fixed_target[:, 1] = -0.10 + torch.tensor(
                    self.place_jitter[1], dtype=torch.float32, device=self.device
                )
                # A tall pharmacy bottle is released above the shallow
                # drawer instead of forcing the gripper through the opening.
                self.fixed_target[:, 2] = 0.88
            action = self._pose_action(
                self.fixed_target,
                self.down_quat,
                -1.0,
                pos_threshold=0.060,
                rot_threshold=3.2,
            )
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
            action = self._pose_action(
                target,
                self.down_quat,
                1.0,
                pos_threshold=0.060,
                rot_threshold=3.2,
            )
            if self.stable_steps >= 3:
                self._set_phase(self._after_place())
            return action

        if self.phase is Phase.CLOSE_APPROACH:
            target = handle.clone()
            # Reorient at clearance after the vertical bottle drop, then move
            # to the handle in CLOSE_ALIGN.
            target[:, 0] -= 0.18
            action = self._pose_action(
                target,
                self.handle_quat,
                1.0,
                pos_threshold=0.030,
                rot_threshold=3.2,
            )
            if self._drawer_pos() <= 0.04 and self.phase_steps > 20:
                self._set_phase(Phase.CLOSE_RELEASE)
                return action
            if self.stable_steps >= 3 or self.phase_steps >= 30:
                self._set_phase(Phase.CLOSE_ALIGN)
            return action

        if self.phase is Phase.CLOSE_ALIGN:
            target = handle.clone()
            target[:, 0] -= 0.008
            action = self._pose_action(
                target,
                self.handle_quat,
                1.0,
                pos_threshold=0.150,
                rot_threshold=3.2,
            )
            if self.stable_steps >= 3 or self.phase_steps >= 30:
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

    def _update_drawer_hold(self) -> None:
        """Keep the open drawer fixed while carrying the payload, then release it for CLOSE."""

        if self.skill != "full":
            return
        carrying_phase = self.phase.name.startswith("PICK_") or self.phase.name.startswith("PLACE_")
        cabinet = self.env.scene["cabinet"]
        if carrying_phase:
            if self.drawer_hold_target is None:
                # The placement controller was validated with a 0.36 m clear
                # opening; the OPEN success gate itself remains 0.30 m.
                self.drawer_hold_target = max(self._drawer_pos(), 0.36)
                cabinet.write_joint_stiffness_to_sim(10000.0, joint_ids=[self.drawer_joint_idx])
                cabinet.write_joint_damping_to_sim(500.0, joint_ids=[self.drawer_joint_idx])
            target = torch.tensor(
                [[self.drawer_hold_target]], dtype=torch.float32, device=self.device
            )
            cabinet.set_joint_position_target(target, joint_ids=[self.drawer_joint_idx])
        elif self.drawer_hold_target is not None and self.phase.name.startswith("CLOSE_"):
            cabinet.write_joint_stiffness_to_sim(0.0, joint_ids=[self.drawer_joint_idx])
            cabinet.write_joint_damping_to_sim(20.0, joint_ids=[self.drawer_joint_idx])
            self.drawer_hold_target = None

    def _object_in_drawer(self) -> bool:
        obj = self._object()[0]
        handle_x = self._handle()[0, 0]
        # The 4.5 cm cuboid center must be at least one half-width plus a
        # small margin behind the handle/front-plane proxy.
        return bool(obj[2] > 0.68 and obj[2] < 0.86 and obj[0] > handle_x + 0.023 and abs(obj[1]) < 0.26)

    def success(self) -> bool:
        obj = self._object()[0, :3]
        fingers = self.env.scene["robot"].data.joint_pos.torch[0, -2:]
        return drawer_skill_success(
            self.skill,
            initial_drawer_m=self.initial_drawer,
            initial_object_xyz=self.initial_object_xyz,
            drawer_m=self._drawer_pos(),
            object_xyz=obj.detach().cpu().numpy(),
            handle_x=float(self._handle()[0, 0]),
            finger_positions=fingers.detach().cpu().numpy(),
        )


def make_cfg() -> TidyBenchDrawerEnvCfg:
    output = args_cli.dataset_file.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and args_cli.overwrite:
        output.unlink()
    cfg = TidyBenchDrawerShowcaseEnvCfg() if args_cli.showcase else TidyBenchDrawerEnvCfg()
    cfg.sim.device = args_cli.device
    if args_cli.actions_only:
        cfg.scene.table_cam = None
        cfg.scene.wrist_cam = None
        cfg.observations.policy.table_cam = None
        cfg.observations.policy.wrist_cam = None
        cfg.image_obs_list = []
        cfg.num_rerenders_on_reset = 0
    if args_cli.skill in ("pick", "place", "close"):
        # This is part of the environment's reset state so RecorderManager
        # serializes the exact prerequisite state and physical replay can
        # reproduce the episode without an out-of-band teleport.
        cfg.scene.cabinet.init_state.joint_pos["drawer_top_joint"] = 0.39 if args_cli.skill == "place" else 0.36
    if args_cli.skill == "place":
        cfg.scene.robot.init_state.joint_pos = PLACE_HELD_JOINT_POS
        cfg.scene.target_object.init_state.pos, cfg.scene.target_object.init_state.rot = PLACE_HELD_OBJECT_POSE
        cfg.scene.cabinet.actuators["drawers"].stiffness = 10000.0
        cfg.scene.cabinet.actuators["drawers"].damping = 500.0
    elif args_cli.skill == "close":
        cfg.scene.target_object.init_state.pos, cfg.scene.target_object.init_state.rot = CLOSE_OBJECT_POSE
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


def add_metadata(
    path: Path,
    attempts: int,
    successes: int,
    successful_phase_traces: list[list[str]],
    successful_policy_payloads: list[dict[str, object]],
) -> None:
    with h5py.File(path, "r+") as dataset:
        data = dataset["data"]
        data.attrs["collector"] = TEACHER_VERSION
        data.attrs["skill"] = args_cli.skill
        data.attrs["language_instruction"] = PROMPTS[args_cli.skill]
        data.attrs["attempts"] = attempts
        data.attrs["successful_episodes"] = successes
        data.attrs["seed"] = args_cli.seed
        string_type = h5py.string_dtype(encoding="utf-8")
        payloads = successful_policy_payloads or [{} for _ in successful_phase_traces]
        for demo, phase_trace, payload in zip(
            data.values(), successful_phase_traces, payloads, strict=True
        ):
            demo.attrs["source"] = "scripted_truth_teacher"
            demo.attrs["teacher_version"] = TEACHER_VERSION
            demo.attrs["skill"] = args_cli.skill
            demo.attrs["language_instruction"] = PROMPTS[args_cli.skill]
            demo.create_dataset(
                "teacher_phase",
                data=np.asarray(phase_trace, dtype=object),
                dtype=string_type,
            )
            if payload:
                demo.attrs["policy"] = "pi0.5-four-skill-lora+dls-live-recovery"
                demo.attrs["policy_residual_weight"] = args_cli.policy_residual_weight
                demo.attrs["mean_infer_ms"] = float(payload["mean_infer_ms"])
                demo.create_dataset("hero_cam", data=payload["hero_cam"], compression="gzip")
                demo.create_dataset(
                    "policy_actions", data=payload["policy_actions"], compression="gzip"
                )
                demo.create_dataset(
                    "policy_skills",
                    data=np.asarray(payload["skills"], dtype=object),
                    dtype=string_type,
                )
                demo.create_dataset(
                    "policy_prompts",
                    data=np.asarray(payload["prompts"], dtype=object),
                    dtype=string_type,
                )


def main() -> int:
    faulthandler.cancel_dump_traceback_later()
    output = args_cli.dataset_file.resolve()
    env = gym.make("Isaac-Open-Drawer-Franka-IK-Rel-v0", cfg=make_cfg()).unwrapped
    teacher = DrawerTeacher(env, args_cli.skill, args_cli.seed)
    adapter = ActionAdapter()
    policy_client = (
        PolicyClient(args_cli.policy_host, args_cli.policy_port, timeout_s=120.0)
        if args_cli.policy_host
        else None
    )
    attempts = 0
    successes = 0
    successful_phase_traces: list[list[str]] = []
    successful_policy_payloads: list[dict[str, object]] = []
    started = time.monotonic()
    try:
        while successes < args_cli.num_demos and attempts < args_cli.max_attempts and simulation_app.is_running():
            attempts += 1
            env.reset(seed=args_cli.seed + attempts - 1)
            if args_cli.showcase:
                env.scene["hero_cam"].set_world_poses_from_view(
                    eyes=torch.tensor([[-1.85, -2.35, 1.65]], device=env.device),
                    targets=torch.tensor([[0.50, 0.0, 0.58]], device=env.device),
                )
                for _ in range(3):
                    env.sim.render()
            initialize_skill_state(env, args_cli.skill)
            teacher.reset_episode()
            print(f"attempt {attempts}/{args_cli.max_attempts} skill={args_cli.skill}", flush=True)
            succeeded = False
            phase_trace: list[str] = []
            hero_frames: list[np.ndarray] = []
            policy_actions: list[np.ndarray] = []
            policy_skills: list[str] = []
            policy_prompts: list[str] = []
            policy_latencies: list[float] = []
            policy_chunk = np.empty((0, 7), dtype=np.float32)
            policy_chunk_index = 0
            policy_chunk_skill = ""
            with torch.inference_mode():
                for step in range(args_cli.max_steps):
                    base_action = teacher.action()
                    active_skill = skill_for_phase(teacher.phase)
                    proposed = np.zeros(7, dtype=np.float32)
                    if policy_client is not None:
                        if (
                            policy_chunk_index >= min(len(policy_chunk), args_cli.policy_replan_steps)
                            or active_skill != policy_chunk_skill
                        ):
                            robot = env.scene["robot"]
                            q = robot.data.joint_pos.torch[0].detach().cpu().numpy().astype(np.float32)
                            qd = robot.data.joint_vel.torch[0].detach().cpu().numpy().astype(np.float32)
                            table = (
                                env.scene["table_cam"].data.output["rgb"].torch[0, ..., :3]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.uint8)
                            )
                            wrist = (
                                env.scene["wrist_cam"].data.output["rgb"].torch[0, ..., :3]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.uint8)
                            )
                            infer_started = time.perf_counter()
                            response = policy_client.infer(
                                {
                                    "observation/image": table,
                                    "observation/wrist_image": wrist,
                                    "observation/state": np.concatenate((q, qd), dtype=np.float32),
                                    "prompt": PROMPTS[active_skill],
                                }
                            )
                            policy_latencies.append((time.perf_counter() - infer_started) * 1000.0)
                            policy_chunk = np.asarray(response["actions"], dtype=np.float32)
                            if (
                                policy_chunk.ndim != 2
                                or policy_chunk.shape[1] != 7
                                or not np.isfinite(policy_chunk).all()
                            ):
                                raise ValueError(f"malformed policy chunk {policy_chunk.shape}")
                            policy_chunk_index = 0
                            policy_chunk_skill = active_skill
                        proposed = policy_chunk[policy_chunk_index]
                        policy_chunk_index += 1
                        raw = base_action[0].detach().cpu().numpy().astype(np.float32)
                        proposed_raw = adapter.to_isaac(proposed)
                        raw[:6] = np.clip(
                            raw[:6] + args_cli.policy_residual_weight * proposed_raw[:6],
                            -1.0,
                            1.0,
                        )
                        raw[6] = float(base_action[0, 6])
                        action = torch.as_tensor(raw[None], dtype=torch.float32, device=env.device)
                    else:
                        action = base_action
                    phase_trace.append(teacher.phase.name)
                    env.step(action)
                    if args_cli.showcase:
                        hero_frames.append(
                            env.scene["hero_cam"].data.output["rgb"].torch[0, ..., :3]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.uint8)
                        )
                    if policy_client is not None:
                        policy_actions.append(proposed.copy())
                        policy_skills.append(active_skill)
                        policy_prompts.append(PROMPTS[active_skill])
                    if step % 40 == 0:
                        eef_pos = teacher._eef()[0][0].detach().cpu().tolist()
                        fixed = (
                            None
                            if teacher.fixed_target is None
                            else teacher.fixed_target[0].detach().cpu().tolist()
                        )
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
                successful_phase_traces.append(phase_trace)
                if policy_client is not None:
                    successful_policy_payloads.append(
                        {
                            "hero_cam": np.asarray(hero_frames),
                            "policy_actions": np.asarray(policy_actions, dtype=np.float32),
                            "skills": policy_skills,
                            "prompts": policy_prompts,
                            "mean_infer_ms": float(np.mean(policy_latencies)),
                        }
                    )
                print(f"SUCCESS {successes}/{args_cli.num_demos} in {step + 1} steps", flush=True)
            else:
                print(f"FAILED: {teacher.last_reason or teacher.phase.name}", flush=True)
        env.recorder_manager.close()
        if output.exists():
            add_metadata(
                output,
                attempts,
                successes,
                successful_phase_traces,
                successful_policy_payloads,
            )
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
        if policy_client is not None:
            policy_client.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
