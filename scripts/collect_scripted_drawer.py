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
parser.add_argument(
    "--home-tabletop",
    action="store_true",
    help="PICK or FULL: lift the complete scene onto a physical table in a home-study set",
)
parser.add_argument("--policy-host", help="query a live pi0.5 server and compose its bounded residual")
parser.add_argument("--policy-port", type=int, default=8000)
parser.add_argument("--policy-residual-weight", type=float, default=0.0001)
parser.add_argument("--policy-replan-steps", type=int, default=4)
parser.add_argument("--policy-gripper-close-confirm-steps", type=int, default=1)
parser.add_argument("--policy-gripper-open-confirm-steps", type=int, default=1)
parser.add_argument(
    "--policy-ensemble-samples",
    type=int,
    default=1,
    help="average independent pi0.5 action chunks at each replan; all samples use the same live observation",
)
parser.add_argument(
    "--dagger-dataset-file",
    type=Path,
    help="save failed on-policy observations with truth-teacher labels for training only",
)
parser.add_argument(
    "--route-handle-gripper-prompt",
    action="store_true",
    help="use the truth-FSM handle-close/pull phase to route a specific language prompt",
)
parser.add_argument(
    "--route-place-release-prompt",
    action="store_true",
    help="use the truth-FSM PLACE_RELEASE phase to ask pi0.5 explicitly to release the bottle",
)
parser.add_argument(
    "--policy-first", action="store_true", help="execute pi0.5 actions with only a bounded DLS correction"
)
parser.add_argument("--dls-max-correction-norm", type=float, default=0.10)
parser.add_argument(
    "--dls-correction-mode", choices=("isotropic", "translation_first", "skill_router"), default="isotropic"
)
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
if args_cli.policy_host and not args_cli.policy_first and not 0.0 < args_cli.policy_residual_weight <= 0.05:
    parser.error("--policy-residual-weight must be in (0, 0.05]")
if args_cli.policy_first and not args_cli.policy_host:
    parser.error("--policy-first requires --policy-host")
if args_cli.dagger_dataset_file and not args_cli.policy_first:
    parser.error("--dagger-dataset-file requires --policy-first")
if args_cli.dagger_dataset_file and args_cli.dagger_dataset_file.exists():
    parser.error("dagger dataset already exists; refusing to overwrite it")
if args_cli.route_handle_gripper_prompt and not args_cli.policy_first:
    parser.error("--route-handle-gripper-prompt requires --policy-first")
if args_cli.route_place_release_prompt and not args_cli.policy_first:
    parser.error("--route-place-release-prompt requires --policy-first")
if args_cli.policy_first and not 0.0 <= args_cli.dls_max_correction_norm <= 0.25:
    parser.error("--dls-max-correction-norm must be in [0, 0.25]")
if args_cli.policy_ensemble_samples < 1:
    parser.error("--policy-ensemble-samples must be positive")
if min(args_cli.policy_gripper_close_confirm_steps, args_cli.policy_gripper_open_confirm_steps) < 1:
    parser.error("gripper confirmation steps must be positive")
if not args_cli.policy_first and (
    args_cli.policy_gripper_close_confirm_steps != 1 or args_cli.policy_gripper_open_confirm_steps != 1
):
    parser.error("model gripper confirmation requires --policy-first")
if args_cli.home_tabletop:
    if args_cli.skill not in ("pick", "full"):
        parser.error("--home-tabletop supports --skill pick or full")
    if args_cli.actions_only:
        parser.error("--home-tabletop requires cameras for the three-view demo")
    args_cli.showcase = True

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
from vla_tidybench.data.isaac_hdf5 import relative_deployable_state  # noqa: E402

_startup_mark("numpy")
import torch  # noqa: E402

_startup_mark("torch")
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402

_startup_mark("recorder_cfg")
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab.utils.math import compute_pose_error  # noqa: E402

_startup_mark("isaac_utils")

from vla_tidybench.isaac import (  # noqa: E402
    TidyBenchDrawerEnvCfg,
    TidyBenchDrawerShowcaseEnvCfg,
    TidyBenchHomeTablePickEnvCfg,
)
from vla_tidybench.policy_bridge.action_adapter import ActionAdapter  # noqa: E402
from vla_tidybench.policy_bridge.policy_first import ModelGripperDebouncer, compose_policy_first  # noqa: E402
from vla_tidybench.policy_bridge.websocket_client import PolicyClient  # noqa: E402

_startup_mark("drawer_cfg")


TEACHER_VERSION = "tidybench_truth_fsm_dls_v1"
PROMPTS = {
    "open": "open the top drawer",
    "pick": "pick up the medicine bottle",
    "place": "put the medicine bottle into the top drawer",
    "close": "close the top drawer",
    "full": "put the medicine bottle into the top drawer and close it",
}
if args_cli.home_tabletop:
    PROMPTS["pick"] = "pick up the medicine bottle from the home table"
HANDLE_GRIP_PROMPT = "close the gripper on the top drawer handle and pull it open"
PLACE_RELEASE_PROMPT = "open the gripper and release the medicine bottle into the top drawer"


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
    PICK_HOLD = enum.auto()
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

    def __init__(self, env, skill: str, seed: int, height_offset: float = 0.0) -> None:
        self.env = env
        self.skill = skill
        self.height_offset = height_offset
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
        if self.skill == "pick":
            return Phase.PICK_HOLD if self.height_offset else Phase.DONE
        return Phase.PLACE_CLEARANCE

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
            if self.phase_steps > 180:
                self.last_reason = "handle approach did not converge within 180 steps"
                self._set_phase(Phase.FAILED)
                return self._hold(1.0)
            target = handle.clone()
            target[:, 0] -= 0.10
            action = self._pose_action(target, self.handle_quat, 1.0, pos_threshold=0.015)
            if self.stable_steps >= 4:
                self._set_phase(Phase.HANDLE_ALIGN)
            return action

        if self.phase is Phase.HANDLE_ALIGN:
            if self.phase_steps > 120:
                self.last_reason = "handle alignment did not converge within 120 steps"
                self._set_phase(Phase.FAILED)
                return self._hold(1.0)
            target = handle.clone()
            target[:, 0] -= 0.008
            action = self._pose_action(
                target,
                self.handle_quat,
                1.0,
                pos_threshold=0.004 if args_cli.policy_first else 0.008,
                rot_threshold=0.06 if args_cli.policy_first else 0.10,
            )
            if self.stable_steps >= (6 if args_cli.policy_first else 4):
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
            clearance = float(torch.linalg.vector_norm(self._eef()[0] - handle, dim=1)[0])
            # The target itself is 6 cm behind the moving handle. Requiring
            # 8 cm clearance made this phase unreachable at a 0.40 m opening.
            if self.stable_steps >= 3 or (args_cli.policy_first and self.phase_steps >= 12 and clearance >= 0.05):
                self._set_phase(self._after_open())
            return action

        if self.phase is Phase.PICK_ABOVE:
            if args_cli.policy_first and self.phase_steps > 200:
                self.last_reason = "pick approach did not converge within 200 steps"
                self._set_phase(Phase.FAILED)
                return self._hold(1.0)
            target = obj.clone()
            target[:, :2] += torch.tensor(self.pick_jitter, dtype=torch.float32, device=self.device)
            target[:, 2] += 0.14
            action = self._pose_action(
                target,
                self.down_quat,
                1.0,
                pos_threshold=0.020 if args_cli.policy_first else 0.014,
                rot_threshold=0.20 if args_cli.policy_first else 0.15,
            )
            if self.stable_steps >= (3 if args_cli.policy_first else 4):
                self._set_phase(Phase.PICK_DESCEND)
            return action

        if self.phase is Phase.PICK_DESCEND:
            if args_cli.policy_first and self.phase_steps > 200:
                self.last_reason = "pick descent did not converge within 200 steps"
                self._set_phase(Phase.FAILED)
                return self._hold(1.0)
            target = obj.clone()
            target[:, :2] += torch.tensor(self.pick_jitter, dtype=torch.float32, device=self.device)
            # Grasp the upper body of the medicine bottle. Descending to the
            # old can-center target makes the Franka palm hit the child-proof
            # cap and slide sideways before the fingers close.
            target[:, 2] += 0.055
            action = self._pose_action(
                target,
                self.down_quat,
                1.0,
                pos_threshold=0.022 if args_cli.policy_first else 0.018,
                rot_threshold=0.20 if args_cli.policy_first else 0.15,
            )
            if self.stable_steps >= (3 if args_cli.policy_first else 4):
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
            target[:, 2] = 0.30 + self.height_offset
            action = self._pose_action(target, self.down_quat, -1.0, pos_threshold=0.025)
            if float(obj[0, 2]) >= 0.18 + self.height_offset:
                if float(obj[0, 2]) < 0.12 + self.height_offset:
                    self.last_reason = f"grasp failed, object z={float(obj[0, 2]):.3f}"
                    self._set_phase(Phase.FAILED)
                else:
                    self._set_phase(self._after_pick())
            return action

        if self.phase is Phase.PICK_HOLD:
            if float(obj[0, 2]) < 0.18 + self.height_offset:
                self.last_reason = "object dropped during tabletop hold"
                self._set_phase(Phase.FAILED)
            elif self.phase_steps >= 15:
                self._set_phase(Phase.DONE)
            return self._hold(-1.0)

        if self.phase is Phase.PLACE_CLEARANCE:
            if self.fixed_target is None:
                self.fixed_target = self._eef()[0].clone()
                self.fixed_target[:, 2] = 0.82 + self.height_offset
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
                self.fixed_target[:, 2] = 0.86 + self.height_offset
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
                self.fixed_target[:, 2] = 0.86 + self.height_offset
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
                self.fixed_target[:, 2] = 0.88 + self.height_offset
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
            target = torch.tensor([[self.drawer_hold_target]], dtype=torch.float32, device=self.device)
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
        return bool(
            0.68 + self.height_offset < obj[2] < 0.86 + self.height_offset
            and obj[0] > handle_x + 0.023
            and abs(obj[1]) < 0.26
        )

    def success(self) -> bool:
        if self.skill == "open":
            return self._drawer_pos() >= 0.30
        if self.skill == "pick":
            return float(self._object()[0, 2]) >= 0.18 + self.height_offset
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
    cfg = (
        TidyBenchHomeTablePickEnvCfg()
        if args_cli.home_tabletop
        else TidyBenchDrawerShowcaseEnvCfg()
        if args_cli.showcase
        else TidyBenchDrawerEnvCfg()
    )
    cfg.sim.device = args_cli.device
    if args_cli.actions_only:
        cfg.scene.table_cam = None
        cfg.scene.wrist_cam = None
        cfg.observations.policy.table_cam = None
        cfg.observations.policy.wrist_cam = None
        cfg.image_obs_list = []
        cfg.num_rerenders_on_reset = 0
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


def add_metadata(
    path: Path,
    attempts: int,
    successes: int,
    successful_phase_traces: list[list[str]],
    successful_policy_payloads: list[dict[str, object]],
    successful_hero_frames: list[np.ndarray],
) -> None:
    with h5py.File(path, "r+") as dataset:
        data = dataset["data"]
        data.attrs["collector"] = TEACHER_VERSION
        data.attrs["skill"] = args_cli.skill
        data.attrs["language_instruction"] = PROMPTS[args_cli.skill]
        data.attrs["attempts"] = attempts
        data.attrs["successful_episodes"] = successes
        data.attrs["seed"] = args_cli.seed
        data.attrs["scene_variant"] = f"home_tabletop_{args_cli.skill}" if args_cli.home_tabletop else "original_drawer"
        prompt_routes = []
        if args_cli.route_handle_gripper_prompt:
            prompt_routes.append("truth_fsm_handle_grip")
        if args_cli.route_place_release_prompt:
            prompt_routes.append("truth_fsm_place_release")
        data.attrs["prompt_router"] = "+".join(prompt_routes) or "skill_only"
        string_type = h5py.string_dtype(encoding="utf-8")
        payloads = successful_policy_payloads or [{} for _ in successful_phase_traces]
        episode_names = sorted(data.keys(), key=lambda name: int(name.removeprefix("demo_")))
        for episode_name, phase_trace, payload, hero_frames in zip(
            episode_names, successful_phase_traces, payloads, successful_hero_frames, strict=True
        ):
            demo = data[episode_name]
            demo.attrs["source"] = (
                "pi0.5_policy_with_bounded_truth_guided_dls" if args_cli.policy_first else "scripted_truth_teacher"
            )
            demo.attrs["teacher_version"] = TEACHER_VERSION
            demo.attrs["skill"] = args_cli.skill
            demo.attrs["language_instruction"] = PROMPTS[args_cli.skill]
            demo.attrs["scene_variant"] = data.attrs["scene_variant"]
            demo.attrs["state_convention"] = "joint_pos_rel_joint_vel_rel"
            demo.attrs["prompt_router"] = data.attrs["prompt_router"]
            demo.create_dataset(
                "teacher_phase",
                data=np.asarray(phase_trace, dtype=object),
                dtype=string_type,
            )
            if payload:
                policy_name = str(payload["policy_name"])
                demo.attrs["policy"] = (
                    f"{policy_name}-primary+bounded-dls"
                    if args_cli.policy_first
                    else f"{policy_name}+dls-live-recovery"
                )
                demo.attrs["policy_checkpoint_sha256"] = str(payload["checkpoint_sha256"])
                demo.attrs["control_mode"] = "policy_first" if args_cli.policy_first else "teacher_first"
                if args_cli.policy_first:
                    demo.attrs["dls_max_correction_norm"] = args_cli.dls_max_correction_norm
                    demo.attrs["dls_correction_mode"] = args_cli.dls_correction_mode
                    demo.attrs["policy_gripper_close_confirm_steps"] = args_cli.policy_gripper_close_confirm_steps
                    demo.attrs["policy_gripper_open_confirm_steps"] = args_cli.policy_gripper_open_confirm_steps
                else:
                    demo.attrs["policy_residual_weight"] = args_cli.policy_residual_weight
                demo.attrs["mean_infer_ms"] = float(payload["mean_infer_ms"])
                demo.attrs["inference_calls"] = int(payload["inference_calls"])
                demo.attrs["policy_ensemble_samples"] = int(payload["ensemble_samples"])
                demo.create_dataset("hero_cam", data=payload["hero_cam"], compression="gzip")
                demo.create_dataset("policy_actions", data=payload["policy_actions"], compression="gzip")
                for name in (
                    "policy_model_raw_actions",
                    "policy_raw_actions",
                    "dls_reference_actions",
                    "correction_actions",
                    "executed_actions",
                ):
                    demo.create_dataset(name, data=payload[name], compression="gzip")
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
            elif len(hero_frames):
                demo.attrs["policy"] = (
                    "scripted-truth-teacher-home-tabletop"
                    if args_cli.home_tabletop
                    else "scripted-teacher-camera-preview"
                )
                demo.create_dataset("hero_cam", data=hero_frames, compression="gzip")


def save_dagger_failure(
    path: Path,
    *,
    table_images: list[np.ndarray],
    wrist_images: list[np.ndarray],
    joint_positions: list[np.ndarray],
    joint_velocities: list[np.ndarray],
    teacher_actions: list[np.ndarray],
    model_raw_actions: list[np.ndarray],
    executed_actions: list[np.ndarray],
    correction_actions: list[np.ndarray],
    phases: list[str],
    prompts: list[str],
    checkpoint_sha256: str,
    reason: str,
) -> None:
    """Record failed on-policy states with teacher labels, never as success."""

    if not teacher_actions:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as recording:
        recording.attrs["format_version"] = 1
        recording.attrs["training_only"] = True
        data = recording.require_group("data")
        episode = data.create_group(f"demo_{len(data)}")
        episode.attrs["success"] = False
        episode.attrs["source"] = "on_policy_truth_teacher_labeled_failure"
        episode.attrs["label_source"] = "truth_guided_teacher"
        episode.attrs["policy_checkpoint_sha256"] = checkpoint_sha256
        episode.attrs["failure_reason"] = reason
        episode.attrs["state_convention"] = "joint_pos_rel_joint_vel_rel"
        episode.create_dataset("actions", data=np.asarray(teacher_actions, dtype=np.float32), compression="gzip")
        episode.create_dataset(
            "model_raw_actions", data=np.asarray(model_raw_actions, dtype=np.float32), compression="gzip"
        )
        episode.create_dataset(
            "executed_actions", data=np.asarray(executed_actions, dtype=np.float32), compression="gzip"
        )
        episode.create_dataset(
            "correction_actions", data=np.asarray(correction_actions, dtype=np.float32), compression="gzip"
        )
        observations = episode.create_group("obs")
        for key, values in (
            ("table_cam", table_images),
            ("wrist_cam", wrist_images),
            ("joint_pos", joint_positions),
            ("joint_vel", joint_velocities),
        ):
            observations.create_dataset(key, data=np.asarray(values), compression="gzip")
        text_type = h5py.string_dtype(encoding="utf-8")
        episode.create_dataset("teacher_phase", data=np.asarray(phases, dtype=object), dtype=text_type)
        episode.create_dataset("policy_prompts", data=np.asarray(prompts, dtype=object), dtype=text_type)


def main() -> int:
    faulthandler.cancel_dump_traceback_later()
    output = args_cli.dataset_file.resolve()
    env = gym.make("Isaac-Open-Drawer-Franka-IK-Rel-v0", cfg=make_cfg()).unwrapped
    teacher = DrawerTeacher(env, args_cli.skill, args_cli.seed, height_offset=0.75 if args_cli.home_tabletop else 0.0)
    adapter = ActionAdapter()
    policy_client = (
        PolicyClient(args_cli.policy_host, args_cli.policy_port, timeout_s=120.0) if args_cli.policy_host else None
    )
    policy_metadata = policy_client.metadata if policy_client is not None else {}
    if policy_client is not None:
        policy_name = str(policy_metadata.get("policy", ""))
        checkpoint_sha256 = str(policy_metadata.get("checkpoint_sha256", ""))
        if not policy_name.startswith("pi0.5-drawer-"):
            raise ValueError(f"unexpected policy service: {policy_name!r}")
        if len(checkpoint_sha256) != 64 or any(char not in "0123456789abcdef" for char in checkpoint_sha256):
            raise ValueError("policy service did not provide a valid checkpoint SHA-256")
    attempts = 0
    successes = 0
    successful_phase_traces: list[list[str]] = []
    successful_policy_payloads: list[dict[str, object]] = []
    successful_hero_frames: list[np.ndarray] = []
    started = time.monotonic()
    try:
        while successes < args_cli.num_demos and attempts < args_cli.max_attempts and simulation_app.is_running():
            attempts += 1
            env.reset(seed=args_cli.seed + attempts - 1)
            if args_cli.showcase:
                offset = 0.75 if args_cli.home_tabletop else 0.0
                env.scene["hero_cam"].set_world_poses_from_view(
                    eyes=torch.tensor([[-1.85, -2.35, 1.65 + offset]], device=env.device),
                    targets=torch.tensor([[0.50, 0.0, 0.58 + offset]], device=env.device),
                )
                if args_cli.home_tabletop:
                    env.scene["table_cam"].set_world_poses_from_view(
                        eyes=torch.tensor([[0.45, -1.35, 1.90]], device=env.device),
                        targets=torch.tensor([[0.25, -0.10, 0.98]], device=env.device),
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
            policy_model_raw_actions: list[np.ndarray] = []
            policy_raw_actions: list[np.ndarray] = []
            gripper_debouncer = ModelGripperDebouncer(
                close_after=args_cli.policy_gripper_close_confirm_steps,
                open_after=args_cli.policy_gripper_open_confirm_steps,
            )
            dls_reference_actions: list[np.ndarray] = []
            correction_actions: list[np.ndarray] = []
            executed_actions: list[np.ndarray] = []
            policy_skills: list[str] = []
            policy_prompts: list[str] = []
            policy_latencies: list[float] = []
            dagger_table_images: list[np.ndarray] = []
            dagger_wrist_images: list[np.ndarray] = []
            dagger_joint_positions: list[np.ndarray] = []
            dagger_joint_velocities: list[np.ndarray] = []
            dagger_teacher_actions: list[np.ndarray] = []
            policy_chunk = np.empty((0, 7), dtype=np.float32)
            policy_chunk_index = 0
            policy_chunk_skill = ""
            policy_chunk_prompt = ""
            with torch.inference_mode():
                for step in range(args_cli.max_steps):
                    base_action = teacher.action()
                    if teacher.phase is Phase.FAILED:
                        break
                    active_skill = skill_for_phase(teacher.phase)
                    active_prompt = (
                        PLACE_RELEASE_PROMPT
                        if args_cli.route_place_release_prompt
                        and teacher.phase in (Phase.PLACE_RELEASE, Phase.PLACE_RETREAT)
                        else HANDLE_GRIP_PROMPT
                        if args_cli.route_handle_gripper_prompt
                        and teacher.phase in (Phase.HANDLE_CLOSE, Phase.DRAWER_PULL)
                        else PROMPTS[active_skill]
                    )
                    proposed = np.zeros(7, dtype=np.float32)
                    if policy_client is not None:
                        if (
                            policy_chunk_index >= min(len(policy_chunk), args_cli.policy_replan_steps)
                            or active_skill != policy_chunk_skill
                            or active_prompt != policy_chunk_prompt
                        ):
                            robot = env.scene["robot"]
                            state = relative_deployable_state(
                                robot.data.joint_pos.torch[0].detach().cpu().numpy(),
                                robot.data.joint_vel.torch[0].detach().cpu().numpy(),
                                robot.data.default_joint_pos.torch[0].detach().cpu().numpy(),
                                robot.data.default_joint_vel.torch[0].detach().cpu().numpy(),
                            )
                            table = (
                                env.scene["table_cam"]
                                .data.output["rgb"]
                                .torch[0, ..., :3]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.uint8)
                            )
                            wrist = (
                                env.scene["wrist_cam"]
                                .data.output["rgb"]
                                .torch[0, ..., :3]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.uint8)
                            )
                            observation = {
                                "observation/image": table,
                                "observation/wrist_image": wrist,
                                "observation/state": state,
                                "prompt": active_prompt,
                            }
                            chunks = []
                            for _ in range(args_cli.policy_ensemble_samples):
                                infer_started = time.perf_counter()
                                response = policy_client.infer(observation)
                                policy_latencies.append((time.perf_counter() - infer_started) * 1000.0)
                                candidate = np.asarray(response["actions"], dtype=np.float32)
                                if (
                                    candidate.ndim != 2
                                    or candidate.shape[1] != 7
                                    or len(candidate) == 0
                                    or not np.isfinite(candidate).all()
                                ):
                                    raise ValueError(f"malformed policy chunk {candidate.shape}")
                                if chunks and candidate.shape != chunks[0].shape:
                                    raise ValueError("ensemble action chunks differ in shape")
                                chunks.append(candidate)
                            policy_chunk = np.mean(chunks, axis=0, dtype=np.float32)
                            policy_chunk_index = 0
                            policy_chunk_skill = active_skill
                            policy_chunk_prompt = active_prompt
                        proposed = policy_chunk[policy_chunk_index]
                        policy_chunk_index += 1
                        reference_raw = base_action[0].detach().cpu().numpy().astype(np.float32)
                        proposed_model_raw = adapter.to_isaac(proposed)
                        proposed_raw = proposed_model_raw.copy()
                        if args_cli.policy_first:
                            proposed_raw[6] = gripper_debouncer.update(float(proposed_model_raw[6]))
                            correction_mode = (
                                ("translation_first" if active_skill in ("pick", "place") else "isotropic")
                                if args_cli.dls_correction_mode == "skill_router"
                                else args_cli.dls_correction_mode
                            )
                            composed = compose_policy_first(
                                proposed_raw,
                                reference_raw,
                                max_correction_norm=args_cli.dls_max_correction_norm,
                                correction_mode=correction_mode,
                            )
                            raw = composed.executed_raw
                            correction = composed.correction_raw
                        else:
                            raw = reference_raw.copy()
                            raw[:6] = np.clip(
                                raw[:6] + args_cli.policy_residual_weight * proposed_raw[:6],
                                -1.0,
                                1.0,
                            )
                            correction = raw - reference_raw
                            raw[6] = reference_raw[6]
                        action = torch.as_tensor(raw[None], dtype=torch.float32, device=env.device)
                        if args_cli.dagger_dataset_file:
                            robot = env.scene["robot"]
                            dagger_state = relative_deployable_state(
                                robot.data.joint_pos.torch[0].detach().cpu().numpy(),
                                robot.data.joint_vel.torch[0].detach().cpu().numpy(),
                                robot.data.default_joint_pos.torch[0].detach().cpu().numpy(),
                                robot.data.default_joint_vel.torch[0].detach().cpu().numpy(),
                            )
                            dagger_joint_positions.append(dagger_state[:9].copy())
                            dagger_joint_velocities.append(dagger_state[9:].copy())
                            for camera_name, destination in (
                                ("table_cam", dagger_table_images),
                                ("wrist_cam", dagger_wrist_images),
                            ):
                                destination.append(
                                    env.scene[camera_name]
                                    .data.output["rgb"]
                                    .torch[0, ..., :3]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.uint8)
                                )
                            dagger_teacher_actions.append(reference_raw.copy())
                    else:
                        action = base_action
                    phase_trace.append(teacher.phase.name)
                    env.step(action)
                    if args_cli.showcase:
                        hero_frames.append(
                            env.scene["hero_cam"]
                            .data.output["rgb"]
                            .torch[0, ..., :3]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.uint8)
                        )
                    if policy_client is not None:
                        policy_actions.append(proposed.copy())
                        policy_model_raw_actions.append(proposed_model_raw.copy())
                        policy_raw_actions.append(proposed_raw.copy())
                        dls_reference_actions.append(reference_raw.copy())
                        correction_actions.append(correction.copy())
                        executed_actions.append(raw.copy())
                        policy_skills.append(active_skill)
                        policy_prompts.append(active_prompt)
                    if step % 40 == 0:
                        eef_pos = teacher._eef()[0][0].detach().cpu().tolist()
                        fixed = (
                            None if teacher.fixed_target is None else teacher.fixed_target[0].detach().cpu().tolist()
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
                successful_hero_frames.append(np.asarray(hero_frames, dtype=np.uint8))
                if policy_client is not None:
                    successful_policy_payloads.append(
                        {
                            "hero_cam": np.asarray(hero_frames),
                            "policy_actions": np.asarray(policy_actions, dtype=np.float32),
                            "policy_model_raw_actions": np.asarray(policy_model_raw_actions, dtype=np.float32),
                            "policy_raw_actions": np.asarray(policy_raw_actions, dtype=np.float32),
                            "dls_reference_actions": np.asarray(dls_reference_actions, dtype=np.float32),
                            "correction_actions": np.asarray(correction_actions, dtype=np.float32),
                            "executed_actions": np.asarray(executed_actions, dtype=np.float32),
                            "skills": policy_skills,
                            "prompts": policy_prompts,
                            "mean_infer_ms": float(np.mean(policy_latencies)),
                            "inference_calls": len(policy_latencies),
                            "ensemble_samples": args_cli.policy_ensemble_samples,
                            "policy_name": policy_metadata["policy"],
                            "checkpoint_sha256": policy_metadata["checkpoint_sha256"],
                        }
                    )
                print(f"SUCCESS {successes}/{args_cli.num_demos} in {step + 1} steps", flush=True)
            else:
                print(f"FAILED: {teacher.last_reason or teacher.phase.name}", flush=True)
                if args_cli.dagger_dataset_file and policy_client is not None:
                    save_dagger_failure(
                        args_cli.dagger_dataset_file,
                        table_images=dagger_table_images,
                        wrist_images=dagger_wrist_images,
                        joint_positions=dagger_joint_positions,
                        joint_velocities=dagger_joint_velocities,
                        teacher_actions=dagger_teacher_actions,
                        model_raw_actions=policy_model_raw_actions,
                        executed_actions=executed_actions,
                        correction_actions=correction_actions,
                        phases=phase_trace,
                        prompts=policy_prompts,
                        checkpoint_sha256=str(policy_metadata["checkpoint_sha256"]),
                        reason=teacher.last_reason or teacher.phase.name,
                    )
                if policy_actions:
                    model_motion = float(np.linalg.norm(np.asarray(policy_raw_actions)[:, :6], axis=1).sum())
                    correction_motion = float(np.linalg.norm(np.asarray(correction_actions)[:, :6], axis=1).sum())
                    print(
                        "  diagnostic_model_motion_fraction="
                        f"{model_motion / max(model_motion + correction_motion, 1e-8):.3f}",
                        flush=True,
                    )
                    for skill_name in dict.fromkeys(policy_skills):
                        mask = np.asarray(policy_skills) == skill_name
                        proposed_skill = np.asarray(policy_raw_actions)[mask]
                        reference_skill = np.asarray(dls_reference_actions)[mask]
                        executed_skill = np.asarray(executed_actions)[mask]
                        correction_skill = np.asarray(correction_actions)[mask]
                        print(
                            "  diagnostic "
                            f"skill={skill_name} steps={int(mask.sum())} "
                            f"policy_close_fraction={float(np.mean(proposed_skill[:, 6] < 0)):.3f} "
                            f"teacher_close_fraction={float(np.mean(reference_skill[:, 6] < 0)):.3f} "
                            f"executed_close_fraction={float(np.mean(executed_skill[:, 6] < 0)):.3f} "
                            f"mean_correction_norm="
                            f"{float(np.mean(np.linalg.norm(correction_skill[:, :6], axis=1))):.3f}",
                            flush=True,
                        )
                    for phase_name in ("HANDLE_CLOSE", "DRAWER_PULL"):
                        phase_mask = np.asarray(phase_trace) == phase_name
                        if not np.any(phase_mask):
                            continue
                        model_gripper = np.asarray(policy_actions)[phase_mask, 6]
                        print(
                            "  diagnostic_gripper "
                            f"phase={phase_name} steps={int(phase_mask.sum())} "
                            f"mean={float(np.mean(model_gripper)):.3f} "
                            f"median={float(np.median(model_gripper)):.3f} "
                            f"q10={float(np.quantile(model_gripper, 0.1)):.3f} "
                            f"q90={float(np.quantile(model_gripper, 0.9)):.3f} "
                            f"close_fraction={float(np.mean(model_gripper < 0)):.3f}",
                            flush=True,
                        )
        env.recorder_manager.close()
        if output.exists():
            add_metadata(
                output,
                attempts,
                successes,
                successful_phase_traces,
                successful_policy_payloads,
                successful_hero_frames,
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
