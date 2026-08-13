"""Print the runtime geometry contract of Isaac Lab's Franka cabinet task."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Open-Drawer-Franka-IK-Rel-v0")
parser.add_argument("--eef-only", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402, F401
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402


def _cpu(value):
    if hasattr(value, "torch"):
        value = value.torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def main() -> None:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    cfg.scene.num_envs = 1
    cfg.observations.policy.enable_corruption = False
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    try:
        env.reset(seed=17)
        zero_action = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
        zero_action[:, 6] = 1.0
        for _ in range(8):
            env.step(zero_action)

        robot = env.scene["robot"]
        print(f"robot_joint_names={robot.joint_names}")
        print(f"robot_body_names={robot.body_names}")
        print(f"robot_joint_pos={_cpu(robot.data.joint_pos)}")
        ee = env.scene["ee_frame"].data
        print(f"ee_target_names={ee.target_frame_names}")
        print(f"ee_target_pos_w={_cpu(ee.target_pos_w)}")
        print(f"ee_target_quat_w={_cpu(ee.target_quat_w)}")
        if not args_cli.eef_only:
            cabinet = env.scene["cabinet"]
            handle = env.scene["cabinet_frame"].data
            print(f"cabinet_joint_names={cabinet.joint_names}")
            print(f"cabinet_body_names={cabinet.body_names}")
            print(f"cabinet_joint_pos={_cpu(cabinet.data.joint_pos)}")
            print(f"cabinet_body_pos_w={_cpu(cabinet.data.body_pos_w)}")
            print(f"cabinet_target_names={handle.target_frame_names}")
            print(f"cabinet_target_pos_w={_cpu(handle.target_pos_w)}")
            print(f"cabinet_target_quat_w={_cpu(handle.target_quat_w)}")
            stage = Usd.Stage.GetCurrent()
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            for prim_path in (
                "/World/envs/env_0/Cabinet/drawer_top",
                "/World/envs/env_0/Cabinet/drawer_handle_top",
            ):
                prim = stage.GetPrimAtPath(prim_path)
                bbox_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                print(f"bbox={prim_path}:{bbox_range.GetMin()}:{bbox_range.GetMax()}")
        print(f"env_origin={_cpu(env.scene.env_origins)}")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
