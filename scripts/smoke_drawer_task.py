"""Instantiate the project drawer task and validate its observation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("results/metrics/drawer_scene_smoke.json"))
parser.add_argument("--num_steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from vla_tidybench.isaac import TidyBenchDrawerEnvCfg  # noqa: E402


def main() -> int:
    cfg = TidyBenchDrawerEnvCfg()
    cfg.sim.device = args_cli.device
    env = gym.make("Isaac-Open-Drawer-Franka-IK-Rel-v0", cfg=cfg).unwrapped
    try:
        observations, _ = env.reset(seed=41)
        action = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
        action[:, 6] = 1.0
        for _ in range(args_cli.num_steps):
            observations, *_ = env.step(action)

        policy = observations["policy"]
        required = {"joint_pos", "joint_vel", "table_cam", "wrist_cam"}
        if set(policy) != required:
            raise RuntimeError(f"unexpected policy observations: {set(policy)}")
        if tuple(policy["table_cam"].shape) != (1, 200, 200, 3):
            raise RuntimeError(f"bad table camera shape: {policy['table_cam'].shape}")
        if tuple(policy["wrist_cam"].shape) != (1, 200, 200, 3):
            raise RuntimeError(f"bad wrist camera shape: {policy['wrist_cam'].shape}")

        cabinet = env.scene["cabinet"]
        target = env.scene["target_object"]
        drawer_joint = cabinet.find_joints("drawer_top_joint")[0][0]
        result = {
            "passed": True,
            "policy_dt_s": env.cfg.sim.dt * env.cfg.decimation,
            "policy_hz": 1.0 / (env.cfg.sim.dt * env.cfg.decimation),
            "observation_keys": sorted(policy),
            "table_cam_shape": list(policy["table_cam"].shape),
            "wrist_cam_shape": list(policy["wrist_cam"].shape),
            "joint_pos_shape": list(policy["joint_pos"].shape),
            "joint_vel_shape": list(policy["joint_vel"].shape),
            "drawer_joint_index": drawer_joint,
            "drawer_joint_pos": float(cabinet.data.joint_pos.torch[0, drawer_joint]),
            "target_pos_w": target.data.root_pos_w.torch[0].detach().cpu().tolist(),
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return 0
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
