"""Replay and validate TidyBench drawer demonstrations in the exact custom scene."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset_file", type=Path, required=True)
parser.add_argument("--skill", choices=("open", "pick", "place", "close", "full"), required=True)
parser.add_argument("--reset_sim_buffer_each_episode", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.dataset_file.is_file():
    parser.error(f"dataset not found: {args_cli.dataset_file}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.datasets import HDF5DatasetFileHandler  # noqa: E402
from vla_tidybench.isaac import TidyBenchDrawerEnvCfg  # noqa: E402
from vla_tidybench.task_metrics import drawer_skill_success  # noqa: E402


def make_cfg() -> TidyBenchDrawerEnvCfg:
    cfg = TidyBenchDrawerEnvCfg()
    cfg.sim.device = args_cli.device
    if args_cli.skill in ("place", "close"):
        cfg.scene.cabinet.init_state.joint_pos["drawer_top_joint"] = 0.36
    if args_cli.skill == "place":
        cfg.scene.cabinet.actuators["drawers"].stiffness = 10000.0
        cfg.scene.cabinet.actuators["drawers"].damping = 500.0
    cfg.recorders = {}
    cfg.terminations = {}
    cfg.rewards = None
    return cfg


def succeeded(env, skill: str, initial_drawer: float, initial_object) -> tuple[bool, str]:
    cabinet = env.scene["cabinet"]
    drawer_idx = cabinet.find_joints("drawer_top_joint")[0][0]
    drawer_q = float(cabinet.data.joint_pos.torch[0, drawer_idx])
    obj = env.scene["target_object"].data.root_pos_w.torch[0]
    handle = env.scene["cabinet_frame"].data.target_pos_w.torch[0, 0]
    fingers = env.scene["robot"].data.joint_pos.torch[0, -2:]
    ok = drawer_skill_success(
        skill,
        initial_drawer_m=initial_drawer,
        initial_object_xyz=initial_object.detach().cpu().numpy(),
        drawer_m=drawer_q,
        object_xyz=obj.detach().cpu().numpy(),
        handle_x=float(handle[0]),
        finger_positions=fingers.detach().cpu().numpy(),
    )
    detail = (
        f"drawer={drawer_q:.3f}, object={obj.detach().cpu().tolist()}, "
        f"gripper_width={float(fingers.sum()):.3f}"
    )
    return ok, detail


def main() -> int:
    handler = HDF5DatasetFileHandler()
    handler.open(str(args_cli.dataset_file.resolve()))
    episode_names = list(handler.get_episode_names())
    if not episode_names:
        raise RuntimeError("dataset contains no episodes")

    env = gym.make("Isaac-Open-Drawer-Franka-IK-Rel-v0", cfg=make_cfg()).unwrapped
    drawer_idx = env.scene["cabinet"].find_joints("drawer_top_joint")[0][0]
    passed = 0
    try:
        env.reset()
        with torch.inference_mode():
            for index, name in enumerate(episode_names):
                episode = handler.load_episode(name, env.device)
                if args_cli.reset_sim_buffer_each_episode:
                    env.sim.reset()
                env.reset_to(episode.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)
                initial_drawer = float(env.scene["cabinet"].data.joint_pos.torch[0, drawer_idx])
                initial_object = env.scene["target_object"].data.root_pos_w.torch[0, :3].clone()
                steps = 0
                while (action := episode.get_next_action()) is not None:
                    env.step(action.unsqueeze(0) if action.ndim == 1 else action)
                    steps += 1
                ok, detail = succeeded(env, args_cli.skill, initial_drawer, initial_object)
                passed += int(ok)
                print(f"episode={index} name={name} steps={steps} success={ok} {detail}", flush=True)
    finally:
        env.close()
        simulation_app.close()
        handler.close()

    print(f"REPLAY_RESULT passed={passed}/{len(episode_names)}", flush=True)
    return 0 if passed == len(episode_names) else 1


if __name__ == "__main__":
    raise SystemExit(main())
