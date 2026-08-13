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


def succeeded(env, skill: str) -> tuple[bool, str]:
    cabinet = env.scene["cabinet"]
    drawer_idx = cabinet.find_joints("drawer_top_joint")[0][0]
    drawer_q = float(cabinet.data.joint_pos.torch[0, drawer_idx])
    obj = env.scene["target_object"].data.root_pos_w.torch[0]
    handle = env.scene["cabinet_frame"].data.target_pos_w.torch[0, 0]
    object_in_drawer = bool(obj[2] > 0.68 and obj[2] < 0.86 and obj[0] > handle[0] + 0.023 and abs(obj[1]) < 0.26)
    picked = float(obj[2]) >= 0.12
    checks = {
        "open": drawer_q >= 0.30,
        "pick": picked,
        "place": object_in_drawer,
        "close": drawer_q <= 0.04,
        "full": drawer_q <= 0.04 and object_in_drawer,
    }
    detail = f"drawer={drawer_q:.3f}, object={obj.detach().cpu().tolist()}, in_drawer={object_in_drawer}"
    return checks[skill], detail


def main() -> int:
    handler = HDF5DatasetFileHandler()
    handler.open(str(args_cli.dataset_file.resolve()))
    episode_names = list(handler.get_episode_names())
    if not episode_names:
        raise RuntimeError("dataset contains no episodes")

    env = gym.make("Isaac-Open-Drawer-Franka-IK-Rel-v0", cfg=make_cfg()).unwrapped
    passed = 0
    try:
        env.reset()
        with torch.inference_mode():
            for index, name in enumerate(episode_names):
                episode = handler.load_episode(name, env.device)
                if args_cli.reset_sim_buffer_each_episode:
                    env.sim.reset()
                env.reset_to(episode.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)
                steps = 0
                while (action := episode.get_next_action()) is not None:
                    env.step(action.unsqueeze(0) if action.ndim == 1 else action)
                    steps += 1
                ok, detail = succeeded(env, args_cli.skill)
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
