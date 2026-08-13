"""Finite-step smoke test for state or visuomotor Franka environments."""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=20)
parser.add_argument("--output", type=Path, default=Path("results/metrics/isaac_smoke.json"))
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]


def _shape_tree(value):
    if isinstance(value, dict):
        return {key: _shape_tree(item) for key, item in value.items()}
    return list(value.shape) if hasattr(value, "shape") else str(type(value).__name__)


def main() -> None:
    if args_cli.num_steps < 1:
        raise ValueError("num_steps must be positive")
    torch.manual_seed(42)
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device or "cuda:0"
        env = gym.make(args_cli.task, cfg=env_cfg)
        observation, _ = env.reset(seed=42)
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(args_cli.num_steps):
            with torch.inference_mode():
                observation, _, terminated, truncated, _ = env.step(actions)
            if bool(torch.any(terminated | truncated)):
                observation, _ = env.reset()
        report = {
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "num_steps": args_cli.num_steps,
            "device": str(env.unwrapped.device),
            "action_shape": list(env.action_space.shape),
            "observation_shapes": _shape_tree(observation),
            "status": "passed",
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("VLA_TIDYBENCH_SMOKE_PASSED")
        print(json.dumps(report, sort_keys=True))
        with contextlib.suppress(Exception):
            env.close()


if __name__ == "__main__":
    main()

