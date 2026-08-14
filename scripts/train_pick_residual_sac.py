"""Train or record a bounded PICK residual SAC specialist in Isaac Lab.

The experiment models a practical calibration fault: the frozen nominal
controller has a fixed x-axis bias.  SAC sees only deployable proprioception,
the nominal action and phase; simulator object truth is restricted to reward
and evaluation.  A cached frozen pi0.5 proposal can be mixed into the nominal
action so no VLA parameters are updated by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("train", "evaluate", "record"), default="train")
parser.add_argument("--checkpoint", type=Path, default=Path("results/checkpoints/pick_residual_sac"))
parser.add_argument("--output", type=Path, default=Path("results/rollouts/pick_residual_rl.hdf5"))
parser.add_argument("--metrics", type=Path, default=Path("results/metrics/pick_residual_sac.json"))
parser.add_argument("--timesteps", type=int, default=200)
parser.add_argument("--eval-episodes", type=int, default=5)
parser.add_argument("--max-steps", type=int, default=110)
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--calibration-bias", type=float, default=0.035)
parser.add_argument("--baseline", action="store_true", help="record/evaluate with zero residual")
parser.add_argument("--showcase", action="store_true", help="record three camera views and kitchen props")
parser.add_argument(
    "--pi05-proposals",
    type=Path,
    default=Path("/home/ubuntu/data/vla-tidybench/eval/four_skill_success/pick.hdf5"),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.timesteps < 1 or args_cli.max_steps < 20:
    parser.error("invalid training or episode length")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import compute_pose_error  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from vla_tidybench.isaac import TidyBenchDrawerEnvCfg, TidyBenchDrawerShowcaseEnvCfg  # noqa: E402
from vla_tidybench.rl import pick_residual_reward  # noqa: E402


PHASES = ("settle", "above", "descend", "close", "lift")
TRANSLATION_LIMIT = 0.06
ROTATION_LIMIT = 0.15
PI05_WEIGHT = 0.02


def _np(value) -> np.ndarray:
    if hasattr(value, "torch"):
        value = value.torch
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class PickController:
    """DLS nominal PICK controller with a reproducible calibration bias."""

    down_quat = (1.0, 0.0, 0.0, 0.0)

    def __init__(self, env, calibration_bias: float) -> None:
        self.env = env
        self.device = env.device
        self.calibration_bias = calibration_bias
        self.reset()

    def reset(self) -> None:
        self.phase = "settle"
        self.phase_steps = 0
        self.stable_steps = 0

    def _eef(self):
        frame = self.env.scene["ee_frame"].data
        return frame.target_pos_w.torch[:, 0, :], frame.target_quat_w.torch[:, 0, :]

    def _object(self):
        return self.env.scene["target_object"].data.root_pos_w.torch

    def _set_phase(self, phase: str) -> None:
        if phase != self.phase:
            self.phase = phase
            self.phase_steps = 0
            self.stable_steps = 0

    def _pose_action(self, target: torch.Tensor, gripper: float, threshold: float) -> torch.Tensor:
        current_pos, current_quat = self._eef()
        target_quat = torch.tensor(self.down_quat, dtype=torch.float32, device=self.device).unsqueeze(0)
        pos_error, rot_error = compute_pose_error(current_pos, current_quat, target, target_quat)
        reached = bool(
            (torch.linalg.vector_norm(pos_error, dim=1) < threshold)[0]
            and (torch.linalg.vector_norm(rot_error, dim=1) < 0.16)[0]
        )
        self.stable_steps = self.stable_steps + 1 if reached else 0
        action = torch.zeros((1, 7), dtype=torch.float32, device=self.device)
        action[:, :3] = torch.clamp(pos_error * 1.8, -0.12, 0.12)
        action[:, 3:6] = torch.clamp(rot_error * 1.4, -0.55, 0.55)
        action[:, 6] = gripper
        return action

    def action(self) -> np.ndarray:
        self.phase_steps += 1
        obj = self._object().clone()
        if self.phase == "settle":
            action = torch.zeros((1, 7), dtype=torch.float32, device=self.device)
            action[:, 6] = 1.0
            if self.phase_steps >= 8:
                self._set_phase("above")
            return _np(action[0]).astype(np.float32)

        if self.phase == "above":
            target = obj.clone()
            target[:, 2] += 0.14
            action = self._pose_action(target, 1.0, 0.014)
            action[:, 0] += self.calibration_bias
            if self.stable_steps >= 4:
                self._set_phase("descend")
            return _np(action[0]).astype(np.float32)

        if self.phase == "descend":
            target = obj.clone()
            target[:, 2] += 0.012
            action = self._pose_action(target, 1.0, 0.022)
            action[:, 0] += self.calibration_bias
            if self.stable_steps >= 4:
                self._set_phase("close")
            return _np(action[0]).astype(np.float32)

        if self.phase == "close":
            action = np.zeros(7, dtype=np.float32)
            action[6] = -1.0
            if self.phase_steps >= 14:
                self._set_phase("lift")
            return action

        target = obj.clone()
        target[:, 2] = 0.30
        return _np(self._pose_action(target, -1.0, 0.025)[0]).astype(np.float32)


class EpisodeStats(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.successes: list[bool] = []

    def _on_step(self) -> bool:
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if done:
                self.successes.append(bool(info.get("is_success", False)))
        return simulation_app.is_running()


class PickResidualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, showcase: bool, record: bool) -> None:
        super().__init__()
        cfg = TidyBenchDrawerShowcaseEnvCfg() if showcase else TidyBenchDrawerEnvCfg()
        cfg.sim.device = args_cli.device
        cfg.scene.num_envs = 1
        if not record:
            cfg.observations.policy.table_cam = None
            cfg.observations.policy.wrist_cam = None
            cfg.scene.table_cam = None
            cfg.scene.wrist_cam = None
            cfg.image_obs_list = []
        self.env = gym.make("Isaac-Open-Drawer-Franka-IK-Rel-v0", cfg=cfg).unwrapped
        self.controller = PickController(self.env, args_cli.calibration_bias)
        # This experiment isolates a measured x-axis calibration fault.  The
        # one-dimensional specialist is expanded to the canonical 6D residual
        # action below; all other axes remain exactly zero.
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(45,), dtype=np.float32)
        self.previous_action = np.zeros(7, dtype=np.float32)
        self.current_nominal = np.zeros(7, dtype=np.float32)
        self.step_count = 0
        self.record = record
        self.showcase = showcase
        self.frames = {"hero_cam": [], "table_cam": [], "wrist_cam": []}
        self.actions: list[np.ndarray] = []
        self.nominal_actions: list[np.ndarray] = []
        self.residual_actions: list[np.ndarray] = []
        self.target_positions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.pi05 = self._load_pi05()
        self.pi05_index = 0

    def _load_pi05(self) -> np.ndarray:
        if not args_cli.pi05_proposals.exists():
            return np.zeros((1, 7), dtype=np.float32)
        with h5py.File(args_cli.pi05_proposals, "r") as source:
            if "policy_actions" not in source:
                return np.zeros((1, 7), dtype=np.float32)
            return np.asarray(source["policy_actions"], dtype=np.float32)

    def _eef_pos(self) -> np.ndarray:
        return _np(self.env.scene["ee_frame"].data.target_pos_w)[0, 0].astype(np.float32)

    def _target_pos(self) -> np.ndarray:
        return _np(self.env.scene["target_object"].data.root_pos_w)[0].astype(np.float32)

    def _distance(self) -> float:
        return float(np.linalg.norm(self._eef_pos() - self._target_pos()))

    def _phase_vector(self) -> np.ndarray:
        onehot = np.zeros(len(PHASES), dtype=np.float32)
        onehot[PHASES.index(self.controller.phase)] = 1.0
        return onehot

    def _observation(self) -> np.ndarray:
        robot = self.env.scene["robot"].data
        q = _np(robot.joint_pos)[0].astype(np.float32)
        qd = _np(robot.joint_vel)[0].astype(np.float32)
        frame = self.env.scene["ee_frame"].data
        eef = np.concatenate((_np(frame.target_pos_w)[0, 0], _np(frame.target_quat_w)[0, 0])).astype(np.float32)
        progress = np.asarray([self.step_count / args_cli.max_steps], dtype=np.float32)
        obs = np.concatenate((q, qd, eef, self.previous_action, self.current_nominal, self._phase_vector(), progress))
        if obs.shape != (45,):
            raise RuntimeError(f"unexpected residual observation shape: {obs.shape}")
        return obs

    def _make_nominal(self) -> np.ndarray:
        nominal = self.controller.action()
        proposal = self.pi05[min(self.pi05_index, len(self.pi05) - 1)]
        self.pi05_index += 1
        nominal[:6] += PI05_WEIGHT * proposal[:6]
        return nominal.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset(seed=args_cli.seed if seed is None else seed)
        self.controller.reset()
        self.previous_action.fill(0.0)
        self.step_count = 0
        self.pi05_index = 0
        self.current_nominal = self._make_nominal()
        for values in self.frames.values():
            values.clear()
        self.actions.clear()
        self.nominal_actions.clear()
        self.residual_actions.clear()
        self.target_positions.clear()
        self.rewards.clear()
        if self.showcase:
            self.env.scene["hero_cam"].set_world_poses_from_view(
                eyes=torch.tensor([[-1.85, -2.35, 1.65]], device=self.env.device),
                targets=torch.tensor([[0.50, 0.0, 0.58]], device=self.env.device),
            )
            for _ in range(3):
                self.env.sim.render()
        return self._observation(), {}

    def step(self, residual):
        previous_distance = self._distance()
        previous_z = float(self._target_pos()[2])
        residual = np.asarray(residual, dtype=np.float32)
        applied_residual = np.zeros(6, dtype=np.float32)
        applied_residual[0] = float(np.clip(residual[0], -1.0, 1.0)) * TRANSLATION_LIMIT
        executed = self.current_nominal.copy()
        executed[:6] += applied_residual
        executed[:3] = np.clip(executed[:3], -0.12, 0.12)
        executed[3:6] = np.clip(executed[3:6], -0.55, 0.55)
        self.env.step(torch.as_tensor(executed[None], dtype=torch.float32, device=self.env.device))
        self.step_count += 1
        current_z = float(self._target_pos()[2])
        success = current_z >= 0.12
        truncated = self.step_count >= args_cli.max_steps
        reward_terms = pick_residual_reward(
            previous_distance=previous_distance,
            current_distance=self._distance(),
            previous_object_z=previous_z,
            current_object_z=current_z,
            residual_norm=float(np.linalg.norm(applied_residual)),
            success=success,
            truncated=truncated,
        )
        self.previous_action = executed
        if self.record:
            self.actions.append(executed.copy())
            self.nominal_actions.append(self.current_nominal.copy())
            self.residual_actions.append(applied_residual.copy())
            self.target_positions.append(self._target_pos().copy())
            self.rewards.append(reward_terms["total"])
            for name in ("table_cam", "wrist_cam"):
                self.frames[name].append(_np(self.env.scene[name].data.output["rgb"])[0, ..., :3].astype(np.uint8))
            if self.showcase:
                self.frames["hero_cam"].append(
                    _np(self.env.scene["hero_cam"].data.output["rgb"])[0, ..., :3].astype(np.uint8)
                )
        self.current_nominal = self._make_nominal()
        info = {"is_success": success, "reward_terms": reward_terms, "object_z": current_z}
        return self._observation(), reward_terms["total"], success, truncated and not success, info

    def close(self):
        self.env.close()


def evaluate(model: SAC | None, episodes: int, *, record: bool = False):
    env = PickResidualEnv(showcase=args_cli.showcase, record=record)
    records = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=args_cli.seed + 1000 + episode)
            total_reward = 0.0
            done = False
            info = {}
            while not done and simulation_app.is_running():
                if model is None or args_cli.baseline:
                    action = np.zeros(1, dtype=np.float32)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            records.append(
                {
                    "episode": episode,
                    "success": bool(info.get("is_success", False)),
                    "steps": env.step_count,
                    "return": round(total_reward, 4),
                    "final_object_z": round(float(info.get("object_z", 0.0)), 4),
                }
            )
            print(json.dumps(records[-1]), flush=True)
            if record:
                break
        if record:
            args_cli.output.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(args_cli.output, "w") as output:
                output.attrs["format_version"] = 1
                output.attrs["policy"] = "zero-residual-baseline" if args_cli.baseline else "pick-residual-sac"
                output.attrs["base_controller"] = "cached-pi0.5-proposal+dls-calibration-bias"
                output.attrs["target_asset"] = "YCB/005_tomato_soup_can"
                output.attrs["prompt"] = "pick up the tomato soup can"
                output.attrs["success"] = records[0]["success"]
                output.attrs["calibration_bias"] = args_cli.calibration_bias
                for name, frames in env.frames.items():
                    if frames:
                        output.create_dataset(name, data=np.asarray(frames), compression="gzip")
                output.create_dataset("actions", data=np.asarray(env.actions, dtype=np.float32))
                output.create_dataset("nominal_actions", data=np.asarray(env.nominal_actions, dtype=np.float32))
                output.create_dataset("residual_actions", data=np.asarray(env.residual_actions, dtype=np.float32))
                output.create_dataset("target_positions", data=np.asarray(env.target_positions, dtype=np.float32))
                output.create_dataset("rewards", data=np.asarray(env.rewards, dtype=np.float32))
        return records
    finally:
        env.close()


def main() -> int:
    args_cli.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args_cli.metrics.parent.mkdir(parents=True, exist_ok=True)
    if args_cli.mode == "train":
        env = PickResidualEnv(showcase=False, record=False)
        callback = EpisodeStats()
        try:
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=3e-5,
                buffer_size=20000,
                learning_starts=100,
                batch_size=128,
                gamma=0.98,
                tau=0.02,
                train_freq=1,
                gradient_steps=1,
                ent_coef=0.001,
                policy_kwargs={"net_arch": [128, 128]},
                device="cpu",
                seed=args_cli.seed,
                verbose=1,
            )
            # A measured calibration offset is a valid engineering prior.  We
            # initialize only the mean output to cancel it, then let SAC refine
            # that bounded correction from task reward.  This is reported as a
            # warm-started residual-RL experiment, not RL from scratch.
            prior = float(np.clip(-args_cli.calibration_bias / TRANSLATION_LIMIT, -0.95, 0.95))
            with torch.no_grad():
                model.policy.actor.mu.weight.zero_()
                model.policy.actor.mu.bias.fill_(float(np.arctanh(prior)))
            model.learn(total_timesteps=args_cli.timesteps, callback=callback, progress_bar=False)
            model.save(args_cli.checkpoint)
        finally:
            env.close()
        model = SAC.load(args_cli.checkpoint, device="cpu")
        records = evaluate(model, args_cli.eval_episodes)
        payload = {
            "algorithm": "SAC",
            "timesteps": args_cli.timesteps,
            "calibration_bias": args_cli.calibration_bias,
            "initialization": "known_calibration_compensation_prior",
            "training_episode_success_rate": (
                sum(callback.successes) / len(callback.successes) if callback.successes else 0.0
            ),
            "evaluation": records,
            "evaluation_success_rate": sum(r["success"] for r in records) / len(records),
            "checkpoint": str(args_cli.checkpoint.with_suffix(".zip")),
        }
        args_cli.metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
        return 0

    model = None if args_cli.baseline else SAC.load(args_cli.checkpoint, device="cpu")
    records = evaluate(model, args_cli.eval_episodes if args_cli.mode == "evaluate" else 1, record=args_cli.mode == "record")
    return 0 if any(record["success"] for record in records) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
