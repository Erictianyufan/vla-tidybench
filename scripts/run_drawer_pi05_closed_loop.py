"""Run the real pi0.5 drawer policy in Isaac Lab and record deployable views.

The policy sees only table RGB, wrist RGB, proprioception, and language.  The
drawer joint is read only for the success metric and is never sent to pi0.5.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--prompt", default="open the top drawer")
parser.add_argument("--max-steps", type=int, default=360)
parser.add_argument("--execute-steps", type=int, default=4)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--showcase", action="store_true", help="add room, props, and 720p hero camera")
parser.add_argument("--teacher-preview", action="store_true", help="use the scripted OPEN teacher for camera QA")
parser.add_argument("--teacher-skill", choices=("open", "pick", "place", "close"), default="open")
parser.add_argument(
    "--dls-contact-recovery",
    action="store_true",
    help="run pi0.5 every step as a bounded residual over the replay-validated DLS OPEN prior",
)
parser.add_argument("--policy-residual-weight", type=float, default=0.02)
parser.add_argument(
    "--recovery-demo",
    type=Path,
    default=Path("/home/ubuntu/data/vla-tidybench/raw/drawer_open_smoke.hdf5"),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.dls_contact_recovery and args_cli.teacher_preview:
    parser.error("--dls-contact-recovery and --teacher-preview are mutually exclusive")
if args_cli.dls_contact_recovery and args_cli.execute_steps != 1:
    parser.error("DLS contact recovery requires --execute-steps 1")
if not 0.0 <= args_cli.policy_residual_weight <= 0.1:
    parser.error("--policy-residual-weight must be in [0, 0.1]")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from vla_tidybench.isaac import TidyBenchDrawerEnvCfg, TidyBenchDrawerShowcaseEnvCfg  # noqa: E402
from vla_tidybench.policy_bridge.action_adapter import ActionAdapter  # noqa: E402
from vla_tidybench.policy_bridge.safety_guard import SafetyGuard  # noqa: E402
from vla_tidybench.policy_bridge.websocket_client import PolicyClient  # noqa: E402


def _numpy(value):
    if hasattr(value, "torch"):
        value = value.torch
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _obs(env):
    data = env.scene
    q = _numpy(data["robot"].data.joint_pos)[0].astype(np.float32)
    qd = _numpy(data["robot"].data.joint_vel)[0].astype(np.float32)
    table = _numpy(data["table_cam"].data.output["rgb"])[0, ..., :3].astype(np.uint8)
    wrist = _numpy(data["wrist_cam"].data.output["rgb"])[0, ..., :3].astype(np.uint8)
    return table, wrist, np.concatenate((q, qd), dtype=np.float32)


def main() -> int:
    cfg = TidyBenchDrawerShowcaseEnvCfg() if args_cli.showcase else TidyBenchDrawerEnvCfg()
    cfg.sim.device = args_cli.device
    cfg.scene.num_envs = 1
    if args_cli.teacher_preview and args_cli.teacher_skill in ("place", "close"):
        cfg.scene.cabinet.init_state.joint_pos["drawer_top_joint"] = 0.36
    if args_cli.teacher_preview and args_cli.teacher_skill == "place":
        cfg.scene.cabinet.actuators["drawers"].stiffness = 10000.0
        cfg.scene.cabinet.actuators["drawers"].damping = 500.0
    env = gym.make("Isaac-Open-Drawer-Franka-IK-Rel-v0", cfg=cfg).unwrapped
    drawer_idx = env.scene["cabinet"].find_joints("drawer_top_joint")[0][0]
    adapter = ActionAdapter()
    guard = SafetyGuard(adapter=adapter)
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    frames_table: list[np.ndarray] = []
    frames_wrist: list[np.ndarray] = []
    frames_hero: list[np.ndarray] = []
    actions_executed: list[np.ndarray] = []
    policy_actions_proposed: list[np.ndarray] = []
    recovery_actions_base: list[np.ndarray] = []
    latencies: list[float] = []
    success = False
    try:
        env.reset(seed=args_cli.seed)
        if args_cli.showcase:
            # Explicit look-at avoids reusing a near-camera quaternion after
            # translating the hero camera. Target sits between the complete
            # Franka workspace and the cabinet, with generous frame margin.
            env.scene["hero_cam"].set_world_poses_from_view(
                eyes=torch.tensor([[-1.85, -2.35, 1.65]], device=env.device),
                targets=torch.tensor([[0.50, 0.0, 0.58]], device=env.device),
            )
            for _ in range(3):
                env.sim.render()
        client_context = None if args_cli.teacher_preview else PolicyClient(args_cli.host, args_cli.port, timeout_s=120.0)
        if client_context is not None:
            print("policy metadata", json.dumps(client_context.metadata, indent=2, default=str), flush=True)
        teacher_actions = None
        teacher_index = 0
        if args_cli.teacher_preview:
            demo_path = f"/home/ubuntu/data/vla-tidybench/raw/drawer_{args_cli.teacher_skill}_smoke.hdf5"
            with h5py.File(demo_path, "r") as demo_file:
                demo = demo_file["data"][sorted(demo_file["data"].keys())[0]]
                teacher_actions = np.asarray(demo["actions"], dtype=np.float32)
        recovery_actions = None
        if args_cli.dls_contact_recovery:
            with h5py.File(args_cli.recovery_demo, "r") as recovery_file:
                demo = recovery_file["data"][sorted(recovery_file["data"].keys())[0]]
                recovery_actions = np.asarray(demo["actions"], dtype=np.float32)
        try:
            step = 0
            while step < args_cli.max_steps and simulation_app.is_running():
                table, wrist, state = _obs(env)
                if teacher_actions is not None:
                    if teacher_index >= len(teacher_actions):
                        break
                    chunk = teacher_actions[teacher_index : teacher_index + 1]
                    teacher_index += 1
                    latencies.append(0.0)
                else:
                    request = {
                        "observation/image": table,
                        "observation/wrist_image": wrist,
                        "observation/state": state,
                        "prompt": args_cli.prompt,
                    }
                    started = time.perf_counter()
                    response = client_context.infer(request)
                    latencies.append((time.perf_counter() - started) * 1000.0)
                    chunk = np.asarray(response["actions"], dtype=np.float32)
                if chunk.ndim != 2 or chunk.shape[1] != 7 or not np.isfinite(chunk).all():
                    raise ValueError(f"malformed action chunk {chunk.shape}")
                for physical in chunk[: args_cli.execute_steps]:
                    proposed = physical.copy()
                    executed = physical.copy()
                    if recovery_actions is not None:
                        if step >= len(recovery_actions):
                            break
                        base_physical = adapter.from_isaac(recovery_actions[step])
                        executed = base_physical.copy()
                        executed[:6] += args_cli.policy_residual_weight * proposed[:6]
                        executed[6] = base_physical[6]
                        executed = guard.apply(executed)
                        raw = adapter.to_isaac(executed)
                        policy_actions_proposed.append(proposed)
                        recovery_actions_base.append(base_physical)
                    else:
                        raw = physical if teacher_actions is not None else guard.to_isaac(physical)
                    env.step(torch.as_tensor(raw[None], dtype=torch.float32, device=env.device))
                    table, wrist, _ = _obs(env)
                    frames_table.append(table)
                    frames_wrist.append(wrist)
                    if args_cli.showcase:
                        hero = _numpy(env.scene["hero_cam"].data.output["rgb"])[0, ..., :3].astype(np.uint8)
                        frames_hero.append(hero)
                    actions_executed.append(executed)
                    step += 1
                    drawer = float(env.scene["cabinet"].data.joint_pos.torch[0, drawer_idx])
                    obj = env.scene["target_object"].data.root_pos_w.torch[0]
                    if args_cli.teacher_skill == "open":
                        success = drawer >= 0.30
                    elif args_cli.teacher_skill == "pick":
                        success = float(obj[2]) >= 0.12
                    elif args_cli.teacher_skill == "place":
                        handle_x = env.scene["cabinet_frame"].data.target_pos_w.torch[0, 0, 0]
                        success = bool(obj[2] > 0.68 and obj[2] < 0.86 and obj[0] > handle_x + 0.023 and abs(obj[1]) < 0.26)
                    elif args_cli.teacher_skill == "close":
                        teacher_length = 0 if teacher_actions is None else len(teacher_actions)
                        success = drawer <= 0.04 and step >= min(40, teacher_length)
                        break
                print(
                    f"step={step} drawer={drawer:.3f} infer_ms={latencies[-1]:.1f} success={success}",
                    flush=True,
                )
                if success:
                    break
        finally:
            if client_context is not None:
                client_context.close()
        with h5py.File(args_cli.output, "w") as output:
            output.attrs["format_version"] = 1
            if teacher_actions is not None:
                policy_name = "scripted-teacher-camera-preview"
            elif recovery_actions is not None:
                policy_name = "pi0.5-drawer-lora+dls-contact-recovery"
            else:
                policy_name = "pi0.5-drawer-lora"
            output.attrs["policy"] = policy_name
            output.attrs["prompt"] = args_cli.prompt
            output.attrs["skill"] = args_cli.teacher_skill if teacher_actions is not None else "open"
            output.attrs["success"] = success
            output.attrs["mean_infer_ms"] = float(np.mean(latencies)) if latencies else -1.0
            output.attrs["policy_residual_weight"] = (
                args_cli.policy_residual_weight if recovery_actions is not None else 0.0
            )
            output.create_dataset("table_cam", data=np.asarray(frames_table), compression="gzip")
            output.create_dataset("wrist_cam", data=np.asarray(frames_wrist), compression="gzip")
            if frames_hero:
                output.create_dataset("hero_cam", data=np.asarray(frames_hero), compression="gzip")
            output.create_dataset("actions", data=np.asarray(actions_executed, dtype=np.float32))
            if policy_actions_proposed:
                output.create_dataset(
                    "policy_actions", data=np.asarray(policy_actions_proposed, dtype=np.float32)
                )
                output.create_dataset(
                    "recovery_base_actions", data=np.asarray(recovery_actions_base, dtype=np.float32)
                )
        print(
            json.dumps(
                {
                    "policy": policy_name,
                    "success": success,
                    "steps": len(actions_executed),
                    "mean_infer_ms": round(float(np.mean(latencies)), 1) if latencies else None,
                    "output": str(args_cli.output),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0 if success else 1
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
