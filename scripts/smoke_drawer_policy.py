#!/usr/bin/env python3
"""Restore the trained drawer checkpoint and run one real offline inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from openpi.policies import policy_config
from openpi.shared import normalize
from vla_tidybench.openpi.drawer_config import make_config as make_open_config
from vla_tidybench.openpi.drawer_four_skill_config import make_config as make_four_skill_config


def synthetic_request(prompt: str) -> dict[str, object]:
    """Return a shape-correct request for checkpoint/inference plumbing checks."""

    return {
        "observation/state": np.zeros(18, dtype=np.float32),
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "prompt": prompt,
    }


def identity_norm_stats() -> dict[str, normalize.NormStats]:
    """Provide explicit model-width identity stats for dataset-free plumbing checks."""

    zeros = np.zeros(32, dtype=np.float32)
    ones = np.ones(32, dtype=np.float32)
    stats = normalize.NormStats(mean=zeros, std=ones, q01=-ones, q99=ones)
    return {"state": stats, "actions": stats}


def dataset_request(dataset: Path, prompt: str) -> dict[str, object]:
    with h5py.File(dataset, "r") as source:
        episode = source["data"][sorted(source["data"].keys())[0]]
        obs = episode["obs"]
        state = np.concatenate((obs["joint_pos"][0], obs["joint_vel"][0])).astype(np.float32)
        return {
            "observation/state": state,
            "observation/image": obs["table_cam"][0],
            "observation/wrist_image": obs["wrist_cam"][0],
            "prompt": prompt,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    request_source = parser.add_mutually_exclusive_group(required=True)
    request_source.add_argument("--dataset", type=Path)
    request_source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--prompt", default="open the top drawer")
    parser.add_argument("--four-skill", action="store_true")
    parser.add_argument("--mode", choices=("lora", "expert", "full"), default="expert")
    parser.add_argument("--runs", type=int, default=2, help="include at least one post-JIT timing sample")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    request = synthetic_request(args.prompt) if args.synthetic else dataset_request(args.dataset, args.prompt)
    make_config = make_four_skill_config if args.four_skill else make_open_config
    policy = policy_config.create_trained_policy(
        make_config(finetune_mode=args.mode),
        args.checkpoint,
        norm_stats=identity_norm_stats() if args.synthetic else None,
    )
    for run in range(1, args.runs + 1):
        response = policy.infer(request)
        actions = np.asarray(response["actions"])
        if actions.shape != (16, 7) or not np.isfinite(actions).all():
            raise ValueError(f"invalid policy actions: shape={actions.shape}, finite={np.isfinite(actions).all()}")
        print(
            f"checkpoint restore passed: run={run}/{args.runs}, mode={args.mode}, "
            f"four_skill={args.four_skill}, synthetic_identity_norm={args.synthetic}, "
            f"actions={actions.shape}, infer_ms={response['policy_timing']['infer_ms']:.1f}"
        )
    print("first_action", actions[0].tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
