#!/usr/bin/env python3
"""Restore the trained drawer checkpoint and run one real offline inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from openpi.policies import policy_config

from vla_tidybench.openpi.drawer_config import make_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prompt", default="open the top drawer")
    args = parser.parse_args()
    with h5py.File(args.dataset, "r") as dataset:
        episode = dataset["data"][sorted(dataset["data"].keys())[0]]
        obs = episode["obs"]
        state = np.concatenate((obs["joint_pos"][0], obs["joint_vel"][0])).astype(np.float32)
        request = {
            "observation/state": state,
            "observation/image": obs["table_cam"][0],
            "observation/wrist_image": obs["wrist_cam"][0],
            "prompt": args.prompt,
        }
    policy = policy_config.create_trained_policy(make_config(), args.checkpoint)
    response = policy.infer(request)
    actions = np.asarray(response["actions"])
    if actions.shape != (16, 7) or not np.isfinite(actions).all():
        raise ValueError(f"invalid policy actions: shape={actions.shape}, finite={np.isfinite(actions).all()}")
    print(f"checkpoint restore passed: actions={actions.shape}, infer_ms={response['policy_timing']['infer_ms']:.1f}")
    print("first_action", actions[0].tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
