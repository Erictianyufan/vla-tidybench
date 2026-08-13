#!/usr/bin/env python3
"""Load one real transformed batch through the locked pi0.5 data pipeline."""

from __future__ import annotations

import argparse

import numpy as np
from openpi.training import data_loader
from vla_tidybench.openpi.stack_config import make_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    config = make_config(batch_size=args.batch_size)
    loader = data_loader.create_data_loader(config, shuffle=False, num_batches=1)
    observation, actions = next(iter(loader))
    print("state", observation.state.shape, observation.state.dtype)
    print("actions", actions.shape, actions.dtype)
    print("tokenized_prompt", observation.tokenized_prompt.shape)
    print("images", {key: value.shape for key, value in observation.images.items()})
    if observation.state.shape != (args.batch_size, 32):
        raise ValueError(f"unexpected padded state shape: {observation.state.shape}")
    if actions.shape != (args.batch_size, 16, 32):
        raise ValueError(f"unexpected padded action shape: {actions.shape}")
    if not np.isfinite(np.asarray(observation.state)).all() or not np.isfinite(np.asarray(actions)).all():
        raise ValueError("transformed batch contains NaN or infinity")
    print("openpi transformed-batch smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
