#!/usr/bin/env python3
"""One-step synthetic capacity test for tuning from Stage3 params.

This is not task training or a validated model. Output is isolated from the
source checkpoint so a failed 12-GB attempt cannot alter its provenance.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from dataclasses import replace
from pathlib import Path

from openpi.training import weight_loaders
from vla_tidybench.openpi.drawer_four_skill_config import make_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-params", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--mode", choices=("expert", "lora"), default="expert")
    args = parser.parse_args()
    if not args.init_params.is_dir():
        parser.error(f"missing params: {args.init_params}")
    if not args.output_root.is_dir():
        parser.error(f"missing output root: {args.output_root}")

    config = make_config(
        finetune_mode=args.mode,
        synthetic_data=True,
        exp_name=args.exp_name,
        num_train_steps=1,
        batch_size=1,
        fsdp_devices=1,
        save_interval=1,
    )
    config = replace(
        config,
        checkpoint_base_dir=str(args.output_root),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(args.init_params)),
    )
    openpi_root = Path(os.environ.get("OPENPI_ROOT", "/home/ubuntu/openpi"))
    train_script = openpi_root / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("openpi_train", train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {train_script}")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    official.main(config)


if __name__ == "__main__":
    main()
