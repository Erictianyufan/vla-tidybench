"""Portable pi0.5 fine-tuning configuration for the drawer dataset."""

from __future__ import annotations

import os
from dataclasses import replace

from vla_tidybench.openpi.stack_config import make_config as make_stack_config

CONFIG_NAMES = {
    "lora": "pi05_tidybench_drawer_lora",
    "expert": "pi05_tidybench_drawer_expert",
    "full": "pi05_tidybench_drawer_full",
}
CONFIG_NAME = CONFIG_NAMES["lora"]
REPO_ID = "erictianyufan/vla_tidybench_drawer_m2_smoke"


def make_config(*, finetune_mode: str = "lora", dataset_repo: str | None = None, **kwargs):
    """Reuse the audited π0.5 architecture/checkpoint with drawer data."""

    config = make_stack_config(finetune_mode=finetune_mode, **kwargs)
    data = config.data
    if data.repo_id != "fake":
        data = replace(
            data,
            repo_id=dataset_repo or os.environ.get("VLA_TIDYBENCH_DRAWER_REPO_ID", REPO_ID),
        )
    return replace(config, name=CONFIG_NAMES[finetune_mode], data=data)
