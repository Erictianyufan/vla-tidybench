"""Portable pi0.5 fine-tuning configuration for the four-skill drawer task."""

from __future__ import annotations

import os
from dataclasses import replace

from vla_tidybench.openpi.stack_config import make_config as make_stack_config

CONFIG_NAMES = {
    "lora": "pi05_tidybench_drawer_four_skill_lora",
    "expert": "pi05_tidybench_drawer_four_skill_expert",
    "full": "pi05_tidybench_drawer_four_skill_full",
}
CONFIG_NAME = CONFIG_NAMES["lora"]
REPO_ID = "erictianyufan/vla_tidybench_drawer_four_skill_mvp"


def make_config(*, finetune_mode: str = "lora", dataset_repo: str | None = None, **kwargs):
    """Reuse the audited pi0.5 architecture with an isolated multi-skill dataset."""

    config = make_stack_config(finetune_mode=finetune_mode, **kwargs)
    data = config.data
    if data.repo_id != "fake":
        data = replace(
            data,
            repo_id=dataset_repo
            or os.environ.get("VLA_TIDYBENCH_DRAWER_FOUR_SKILL_REPO_ID", REPO_ID),
        )
    return replace(config, name=CONFIG_NAMES[finetune_mode], data=data)
