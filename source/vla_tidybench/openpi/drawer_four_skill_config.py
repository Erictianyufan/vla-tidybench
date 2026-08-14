"""Locked pi0.5 LoRA configuration for the four-skill drawer MVP."""

from __future__ import annotations

from dataclasses import replace

from vla_tidybench.openpi.stack_config import make_config as make_stack_config

CONFIG_NAME = "pi05_tidybench_drawer_four_skill_lora"
REPO_ID = "erictianyufan/vla_tidybench_drawer_four_skill_mvp"


def make_config(**kwargs):
    """Reuse the audited pi0.5 architecture with an isolated multi-skill dataset."""

    config = make_stack_config(**kwargs)
    return replace(
        config,
        name=CONFIG_NAME,
        data=replace(config.data, repo_id=REPO_ID),
    )
