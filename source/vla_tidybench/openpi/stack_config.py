"""Locked pi0.5 LoRA configuration for the Franka stack smoke dataset."""

from __future__ import annotations

from openpi.models import pi0_config
from openpi.training import config as training_config
from openpi.training import weight_loaders


CONFIG_NAME = "pi05_tidybench_stack_lora"
REPO_ID = "erictianyufan/vla_tidybench_stack_m1_smoke"
# The provisioned download contains a truncated first attempt at ``params/``
# and a complete retry under ``pi05_droid/params/``. Keep the exact audited
# path explicit instead of allowing Orbax to read the shorter shard files.
CHECKPOINT_PARAMS = (
    "/home/ubuntu/data/vla-tidybench/checkpoints/openpi-assets/checkpoints/"
    "pi05_droid/pi05_droid/params"
)
ASSETS_BASE = "/home/ubuntu/data/vla-tidybench/checkpoints/openpi-assets/tidybench-assets"
CHECKPOINT_BASE = "/home/ubuntu/data/vla-tidybench/checkpoints/openpi-runs"


def make_config(
    *,
    exp_name: str = "smoke",
    num_train_steps: int = 2,
    batch_size: int = 1,
    fsdp_devices: int = 1,
    overwrite: bool = False,
    resume: bool = False,
) -> training_config.TrainConfig:
    """Return a config without modifying the external OpenPI checkout."""

    model = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    return training_config.TrainConfig(
        name=CONFIG_NAME,
        exp_name=exp_name,
        project_name="vla-tidybench",
        model=model,
        data=training_config.LeRobotLiberoDataConfig(
            repo_id=REPO_ID,
            base_config=training_config.DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(CHECKPOINT_PARAMS),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        assets_base_dir=ASSETS_BASE,
        checkpoint_base_dir=CHECKPOINT_BASE,
        batch_size=batch_size,
        fsdp_devices=fsdp_devices,
        num_workers=0,
        num_train_steps=num_train_steps,
        log_interval=1,
        save_interval=max(1, num_train_steps - 1),
        keep_period=None,
        wandb_enabled=False,
        overwrite=overwrite,
        resume=resume,
    )
