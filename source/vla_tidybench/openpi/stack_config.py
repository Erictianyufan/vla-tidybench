"""Portable pi0.5 fine-tuning configuration for VLA-TidyBench."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import optax
from flax import nnx
from openpi.models import pi0_config
from openpi.shared import nnx_utils
from openpi.training import config as training_config
from openpi.training import optimizer as training_optimizer
from openpi.training import weight_loaders

CONFIG_NAMES = {
    "lora": "pi05_tidybench_stack_lora",
    "expert": "pi05_tidybench_stack_expert",
    "full": "pi05_tidybench_stack_full",
}
CONFIG_NAME = CONFIG_NAMES["lora"]
REPO_ID = "erictianyufan/vla_tidybench_stack_m1_smoke"


@dataclass(frozen=True)
class MemoryEfficientAdafactor:
    """Factored optimizer state for true full-model tuning on 24-GB GPUs."""

    min_dim_size_to_factor: int = 128
    decay_rate: float = 0.8
    clipping_threshold: float = 1.0
    weight_decay: float = 1e-10

    def create(self, lr, weight_decay_mask=None):
        return optax.adafactor(
            learning_rate=lr,
            min_dim_size_to_factor=self.min_dim_size_to_factor,
            decay_rate=self.decay_rate,
            multiply_by_parameter_scale=False,
            clipping_threshold=self.clipping_threshold,
            momentum=None,
            weight_decay_rate=self.weight_decay,
            weight_decay_mask=weight_decay_mask,
        )


def data_root() -> Path:
    """Return the large-artifact root, which may live on a separate disk."""

    return Path(os.environ.get("VLA_TIDYBENCH_DATA", Path.home() / "data" / "vla-tidybench")).expanduser()


def checkpoint_params() -> Path:
    """Return the complete pi0.5-DROID Orbax parameter directory."""

    default = data_root() / "checkpoints" / "openpi-assets" / "checkpoints" / "pi05_droid" / "params"
    return Path(os.environ.get("PI05_CHECKPOINT_PARAMS", default)).expanduser()


def full_ema_decay() -> float | None:
    """Return optional EMA for high-memory full fine-tuning hosts.

    A full float32 EMA copy does not fit together with Adam state on three
    24-GB RTX 4090 GPUs. Leave it disabled there and opt in on larger GPUs.
    """

    value = os.environ.get("PI05_EMA_DECAY")
    if not value:
        return None
    decay = float(value)
    if not 0.0 < decay < 1.0:
        raise ValueError("PI05_EMA_DECAY must be between 0 and 1")
    return decay


def expert_freeze_filter() -> nnx.filterlib.Filter:
    """Freeze the vision encoder and 2B VLM, but train the action expert/heads."""

    llm = nnx_utils.PathRegex(".*llm.*")
    action_expert = nnx_utils.PathRegex(".*llm.*_1.*")
    return nnx.Any(
        nnx_utils.PathRegex(".*img.*"),
        nnx.All(llm, nnx.Not(action_expert)),
    )


def make_config(
    *,
    exp_name: str = "smoke",
    num_train_steps: int = 2,
    batch_size: int = 1,
    fsdp_devices: int = 1,
    finetune_mode: str = "lora",
    optimizer_name: str | None = None,
    peak_lr: float | None = None,
    warmup_steps: int | None = None,
    save_interval: int | None = None,
    synthetic_data: bool = False,
    overwrite: bool = False,
    resume: bool = False,
) -> training_config.TrainConfig:
    """Return a LoRA, action-expert, or full fine-tuning config."""

    if finetune_mode not in CONFIG_NAMES:
        raise ValueError(f"unsupported fine-tuning mode: {finetune_mode!r}")
    optimizer_name = optimizer_name or ("adafactor" if finetune_mode == "full" else "adamw")
    if optimizer_name not in ("adamw", "adafactor"):
        raise ValueError(f"unsupported optimizer: {optimizer_name!r}")
    model_kwargs = {"pi05": True, "action_dim": 32, "action_horizon": 16}
    if finetune_mode == "lora":
        model_kwargs.update(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
    model = pi0_config.Pi0Config(**model_kwargs)

    effective_peak_lr = peak_lr if peak_lr is not None else (2.5e-5 if finetune_mode == "lora" else 1.0e-5)
    effective_warmup = warmup_steps if warmup_steps is not None else min(1_000, max(1, num_train_steps // 10))
    effective_save_interval = save_interval if save_interval is not None else max(1, min(1_000, num_train_steps - 1))
    root = data_root()
    return training_config.TrainConfig(
        name=CONFIG_NAMES[finetune_mode],
        exp_name=exp_name,
        project_name="vla-tidybench",
        model=model,
        data=(
            training_config.FakeDataConfig()
            if synthetic_data
            else training_config.LeRobotLiberoDataConfig(
                repo_id=os.environ.get("VLA_TIDYBENCH_STACK_REPO_ID", REPO_ID),
                base_config=training_config.DataConfig(prompt_from_task=True),
                extra_delta_transform=False,
            )
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(checkpoint_params())),
        freeze_filter=(
            model.get_freeze_filter()
            if finetune_mode == "lora"
            else expert_freeze_filter()
            if finetune_mode == "expert"
            else nnx.Nothing()
        ),
        ema_decay=None if finetune_mode == "lora" else full_ema_decay(),
        lr_schedule=training_optimizer.CosineDecaySchedule(
            warmup_steps=effective_warmup,
            peak_lr=effective_peak_lr,
            decay_steps=max(num_train_steps, effective_warmup + 1),
            decay_lr=effective_peak_lr / 10,
        ),
        optimizer=MemoryEfficientAdafactor() if optimizer_name == "adafactor" else training_optimizer.AdamW(),
        assets_base_dir=str(root / "checkpoints" / "openpi-assets" / "tidybench-assets"),
        checkpoint_base_dir=str(root / "checkpoints" / "openpi-runs"),
        batch_size=batch_size,
        fsdp_devices=fsdp_devices,
        num_workers=0,
        num_train_steps=num_train_steps,
        log_interval=1,
        save_interval=effective_save_interval,
        keep_period=None,
        wandb_enabled=False,
        overwrite=overwrite,
        resume=resume,
    )
