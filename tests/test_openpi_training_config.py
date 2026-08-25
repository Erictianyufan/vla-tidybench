from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openpi")

from vla_tidybench.openpi.drawer_four_skill_config import make_config
from vla_tidybench.openpi.stack_config import MemoryEfficientAdafactor


def test_four_skill_configs_use_external_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_root = tmp_path / "large-disk" / "vla-tidybench"
    params = data_root / "models" / "pi05-droid" / "params"
    monkeypatch.setenv("VLA_TIDYBENCH_DATA", str(data_root))
    monkeypatch.setenv("PI05_CHECKPOINT_PARAMS", str(params))
    monkeypatch.setenv("VLA_TIDYBENCH_DRAWER_FOUR_SKILL_REPO_ID", "local/tidybench_four_skill_v1")

    lora = make_config(finetune_mode="lora", num_train_steps=10, batch_size=3, fsdp_devices=3)
    full = make_config(finetune_mode="full", num_train_steps=10, batch_size=3, fsdp_devices=3)
    expert = make_config(finetune_mode="expert", num_train_steps=10, batch_size=3, fsdp_devices=3)

    assert lora.name == "pi05_tidybench_drawer_four_skill_lora"
    assert full.name == "pi05_tidybench_drawer_four_skill_full"
    assert expert.name == "pi05_tidybench_drawer_four_skill_expert"
    assert lora.weight_loader.params_path == str(params)
    assert full.weight_loader.params_path == str(params)
    assert lora.data.repo_id == full.data.repo_id == "local/tidybench_four_skill_v1"
    assert Path(lora.assets_base_dir).is_relative_to(data_root)
    assert Path(full.checkpoint_base_dir).is_relative_to(data_root)
    assert lora.ema_decay is None
    assert full.ema_decay is None
    assert expert.ema_decay is None
    assert isinstance(full.optimizer, MemoryEfficientAdafactor)
    assert not isinstance(lora.optimizer, MemoryEfficientAdafactor)
    assert lora.batch_size == full.batch_size == 3
    assert lora.fsdp_devices == full.fsdp_devices == 3


def test_invalid_finetune_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported fine-tuning mode"):
        make_config(finetune_mode="unsupported")


def test_full_ema_is_an_explicit_high_memory_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI05_EMA_DECAY", "0.99")
    assert make_config(finetune_mode="full").ema_decay == 0.99
