from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("summarize_pi05_eval", ROOT / "scripts/summarize_pi05_eval.py")
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)


def write_rollout(
    path: Path,
    *,
    skill: str,
    seed: int,
    success: bool,
    checkpoint: str = "/checkpoints/stage3/2999",
    policy: str = "pi0.5-drawer-expert",
    residual_weight: float = 0.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as output:
        output.attrs["format_version"] = 1
        output.attrs["skill"] = skill
        output.attrs["seed"] = seed
        output.attrs["success"] = success
        output.attrs["policy"] = policy
        output.attrs["policy_checkpoint"] = checkpoint
        output.attrs["policy_residual_weight"] = residual_weight
        output.create_dataset("actions", data=np.zeros((20, 7), dtype=np.float32))
        output.create_dataset("inference_ms", data=np.asarray([90.0, 110.0], dtype=np.float32))


def test_four_skill_autonomous_gate_passes(tmp_path: Path) -> None:
    episodes = []
    for skill in evaluation.SKILLS:
        for seed in (1, 2, 3):
            path = tmp_path / skill / f"seed_{seed}.hdf5"
            write_rollout(path, skill=skill, seed=seed, success=seed != 3)
            episodes.append(evaluation.read_episode(path, allow_assisted=False))

    report = evaluation.summarize(
        episodes,
        required_skills=evaluation.SKILLS,
        min_episodes_per_skill=3,
        min_success_rate=0.6,
        max_p95_infer_ms=250.0,
        input_root=tmp_path,
    )

    assert report["gate_passed"] is True
    assert report["autonomous_only"] is True
    assert report["episode_count"] == 12
    assert report["overall_success_rate"] == pytest.approx(2 / 3)
    assert report["p95_infer_ms"] <= 110.0


def test_assisted_rollout_is_rejected_by_formal_gate(tmp_path: Path) -> None:
    path = tmp_path / "open.hdf5"
    write_rollout(
        path,
        skill="open",
        seed=1,
        success=True,
        policy="pi0.5-drawer-expert+dls-contact-recovery",
        residual_weight=0.02,
    )
    with pytest.raises(ValueError, match="assisted rollout"):
        evaluation.read_episode(path, allow_assisted=False)


def test_mixed_checkpoints_fail_the_gate(tmp_path: Path) -> None:
    first = tmp_path / "open-1.hdf5"
    second = tmp_path / "open-2.hdf5"
    write_rollout(first, skill="open", seed=1, success=True, checkpoint="/checkpoints/a")
    write_rollout(second, skill="open", seed=2, success=True, checkpoint="/checkpoints/b")
    report = evaluation.summarize(
        [
            evaluation.read_episode(first, allow_assisted=False),
            evaluation.read_episode(second, allow_assisted=False),
        ],
        required_skills=("open",),
        min_episodes_per_skill=1,
        min_success_rate=None,
        max_p95_infer_ms=None,
        input_root=tmp_path,
    )
    assert report["gate_passed"] is False
    assert any("expected one checkpoint" in violation for violation in report["violations"])


def test_closed_loop_records_selected_skill_and_checkpoint() -> None:
    source = (ROOT / "scripts/run_drawer_pi05_closed_loop.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--skill"' in source
    assert '"pick": "pick up the medicine bottle"' in source
    assert '"place": "put the medicine bottle into the top drawer"' in source
    assert "tomato soup can" not in source
    assert 'output.attrs["skill"] = skill' in source
    assert 'output.attrs["policy_checkpoint"]' in source
    assert 'output.create_dataset("inference_ms"' in source


def test_three_stage_runner_uses_true_full_tuning() -> None:
    runner = (ROOT / "scripts/run_pi05_three_stage.py").read_text(encoding="utf-8")
    assert 'Stage(2, "stage2-full", "full"' in runner
    assert '"stage3-hard-recovery",\n        "full",' in runner
    assert '"--optimizer", "adafactor"' in runner
    assert '"--fsdp-min-size-mbytes", "0"' in runner
