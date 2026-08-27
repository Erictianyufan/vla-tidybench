from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from vla_tidybench.task_metrics import FORMAL_SUCCESS_HOLD_STEPS, SUCCESS_PREDICATE_VERSION

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = "a" * 64
SPEC = importlib.util.spec_from_file_location("summarize_pi05_eval", ROOT / "scripts/summarize_pi05_eval.py")
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)

SUITE_SPEC = importlib.util.spec_from_file_location("run_pi05_eval_suite", ROOT / "scripts/run_pi05_eval_suite.py")
assert SUITE_SPEC is not None and SUITE_SPEC.loader is not None
evaluation_suite = importlib.util.module_from_spec(SUITE_SPEC)
sys.modules[SUITE_SPEC.name] = evaluation_suite
SUITE_SPEC.loader.exec_module(evaluation_suite)


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
    initial_drawer = 0.36
    initial_object = np.asarray([0.20, -0.20, 0.066], dtype=np.float32)
    final_drawer = initial_drawer
    final_object = initial_object.copy()
    final_gripper_width = 0.08
    final_handle_x = 0.29
    if success and skill == "open":
        final_drawer = 0.31
    elif success and skill == "pick":
        final_object = np.asarray([0.20, -0.20, 0.185], dtype=np.float32)
        final_gripper_width = 0.048
    elif success and skill == "place":
        final_object = np.asarray([0.325, -0.20, 0.712], dtype=np.float32)
    elif success and skill == "close":
        initial_object = np.asarray([0.61, -0.107, 0.731], dtype=np.float32)
        final_drawer = 0.001
        final_object = np.asarray([0.96, -0.107, 0.712], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as output:
        output.attrs["format_version"] = 1
        output.attrs["skill"] = skill
        output.attrs["seed"] = seed
        output.attrs["success"] = success
        output.attrs["policy"] = policy
        output.attrs["policy_checkpoint"] = checkpoint
        output.attrs["policy_checkpoint_sha256"] = CHECKPOINT_SHA256
        output.attrs["policy_residual_weight"] = residual_weight
        output.attrs["initial_state_file"] = f"/data/drawer_{skill}_formal.hdf5"
        output.attrs["initial_state_episode"] = f"demo_{seed}"
        output.attrs["success_predicate_version"] = SUCCESS_PREDICATE_VERSION
        output.attrs["success_hold_steps_required"] = FORMAL_SUCCESS_HOLD_STEPS
        output.attrs["success_hold_steps_observed"] = FORMAL_SUCCESS_HOLD_STEPS if success else 0
        output.attrs["initial_drawer_m"] = initial_drawer
        output.attrs["final_drawer_m"] = final_drawer
        output.attrs["initial_object_xyz_m"] = initial_object
        output.attrs["final_object_xyz_m"] = final_object
        output.attrs["final_gripper_width_m"] = final_gripper_width
        output.attrs["final_handle_x_m"] = final_handle_x
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
    assert report["checkpoint_sha256"] == CHECKPOINT_SHA256
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


def test_legacy_success_predicate_is_rejected_by_formal_gate(tmp_path: Path) -> None:
    path = tmp_path / "open.hdf5"
    write_rollout(path, skill="open", seed=1, success=True)
    with h5py.File(path, "r+") as output:
        del output.attrs["success_predicate_version"]
    with pytest.raises(ValueError, match="success predicate"):
        evaluation.read_episode(path, allow_assisted=False)


def test_forged_success_label_is_rejected_by_state_audit(tmp_path: Path) -> None:
    path = tmp_path / "pick.hdf5"
    write_rollout(path, skill="pick", seed=1, success=True)
    with h5py.File(path, "r+") as output:
        output.attrs["final_object_xyz_m"] = output.attrs["initial_object_xyz_m"]
    with pytest.raises(ValueError, match="disagrees with audited"):
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


def test_mixed_checkpoint_digests_fail_the_gate(tmp_path: Path) -> None:
    first = tmp_path / "open-1.hdf5"
    second = tmp_path / "open-2.hdf5"
    write_rollout(first, skill="open", seed=1, success=True)
    write_rollout(second, skill="open", seed=2, success=True)
    with h5py.File(second, "r+") as output:
        output.attrs["policy_checkpoint_sha256"] = "b" * 64
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
    assert any("expected one checkpoint SHA-256" in violation for violation in report["violations"])


def test_duplicate_contexts_fail_the_gate(tmp_path: Path) -> None:
    first = tmp_path / "open-1.hdf5"
    second = tmp_path / "open-2.hdf5"
    write_rollout(first, skill="open", seed=1, success=True)
    write_rollout(second, skill="open", seed=2, success=True)
    with h5py.File(second, "r+") as output:
        output.attrs["initial_state_episode"] = "demo_1"
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
    assert any("duplicate held-out initial-state contexts" in violation for violation in report["violations"])


def test_eval_suite_loads_distinct_validation_contexts(tmp_path: Path) -> None:
    sources = []
    raw = tmp_path / "raw"
    raw.mkdir()
    for skill in evaluation_suite.SKILLS:
        source = raw / f"drawer_{skill}_formal.hdf5"
        with h5py.File(source, "w") as dataset:
            dataset.attrs["format_version"] = 1
            data = dataset.create_group("data")
            for name in ("demo_5", "demo_12", "demo_27"):
                data.create_group(name)
        sources.append(
            {
                "file": source.name,
                "episode_indices": [2, 0, 1],
                "prompt": skill,
                "role": "nominal",
            }
        )
    manifest = tmp_path / "main_validation.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "split": "validation", "sources": sources}),
        encoding="utf-8",
    )

    contexts = evaluation_suite.load_contexts(
        manifest,
        raw,
        skills=list(evaluation_suite.SKILLS),
        seeds=[300, 301, 302],
    )

    assert contexts[("open", 300)] == ((raw / "drawer_open_formal.hdf5").resolve(), "demo_5")
    assert contexts[("open", 301)][1] == "demo_12"
    assert contexts[("open", 302)][1] == "demo_27"
    for skill in evaluation_suite.SKILLS:
        assert len({contexts[(skill, seed)][1] for seed in (300, 301, 302)}) == 3
        assert all(contexts[(skill, seed)][0].name == f"drawer_{skill}_formal.hdf5" for seed in (300, 301, 302))


def test_closed_loop_records_selected_skill_and_checkpoint() -> None:
    source = (ROOT / "scripts/run_drawer_pi05_closed_loop.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--skill"' in source
    assert '"pick": "pick up the medicine bottle"' in source
    assert '"place": "put the medicine bottle into the top drawer"' in source
    assert "tomato soup can" not in source
    assert 'output.attrs["skill"] = skill' in source
    assert 'output.attrs["policy_checkpoint"]' in source
    assert 'output.attrs["policy_checkpoint_sha256"]' in source
    assert 'parser.add_argument("--initial-state-file"' in source
    assert 'parser.add_argument("--initial-state-episode")' in source
    assert "env.reset_to(context.get_initial_state()" in source
    assert 'output.attrs["initial_state_episode"]' in source
    assert 'parser.add_argument("--success-hold-steps"' in source
    assert 'output.attrs["success_predicate_version"]' in source
    assert "drawer_skill_success(" in source
    assert 'output.create_dataset("inference_ms"' in source


def test_three_stage_runner_uses_true_full_tuning() -> None:
    runner = (ROOT / "scripts/run_pi05_three_stage.py").read_text(encoding="utf-8")
    assert 'Stage(2, "stage2-full", "full"' in runner
    assert '"stage3-hard-recovery",\n        "full",' in runner
    assert '"--optimizer", "adafactor"' in runner
    assert '"--fsdp-min-size-mbytes", "0"' in runner
