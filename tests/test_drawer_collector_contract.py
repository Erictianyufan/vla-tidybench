from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = PROJECT_ROOT / "scripts" / "collect_scripted_drawer.py"
ENV_CFG = PROJECT_ROOT / "source" / "vla_tidybench" / "isaac" / "drawer_env_cfg.py"


def _source(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    ast.parse(text)
    return text


def test_drawer_policy_observation_excludes_privileged_truth() -> None:
    text = _source(ENV_CFG)
    policy_block = text.split("class PolicyCfg", 1)[1].split("def __post_init__", 1)[0]
    assert "joint_pos" in policy_block
    assert "joint_vel" in policy_block
    assert "table_cam" in policy_block
    assert "wrist_cam" in policy_block
    for forbidden in ("target_object", "cabinet_frame", "object_pose", "drawer_pos", "handle_pose"):
        assert forbidden not in policy_block


def test_drawer_teacher_uses_native_dls_action_contract() -> None:
    text = _source(COLLECTOR)
    assert "compute_pose_error" in text
    assert "torch.zeros((1, 7)" in text
    # The teacher emits task-space deltas; it must not implement a joint-space
    # pseudo-inverse itself because the environment owns the DLS solve.
    assert "torch.linalg.pinv" not in text
    assert "jacobian" not in text.lower()
    assert 'gym.make("Isaac-Open-Drawer-Franka-IK-Rel-v0"' in text


def test_place_path_has_explicit_vertical_clearance() -> None:
    text = _source(COLLECTOR)
    assert "PLACE_CLEARANCE" in text
    assert "PLACE_INSERT" in text
    assert "self.fixed_target[:, 2] = 0.82" in text
    assert "self.place_handle_anchor" in text
    assert "return Phase.DONE if self.skill == \"pick\" else Phase.PLACE_CLEARANCE" in text


def test_skill_prerequisites_are_part_of_recorded_reset_state() -> None:
    text = _source(COLLECTOR)
    assert 'if args_cli.skill in ("pick", "place", "close")' in text
    assert '0.39 if args_cli.skill == "place" else 0.36' in text
    assert "PLACE_HELD_JOINT_POS" in text
    assert "PLACE_HELD_OBJECT_POSE" in text
    assert "CLOSE_OBJECT_POSE" in text
    assert "return Phase.PLACE_ABOVE" in text
    assert "reset_to" not in text
