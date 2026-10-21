from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = PROJECT_ROOT / "scripts" / "collect_scripted_drawer.py"
ENV_CFG = PROJECT_ROOT / "source" / "vla_tidybench" / "isaac" / "drawer_env_cfg.py"
HOME_FLOOR = PROJECT_ROOT / "assets" / "home_oak_floor.usda"
HOME_FLOOR_TEXTURE = PROJECT_ROOT / "assets" / "home_oak_floor.png"


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
    assert "return Phase.PICK_HOLD if self.height_offset else Phase.DONE" in text
    assert "return Phase.PLACE_CLEARANCE" in text


def test_skill_prerequisites_are_part_of_recorded_reset_state() -> None:
    text = _source(COLLECTOR)
    assert 'if args_cli.skill in ("place", "close")' in text
    assert 'joint_pos["drawer_top_joint"] = 0.36' in text
    assert "reset_to" not in text


def test_home_tabletop_full_shifts_placement_and_preserves_policy_priority() -> None:
    text = _source(COLLECTOR)
    assert 'args_cli.skill not in ("pick", "full")' in text
    for height in ("0.82", "0.86", "0.88"):
        assert f"self.fixed_target[:, 2] = {height} + self.height_offset" in text
    assert "0.68 + self.height_offset < obj[2] < 0.86 + self.height_offset" in text
    assert "compose_policy_first(" in text
    assert 'parser.error("--policy-first requires --policy-host")' in text


def test_home_tabletop_uses_portable_textured_floor() -> None:
    text = _source(ENV_CFG).split("class TidyBenchHomeTablePickEnvCfg", 1)[1]
    assert "self.scene.room = None" in text
    assert 'setattr(self.scene, prop, None)' in text
    assert 'home_oak_floor.usda' in text
    assert 'floor_board_joint_' not in text
    floor = HOME_FLOOR.read_text(encoding="utf-8")
    assert 'defaultPrim = "OakFloor"' in floor
    assert 'UsdUVTexture' in floor
    assert '@./home_oak_floor.png@' in floor
    assert HOME_FLOOR_TEXTURE.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
