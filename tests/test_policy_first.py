import numpy as np
import pytest
from vla_tidybench.policy_bridge.policy_first import ModelGripperDebouncer, compose_policy_first


def test_policy_first_is_primary_and_correction_is_bounded() -> None:
    policy = np.array([0.35, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    reference = np.array([-0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    result = compose_policy_first(policy, reference, max_correction_norm=0.1)
    assert np.linalg.norm(result.correction_raw[:6]) == pytest.approx(0.1)
    assert result.executed_raw[0] == pytest.approx(0.25)
    assert result.executed_raw[6] == -1.0  # DLS cannot override policy gripper.


def test_policy_first_zero_limit_is_exact_policy() -> None:
    policy = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    reference = -policy
    result = compose_policy_first(policy, reference, max_correction_norm=0.0)
    np.testing.assert_array_equal(result.executed_raw, policy)
    np.testing.assert_array_equal(result.correction_raw, np.zeros(7))


def test_policy_first_rejects_bad_actions() -> None:
    with pytest.raises(ValueError):
        compose_policy_first(np.zeros(6), np.zeros(7), max_correction_norm=0.1)
    with pytest.raises(ValueError):
        compose_policy_first(np.full(7, np.nan), np.zeros(7), max_correction_norm=0.1)


def test_model_gripper_debounce_uses_only_model_sign_and_resets_per_episode() -> None:
    gate = ModelGripperDebouncer(close_after=3, open_after=2)
    assert [gate.update(value) for value in (-1, -1, 1, -1, -1, -1, 1, 1)] == [
        1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0
    ]
    assert ModelGripperDebouncer(close_after=3, open_after=2).update(-1) == 1.0


def test_translation_first_correction_prioritizes_position_without_exceeding_bound() -> None:
    policy = np.array([-0.12, 0, 0, -0.5, 0, 0, -1], dtype=np.float32)
    reference = np.array([0.12, 0, 0, 0.5, 0, 0, 1], dtype=np.float32)
    isotropic = compose_policy_first(policy, reference, max_correction_norm=0.2)
    prioritized = compose_policy_first(
        policy, reference, max_correction_norm=0.2, correction_mode="translation_first"
    )
    assert prioritized.correction_raw[0] > isotropic.correction_raw[0]
    assert np.linalg.norm(prioritized.correction_raw[:6]) <= 0.2 + 1e-6
    assert prioritized.executed_raw[6] == -1.0
