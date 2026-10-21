import numpy as np
import pytest
from vla_tidybench.policy_bridge.policy_first import compose_policy_first


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
