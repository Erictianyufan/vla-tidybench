import numpy as np

from vla_tidybench.rl import ResidualComposer, open_drawer_reward


def test_zero_residual_is_exactly_nominal() -> None:
    nominal = np.array([0.01, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    np.testing.assert_array_equal(ResidualComposer().compose(nominal, np.zeros(6)), nominal)


def test_residual_is_bounded_and_never_changes_gripper() -> None:
    nominal = np.zeros(7, dtype=np.float32)
    nominal[6] = 1
    output = ResidualComposer(beta=1).compose(nominal, np.full(6, 100))
    np.testing.assert_allclose(output[:3], 0.015)
    np.testing.assert_allclose(output[3:6], 0.08)
    assert output[6] == 1


def test_successful_progress_outranks_stalling() -> None:
    moving = open_drawer_reward(
        previous_q=0.1, current_q=0.15, contact=True, collision=False, residual_norm=0.1, success=False
    )
    stalled = open_drawer_reward(
        previous_q=0.1, current_q=0.1, contact=True, collision=False, residual_norm=0.1, success=False
    )
    success = open_drawer_reward(
        previous_q=0.29, current_q=0.31, contact=True, collision=False, residual_norm=0.1, success=True
    )
    assert success["total"] > moving["total"] > stalled["total"]
