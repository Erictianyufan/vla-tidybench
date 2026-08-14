import numpy as np
from vla_tidybench.rl import ResidualComposer, open_drawer_reward, pick_residual_reward


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


def test_pick_reward_prefers_lift_and_success() -> None:
    stalled = pick_residual_reward(
        previous_distance=0.1,
        current_distance=0.1,
        previous_object_z=0.02,
        current_object_z=0.02,
        residual_norm=0.0,
        success=False,
        truncated=False,
    )
    lifted = pick_residual_reward(
        previous_distance=0.1,
        current_distance=0.08,
        previous_object_z=0.02,
        current_object_z=0.04,
        residual_norm=0.1,
        success=False,
        truncated=False,
    )
    success = pick_residual_reward(
        previous_distance=0.03,
        current_distance=0.03,
        previous_object_z=0.11,
        current_object_z=0.12,
        residual_norm=0.1,
        success=True,
        truncated=False,
    )
    assert success["total"] > lifted["total"] > stalled["total"]
