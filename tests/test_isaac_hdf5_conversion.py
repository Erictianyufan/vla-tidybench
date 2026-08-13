import numpy as np
import pytest
from vla_tidybench.data.isaac_hdf5 import canonical_actions, deployable_state


def test_converts_isaac_raw_action_to_canonical_physical_action():
    raw = np.array([[0.10, -0.08, 0.04, 0.2, -0.1, 0.0, 0.2]], dtype=np.float32)
    converted = canonical_actions(raw)
    np.testing.assert_allclose(converted[0, :6], raw[0, :6] * 0.5)
    assert converted[0, 6] == 1.0


def test_builds_only_deployable_18d_robot_state():
    q = np.arange(18, dtype=np.float32).reshape(2, 9)
    qdot = -q
    state = deployable_state(q, qdot)
    assert state.shape == (2, 18)
    np.testing.assert_array_equal(state[:, :9], q)
    np.testing.assert_array_equal(state[:, 9:], qdot)


def test_rejects_malformed_action_shape():
    with pytest.raises(ValueError, match="shape"):
        canonical_actions(np.zeros((4, 8), dtype=np.float32))
