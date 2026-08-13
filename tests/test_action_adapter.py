import numpy as np
import pytest

from vla_tidybench.policy_bridge import ActionAdapter


def test_physical_isaac_roundtrip_inside_limits():
    adapter = ActionAdapter()
    action = np.array([0.01, -0.02, 0.005, 0.05, -0.08, 0.1, -1.0], dtype=np.float32)
    np.testing.assert_allclose(adapter.from_isaac(adapter.to_isaac(action)), action, atol=1e-7)


def test_adapter_clips_and_binarizes():
    adapted = ActionAdapter().to_isaac([1, -1, 0, 2, -2, 0, 0.2])
    np.testing.assert_allclose(adapted[:3], [0.05, -0.05, 0.0])
    np.testing.assert_allclose(adapted[3:6], [0.24, -0.24, 0.0])
    assert adapted[6] == 1.0


@pytest.mark.parametrize("bad", ([0] * 6, [0] * 8, [0, 0, 0, 0, 0, np.nan, 1]))
def test_adapter_rejects_bad_actions(bad):
    with pytest.raises(ValueError):
        ActionAdapter().to_isaac(bad)

