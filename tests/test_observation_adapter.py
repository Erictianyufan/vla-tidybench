import numpy as np
import pytest
from vla_tidybench.policy_bridge.observation_adapter import make_observation


def test_accepts_deployable_observation():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    observation = make_observation(image, image, np.zeros(18), "pick up the red cube")
    assert observation.table_rgb.shape == (200, 200, 3)
    assert observation.robot_state.dtype == np.float32


def test_rejects_privileged_or_malformed_image_dtype():
    image = np.zeros((200, 200, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        make_observation(image, image, np.zeros(18), "pick")

