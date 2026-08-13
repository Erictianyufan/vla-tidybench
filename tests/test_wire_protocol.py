import numpy as np

from vla_tidybench.policy_bridge import wire_protocol


def test_numpy_wire_roundtrip():
    value = {
        "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "state": np.array([1.5, -2.0], dtype=np.float32),
    }
    restored = wire_protocol.unpackb(wire_protocol.packb(value))
    np.testing.assert_array_equal(restored["image"], value["image"])
    np.testing.assert_array_equal(restored["state"], value["state"])
