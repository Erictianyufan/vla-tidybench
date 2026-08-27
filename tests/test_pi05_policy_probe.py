from __future__ import annotations

import numpy as np
from vla_tidybench.policy_bridge.probe_contract import (
    latency_summary,
    synthetic_request,
)


def test_synthetic_probe_request_matches_policy_contract() -> None:
    request = synthetic_request("open the top drawer")

    assert request["observation/state"].shape == (18,)
    assert request["observation/image"].shape == (224, 224, 3)
    assert request["observation/wrist_image"].shape == (224, 224, 3)
    assert request["observation/image"].dtype == np.uint8


def test_latency_summary_uses_all_warm_samples() -> None:
    samples = [
        {"run": index, "policy_infer_ms": value, "round_trip_ms": value + 10.0}
        for index, value in enumerate((100.0, 110.0, 120.0, 130.0, 140.0), start=1)
    ]

    summary = latency_summary(samples)

    assert summary == {
        "policy_p50_ms": 120.0,
        "policy_p95_ms": 138.0,
        "policy_max_ms": 140.0,
        "round_trip_p50_ms": 130.0,
        "round_trip_p95_ms": 148.0,
        "round_trip_max_ms": 150.0,
    }
