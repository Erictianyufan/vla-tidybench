"""Pure request and latency helpers for live policy readiness probes."""

from __future__ import annotations

import numpy as np


def synthetic_request(prompt: str) -> dict[str, object]:
    return {
        "observation/state": np.zeros(18, dtype=np.float32),
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "prompt": prompt,
    }


def latency_summary(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        raise ValueError("cannot summarize an empty latency sample")
    policy = np.asarray([sample["policy_infer_ms"] for sample in samples], dtype=np.float64)
    round_trip = np.asarray([sample["round_trip_ms"] for sample in samples], dtype=np.float64)
    return {
        "policy_p50_ms": float(np.percentile(policy, 50)),
        "policy_p95_ms": float(np.percentile(policy, 95)),
        "policy_max_ms": float(np.max(policy)),
        "round_trip_p50_ms": float(np.percentile(round_trip, 50)),
        "round_trip_p95_ms": float(np.percentile(round_trip, 95)),
        "round_trip_max_ms": float(np.max(round_trip)),
    }
