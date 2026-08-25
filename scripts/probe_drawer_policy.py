#!/usr/bin/env python3
"""Probe a live drawer policy service with a shape-correct synthetic request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from vla_tidybench.policy_bridge.websocket_client import PolicyClient


def synthetic_request(prompt: str) -> dict[str, object]:
    return {
        "observation/state": np.zeros(18, dtype=np.float32),
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "prompt": prompt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="open the top drawer")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--expect-mode", choices=("lora", "expert", "full"), default="full")
    parser.add_argument("--expect-deployment", type=Path)
    parser.add_argument("--require-evaluation", action="store_true")
    parser.add_argument("--max-last-infer-ms", type=float)
    args = parser.parse_args()
    if args.runs < 1 or args.timeout_s <= 0:
        parser.error("--runs and --timeout-s must be positive")

    samples: list[dict[str, float]] = []
    with PolicyClient(args.host, args.port, timeout_s=args.timeout_s) as client:
        metadata = client.metadata
        expected_policy = f"pi0.5-drawer-{args.expect_mode}"
        if metadata.get("policy") != expected_policy:
            raise ValueError(f"expected policy {expected_policy!r}, got {metadata.get('policy')!r}")
        if args.expect_deployment is not None:
            expected = str(args.expect_deployment.expanduser().resolve())
            if metadata.get("deployment") != expected:
                raise ValueError(f"expected deployment {expected!r}, got {metadata.get('deployment')!r}")
        if args.require_evaluation and not bool(metadata.get("evaluation_gate_passed", False)):
            raise ValueError("policy service is not backed by a passing formal evaluation")

        for run in range(1, args.runs + 1):
            started = time.perf_counter()
            response = client.infer(synthetic_request(args.prompt))
            round_trip_ms = (time.perf_counter() - started) * 1_000
            actions = np.asarray(response.get("actions"))
            if actions.shape != (16, 7) or not np.isfinite(actions).all():
                raise ValueError(f"invalid actions: shape={actions.shape}, finite={np.isfinite(actions).all()}")
            policy_ms = float(response.get("policy_timing", {}).get("infer_ms", float("nan")))
            if not np.isfinite(policy_ms):
                raise ValueError("policy response has no finite policy_timing.infer_ms")
            samples.append({"run": run, "policy_infer_ms": policy_ms, "round_trip_ms": round_trip_ms})

    if args.max_last_infer_ms is not None and samples[-1]["policy_infer_ms"] > args.max_last_infer_ms:
        raise ValueError(
            f"last inference took {samples[-1]['policy_infer_ms']:.1f} ms, "
            f"limit is {args.max_last_infer_ms:.1f} ms"
        )
    print(
        json.dumps(
            {
                "host": args.host,
                "port": args.port,
                "policy": expected_policy,
                "checkpoint": metadata.get("checkpoint"),
                "deployment": metadata.get("deployment"),
                "evaluation_gate_passed": bool(metadata.get("evaluation_gate_passed", False)),
                "synthetic_identity_norm": bool(metadata.get("synthetic_identity_norm", False)),
                "actions_shape": [16, 7],
                "samples": samples,
                "passed": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
