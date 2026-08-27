#!/usr/bin/env python3
"""Probe a live drawer policy service with a shape-correct synthetic request."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from vla_tidybench.policy_bridge.probe_contract import latency_summary, synthetic_request
from vla_tidybench.policy_bridge.websocket_client import PolicyClient


def infer_once(client: PolicyClient, prompt: str, run: int) -> dict[str, float]:
    started = time.perf_counter()
    response = client.infer(synthetic_request(prompt))
    round_trip_ms = (time.perf_counter() - started) * 1_000
    actions = np.asarray(response.get("actions"))
    if actions.shape != (16, 7) or not np.isfinite(actions).all():
        raise ValueError(f"invalid actions: shape={actions.shape}, finite={np.isfinite(actions).all()}")
    policy_ms = float(response.get("policy_timing", {}).get("infer_ms", float("nan")))
    if not np.isfinite(policy_ms) or policy_ms < 0:
        raise ValueError("policy response has no finite non-negative policy_timing.infer_ms")
    return {"run": run, "policy_infer_ms": policy_ms, "round_trip_ms": round_trip_ms}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="open the top drawer")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--expect-mode", choices=("lora", "expert", "full"), default="full")
    parser.add_argument("--expect-deployment", type=Path)
    parser.add_argument("--require-evaluation", action="store_true")
    parser.add_argument("--max-last-infer-ms", type=float)
    parser.add_argument("--max-p95-infer-ms", type=float)
    parser.add_argument("--max-p95-round-trip-ms", type=float)
    args = parser.parse_args()
    if args.warmup_runs < 0 or args.runs < 1 or args.timeout_s <= 0:
        parser.error("--warmup-runs must be non-negative; --runs and --timeout-s must be positive")
    thresholds = (args.max_last_infer_ms, args.max_p95_infer_ms, args.max_p95_round_trip_ms)
    if any(value is not None and value <= 0 for value in thresholds):
        parser.error("latency thresholds must be positive")

    warmup_samples: list[dict[str, float]] = []
    samples: list[dict[str, float]] = []
    with PolicyClient(args.host, args.port, timeout_s=args.timeout_s) as client:
        metadata = client.metadata
        expected_policy = f"pi0.5-drawer-{args.expect_mode}"
        if metadata.get("policy") != expected_policy:
            raise ValueError(f"expected policy {expected_policy!r}, got {metadata.get('policy')!r}")
        checkpoint_sha256 = str(metadata.get("checkpoint_sha256", ""))
        if len(checkpoint_sha256) != 64 or any(char not in "0123456789abcdef" for char in checkpoint_sha256):
            raise ValueError("policy service has no valid checkpoint SHA-256")
        if args.expect_deployment is not None:
            expected = str(args.expect_deployment.expanduser().resolve())
            if metadata.get("deployment") != expected:
                raise ValueError(f"expected deployment {expected!r}, got {metadata.get('deployment')!r}")
        if args.require_evaluation and not bool(metadata.get("evaluation_gate_passed", False)):
            raise ValueError("policy service is not backed by a passing formal evaluation")
        if args.require_evaluation and not bool(metadata.get("training_completion_verified", False)):
            raise ValueError("policy service is not backed by a verified training completion")
        training_dataset_sha256 = str(metadata.get("training_dataset_sha256", ""))
        if args.require_evaluation and (
            len(training_dataset_sha256) != 64
            or any(char not in "0123456789abcdef" for char in training_dataset_sha256)
        ):
            raise ValueError("policy service has no verified training dataset SHA-256")
        training_openpi_sha256 = str(metadata.get("openpi_source_sha256", ""))
        runtime_openpi_sha256 = str(metadata.get("runtime_openpi_source_sha256", ""))
        if args.require_evaluation and (
            len(runtime_openpi_sha256) != 64
            or runtime_openpi_sha256 != training_openpi_sha256
        ):
            raise ValueError("policy runtime OpenPI source differs from its training source")

        for run in range(1, args.warmup_runs + 1):
            warmup_samples.append(infer_once(client, args.prompt, run))
        for run in range(1, args.runs + 1):
            samples.append(infer_once(client, args.prompt, run))

    latency = latency_summary(samples)
    if args.max_last_infer_ms is not None and samples[-1]["policy_infer_ms"] > args.max_last_infer_ms:
        raise ValueError(
            f"last inference took {samples[-1]['policy_infer_ms']:.1f} ms, "
            f"limit is {args.max_last_infer_ms:.1f} ms"
        )
    if args.max_p95_infer_ms is not None and latency["policy_p95_ms"] > args.max_p95_infer_ms:
        raise ValueError(
            f"P95 policy inference took {latency['policy_p95_ms']:.1f} ms, "
            f"limit is {args.max_p95_infer_ms:.1f} ms"
        )
    if (
        args.max_p95_round_trip_ms is not None
        and latency["round_trip_p95_ms"] > args.max_p95_round_trip_ms
    ):
        raise ValueError(
            f"P95 round trip took {latency['round_trip_p95_ms']:.1f} ms, "
            f"limit is {args.max_p95_round_trip_ms:.1f} ms"
        )
    print(
        json.dumps(
            {
                "host": args.host,
                "port": args.port,
                "policy": expected_policy,
                "checkpoint": metadata.get("checkpoint"),
                "checkpoint_sha256": checkpoint_sha256,
                "deployment": metadata.get("deployment"),
                "evaluation_gate_passed": bool(metadata.get("evaluation_gate_passed", False)),
                "training_completion_verified": bool(
                    metadata.get("training_completion_verified", False)
                ),
                "training_dataset_sha256": training_dataset_sha256 or None,
                "training_openpi_source_sha256": training_openpi_sha256 or None,
                "runtime_openpi_source_sha256": runtime_openpi_sha256 or None,
                "synthetic_identity_norm": bool(metadata.get("synthetic_identity_norm", False)),
                "actions_shape": [16, 7],
                "warmup_samples": warmup_samples,
                "samples": samples,
                "latency": latency,
                "passed": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
