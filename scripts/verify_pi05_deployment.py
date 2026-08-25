#!/usr/bin/env python3
"""Verify a pi0.5 deployment bundle before starting simulation inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_tidybench.openpi.deployment import load_deployment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="inspect a systems-smoke deployment without requiring formal evaluation",
    )
    args = parser.parse_args()
    deployment = load_deployment(args.deployment, require_validated=not args.allow_unvalidated)
    print(
        json.dumps(
            {
                "deployment": str(deployment.root),
                "checkpoint": str(deployment.checkpoint),
                "policy_mode": deployment.policy_mode,
                "project_commit": deployment.manifest.get("project_commit"),
                "file_count": deployment.manifest.get("file_count"),
                "byte_count": deployment.manifest.get("byte_count"),
                "evaluation_gate_passed": deployment.evaluation is not None,
                "verified": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
