#!/usr/bin/env python3
"""Materialize the locked OOD evaluation plan without launching the simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval/drawer_ood_smoke.json"))
    parser.add_argument("--output", type=Path, default=Path("results/metrics/drawer_ood_manifest.json"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    episodes = []
    seen_seeds: set[int] = set()
    for bucket, settings in config["buckets"].items():
        for seed in settings["seeds"]:
            if seed in seen_seeds:
                raise ValueError(f"duplicate evaluation seed: {seed}")
            seen_seeds.add(seed)
            episodes.append(
                {
                    "bucket": bucket,
                    "seed": seed,
                    "settings": {k: v for k, v in settings.items() if k != "seeds"},
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "episodes": episodes}, indent=2) + "\n", encoding="utf-8")
    print(f"planned {len(episodes)} locked OOD smoke episodes -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
