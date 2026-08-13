#!/usr/bin/env python3
"""Write an auditable manifest for the cost-bounded project release candidate."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/ubuntu/data/vla-tidybench/raw")
CHECKPOINT = Path(
    "/home/ubuntu/data/vla-tidybench/checkpoints/openpi-runs/"
    "pi05_tidybench_drawer_lora/drawer-smoke/1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_record(skill: str) -> dict:
    path = DATA_ROOT / f"drawer_{skill}_smoke.hdf5"
    with h5py.File(path, "r") as dataset:
        episode = dataset["data"][sorted(dataset["data"].keys())[0]]
        return {
            "skill": skill,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "steps": int(episode["actions"].shape[0]),
            "action_shape": list(episode["actions"].shape[1:]),
            "observation_keys": sorted(episode["obs"].keys()),
            "recorded_success": bool(episode.attrs["success"]),
        }


def main() -> int:
    video = ROOT / "artifacts/demo/vla-tidybench-demo.mp4"
    checkpoint_metadata = CHECKPOINT / "_CHECKPOINT_METADATA"
    datasets = [dataset_record(skill) for skill in ("open", "pick", "place", "close")]
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "release_scope": "cost_bounded_core_chain",
        "dataset": {
            "episodes": len(datasets),
            "frames": sum(record["steps"] for record in datasets),
            "fps": 20,
            "records": datasets,
        },
        "openpi": {
            "config": "pi05_tidybench_drawer_lora",
            "training_steps": 2,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_metadata_sha256": sha256(checkpoint_metadata),
            "restore_and_offline_inference": "passed",
            "action_chunk_shape": [16, 7],
            "first_jit_inference_ms": 17145.0,
            "claim": "pipeline smoke only; no learned-policy success claim",
        },
        "demo": {
            "path": str(video),
            "bytes": video.stat().st_size,
            "sha256": sha256(video),
            "kind": "four replay-validated scripted-teacher atomic skills",
        },
        "extensions": {
            "mimic": "config retained; long generation not run",
            "ood": "locked 8-episode smoke manifest retained; rollouts not run",
            "residual_rl": "bounded composer/reward/config tested; training not run",
        },
        "known_limitations": [
            "continuous OPEN-PICK-PLACE-CLOSE TaskGraph rollout has not passed",
            "demo is not a learned VLA policy rollout",
            "two-step LoRA checkpoint is not a quality checkpoint",
        ],
    }
    output = ROOT / "results/metrics/drawer_release_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
