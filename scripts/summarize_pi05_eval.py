#!/usr/bin/env python3
"""Audit closed-loop HDF5 rollouts and emit a machine-readable π0.5 evaluation gate."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import numpy as np


SKILLS = ("open", "pick", "place", "close")


@dataclass(frozen=True)
class Episode:
    path: Path
    skill: str
    seed: int
    success: bool
    steps: int
    policy: str
    checkpoint: str
    residual_weight: float
    context: str
    inference_ms: np.ndarray


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def read_episode(path: Path, *, allow_assisted: bool) -> Episode:
    with h5py.File(path, "r") as source:
        if int(source.attrs.get("format_version", -1)) != 1:
            raise ValueError(f"unsupported evaluation format: {path}")
        skill = _text(source.attrs.get("skill", ""))
        if skill not in SKILLS:
            raise ValueError(f"invalid or missing skill in {path}: {skill!r}")
        policy = _text(source.attrs.get("policy", ""))
        checkpoint = _text(source.attrs.get("policy_checkpoint", ""))
        residual_weight = float(source.attrs.get("policy_residual_weight", 0.0))
        context_file = _text(source.attrs.get("initial_state_file", ""))
        context_episode = _text(source.attrs.get("initial_state_episode", ""))
        context = f"{Path(context_file).name}::{context_episode}" if context_file and context_episode else ""
        assisted = residual_weight != 0.0 or "teacher" in policy.lower() or "+dls" in policy.lower()
        if assisted and not allow_assisted:
            raise ValueError(f"assisted rollout is not valid for autonomous policy evaluation: {path}")
        if not assisted and not checkpoint:
            raise ValueError(f"autonomous rollout does not identify its checkpoint: {path}")
        if not assisted and not context:
            raise ValueError(f"autonomous rollout does not identify its held-out initial-state context: {path}")
        if "actions" not in source or "inference_ms" not in source:
            raise ValueError(f"missing actions/inference_ms in {path}")
        inference_ms = np.asarray(source["inference_ms"], dtype=np.float64)
        if inference_ms.ndim != 1 or not np.isfinite(inference_ms).all() or np.any(inference_ms < 0):
            raise ValueError(f"invalid inference timing data in {path}")
        if not assisted and inference_ms.size == 0:
            raise ValueError(f"autonomous rollout has no policy timing samples: {path}")
        return Episode(
            path=path,
            skill=skill,
            seed=int(source.attrs.get("seed", -1)),
            success=bool(source.attrs.get("success", False)),
            steps=int(source["actions"].shape[0]),
            policy=policy,
            checkpoint=checkpoint,
            residual_weight=residual_weight,
            context=context,
            inference_ms=inference_ms,
        )


def summarize(
    episodes: list[Episode],
    *,
    required_skills: tuple[str, ...],
    min_episodes_per_skill: int,
    min_success_rate: float | None,
    max_p95_infer_ms: float | None,
    input_root: Path,
) -> dict[str, object]:
    violations: list[str] = []
    policies = sorted({episode.policy for episode in episodes})
    checkpoints = sorted({episode.checkpoint for episode in episodes if episode.checkpoint})
    if len(policies) != 1:
        violations.append(f"expected one policy, found {policies}")
    if len(checkpoints) != 1:
        violations.append(f"expected one checkpoint, found {checkpoints}")

    per_skill: dict[str, object] = {}
    all_timings: list[np.ndarray] = []
    for skill in required_skills:
        selected = [episode for episode in episodes if episode.skill == skill]
        if len(selected) < min_episodes_per_skill:
            violations.append(f"{skill}: {len(selected)} episodes < required {min_episodes_per_skill}")
        if len({episode.seed for episode in selected}) != len(selected):
            violations.append(f"{skill}: duplicate evaluation seeds")
        if len({episode.context for episode in selected}) != len(selected):
            violations.append(f"{skill}: duplicate held-out initial-state contexts")
        successes = sum(episode.success for episode in selected)
        success_rate = successes / len(selected) if selected else 0.0
        if min_success_rate is not None and success_rate < min_success_rate:
            violations.append(f"{skill}: success_rate {success_rate:.3f} < required {min_success_rate:.3f}")
        timings = np.concatenate([episode.inference_ms for episode in selected if episode.inference_ms.size]) if any(
            episode.inference_ms.size for episode in selected
        ) else np.asarray([], dtype=np.float64)
        all_timings.extend(episode.inference_ms for episode in selected if episode.inference_ms.size)
        per_skill[skill] = {
            "episodes": len(selected),
            "successes": successes,
            "success_rate": success_rate,
            "mean_steps": float(np.mean([episode.steps for episode in selected])) if selected else None,
            "mean_infer_ms": float(np.mean(timings)) if timings.size else None,
            "p95_infer_ms": float(np.percentile(timings, 95)) if timings.size else None,
        }

    combined_timings = np.concatenate(all_timings) if all_timings else np.asarray([], dtype=np.float64)
    overall_p95 = float(np.percentile(combined_timings, 95)) if combined_timings.size else None
    if max_p95_infer_ms is not None and (overall_p95 is None or overall_p95 > max_p95_infer_ms):
        violations.append(f"overall p95 inference {overall_p95} ms exceeds {max_p95_infer_ms:.1f} ms")

    used = [episode for episode in episodes if episode.skill in required_skills]
    return {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_root": str(input_root.resolve()),
        "autonomous_only": all(episode.residual_weight == 0.0 for episode in used),
        "policy": policies[0] if len(policies) == 1 else None,
        "checkpoint": checkpoints[0] if len(checkpoints) == 1 else None,
        "required_skills": list(required_skills),
        "episode_count": len(used),
        "successes": sum(episode.success for episode in used),
        "overall_success_rate": sum(episode.success for episode in used) / len(used) if used else 0.0,
        "p50_infer_ms": float(np.percentile(combined_timings, 50)) if combined_timings.size else None,
        "p95_infer_ms": overall_p95,
        "per_skill": per_skill,
        "thresholds": {
            "min_episodes_per_skill": min_episodes_per_skill,
            "min_success_rate": min_success_rate,
            "max_p95_infer_ms": max_p95_infer_ms,
        },
        "gate_passed": not violations,
        "violations": violations,
        "episodes": [
            {
                "path": str(episode.path.resolve()),
                "bytes": episode.path.stat().st_size,
                "skill": episode.skill,
                "seed": episode.seed,
                "context": episode.context,
                "success": episode.success,
                "steps": episode.steps,
                "mean_infer_ms": float(np.mean(episode.inference_ms)) if episode.inference_ms.size else None,
            }
            for episode in used
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skills", nargs="+", choices=SKILLS, default=list(SKILLS))
    parser.add_argument("--min-episodes-per-skill", type=int, default=3)
    parser.add_argument("--min-success-rate", type=float)
    parser.add_argument("--max-p95-infer-ms", type=float)
    parser.add_argument("--allow-assisted", action="store_true")
    args = parser.parse_args()
    if args.min_episodes_per_skill < 1:
        parser.error("--min-episodes-per-skill must be positive")
    if args.min_success_rate is not None and not 0.0 <= args.min_success_rate <= 1.0:
        parser.error("--min-success-rate must be in [0, 1]")
    if args.max_p95_infer_ms is not None and args.max_p95_infer_ms <= 0:
        parser.error("--max-p95-infer-ms must be positive")

    paths = sorted(args.input_root.rglob("*.hdf5"))
    if not paths:
        parser.error(f"no HDF5 evaluations below {args.input_root}")
    try:
        episodes = [read_episode(path, allow_assisted=args.allow_assisted) for path in paths]
        report = summarize(
            episodes,
            required_skills=tuple(args.skills),
            min_episodes_per_skill=args.min_episodes_per_skill,
            min_success_rate=args.min_success_rate,
            max_p95_infer_ms=args.max_p95_infer_ms,
            input_root=args.input_root,
        )
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
