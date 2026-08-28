#!/usr/bin/env python3
"""Run a deterministic four-skill Isaac closed-loop suite and summarize its deployment gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import h5py
from vla_tidybench.evaluation_contexts import validate_context_lock  # noqa: E402

SKILLS = ("open", "pick", "place", "close")
DEFAULT_MAX_STEPS = {"open": 360, "pick": 300, "place": 420, "close": 300}


def default_data_root() -> Path:
    return Path(os.environ.get("VLA_TIDYBENCH_DATA", f"/data/{os.environ.get('USER', 'user')}/vla-tidybench"))


def sorted_episode_names(data: h5py.Group) -> list[str]:
    try:
        return sorted(data.keys(), key=lambda name: int(name.removeprefix("demo_")))
    except ValueError as error:
        raise ValueError("context episodes must use demo_<integer> names") from error


def load_contexts(
    manifest_path: Path,
    data_root: Path,
    *,
    skills: list[str],
    seeds: list[int],
) -> dict[tuple[str, int], tuple[Path, str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != 1 or manifest.get("split") != "validation":
        raise ValueError(f"expected a schema-1 validation manifest: {manifest_path}")
    data_root = data_root.expanduser().resolve()
    contexts: dict[tuple[str, int], tuple[Path, str]] = {}
    for skill in skills:
        expected_name = f"drawer_{skill}_formal.hdf5"
        matches = [source for source in manifest.get("sources", []) if Path(str(source["file"])).name == expected_name]
        if len(matches) != 1:
            raise ValueError(f"{manifest_path} must contain exactly one source named {expected_name}")
        source = matches[0]
        indices = sorted({int(index) for index in source.get("episode_indices", [])})
        if len(indices) < len(seeds):
            raise ValueError(f"{expected_name} has {len(indices)} contexts, needs {len(seeds)}")
        source_path = (data_root / str(source["file"])).resolve()
        if not source_path.is_relative_to(data_root) or not source_path.is_file():
            raise ValueError(f"invalid or missing context source: {source_path}")
        with h5py.File(source_path, "r") as dataset:
            if int(dataset.attrs.get("format_version", -1)) != 1 or "data" not in dataset:
                raise ValueError(f"unsupported context HDF5 format: {source_path}")
            names = sorted_episode_names(dataset["data"])
        if any(index < 0 or index >= len(names) for index in indices):
            raise ValueError(f"{expected_name} contains an out-of-range episode index")
        for seed, index in zip(seeds, indices):
            contexts[(skill, seed)] = (source_path, names[index])
    return contexts


def main() -> int:
    data_root = default_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skills", nargs="+", choices=SKILLS, default=list(SKILLS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[300, 301, 302, 303, 304])
    parser.add_argument(
        "--context-manifest",
        type=Path,
        default=data_root / "manifests" / "pi05-formal" / "main_validation.json",
        help="frozen validation manifest whose episode initial states define the evaluation contexts",
    )
    parser.add_argument("--data-root", type=Path, default=data_root / "raw")
    parser.add_argument(
        "--context-lock",
        type=Path,
        help="defaults to <context-manifest stem>.lock.json beside the manifest",
    )
    parser.add_argument(
        "--skip-context-integrity",
        action="store_true",
        help="systems dry-runs only; formal evaluation requires the content lock",
    )
    parser.add_argument("--execute-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, help="override the skill-specific limits")
    parser.add_argument("--min-success-rate", type=float, default=0.6)
    parser.add_argument("--max-p95-infer-ms", type=float, default=250.0)
    parser.add_argument("--showcase", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be a non-empty unique list")
    if len(set(args.skills)) != len(args.skills):
        parser.error("--skills must not contain duplicates")
    if args.execute_steps < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("step counts must be positive")
    if args.skip_context_integrity and not args.dry_run:
        parser.error("--skip-context-integrity is permitted only with --dry-run")

    project_root = Path(__file__).resolve().parents[1]
    isaac_runner = project_root / "scripts" / "run_isaac.sh"
    rollout_script = project_root / "scripts" / "run_drawer_pi05_closed_loop.py"
    output_root = args.output_root.expanduser().resolve()
    infrastructure_failures: list[str] = []
    expected_outputs: list[Path] = []
    context_manifest = args.context_manifest.expanduser().resolve()
    context_lock = (
        args.context_lock.expanduser().resolve()
        if args.context_lock
        else context_manifest.with_suffix(".lock.json")
    )
    try:
        if not args.skip_context_integrity:
            validate_context_lock(context_lock, context_manifest, args.data_root)
        contexts = load_contexts(
            context_manifest,
            args.data_root,
            skills=args.skills,
            seeds=args.seeds,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    for skill in args.skills:
        for seed in args.seeds:
            output = output_root / skill / f"seed_{seed}.hdf5"
            expected_outputs.append(output)
            if output.exists() and not args.overwrite and not args.dry_run:
                parser.error(f"evaluation output exists: {output}; pass --overwrite")
            command = [
                str(isaac_runner),
                str(rollout_script),
                "--skill",
                skill,
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--seed",
                str(seed),
                "--initial-state-file",
                str(contexts[(skill, seed)][0]),
                "--initial-state-episode",
                contexts[(skill, seed)][1],
                "--max-steps",
                str(args.max_steps or DEFAULT_MAX_STEPS[skill]),
                "--execute-steps",
                str(args.execute_steps),
                "--output",
                str(output),
                "--device",
                args.device,
                "--enable_cameras",
                "--viz",
                "none",
            ]
            if args.showcase:
                command.append("--showcase")
            print("command:", " ".join(command), flush=True)
            if args.dry_run:
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            if args.overwrite:
                output.unlink(missing_ok=True)
            result = subprocess.run(command, cwd=project_root, check=False)
            if not output.is_file():
                infrastructure_failures.append(f"{skill}/seed_{seed}: exit={result.returncode}, no HDF5")
            elif result.returncode:
                print(f"recorded policy failure: skill={skill} seed={seed}", flush=True)

    report = output_root / "evaluation.json"
    summary_command = [
        str(project_root / "scripts" / "run_openpi.sh"),
        str(project_root / "scripts" / "summarize_pi05_eval.py"),
        "--input-root",
        str(output_root),
        "--output",
        str(report),
        "--skills",
        *args.skills,
        "--min-episodes-per-skill",
        str(len(args.seeds)),
        "--min-success-rate",
        str(args.min_success_rate),
        "--max-p95-infer-ms",
        str(args.max_p95_infer_ms),
        "--context-lock",
        str(context_lock),
    ]
    for output in expected_outputs:
        summary_command.extend(("--episode", str(output)))
    print("summary:", " ".join(summary_command), flush=True)
    if args.dry_run:
        return 0
    if infrastructure_failures:
        raise RuntimeError("evaluation infrastructure failures: " + "; ".join(infrastructure_failures))
    return subprocess.run(summary_command, cwd=project_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
