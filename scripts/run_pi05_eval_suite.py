#!/usr/bin/env python3
"""Run a deterministic four-skill Isaac closed-loop suite and summarize its deployment gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


SKILLS = ("open", "pick", "place", "close")
DEFAULT_MAX_STEPS = {"open": 360, "pick": 300, "place": 420, "close": 300}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skills", nargs="+", choices=SKILLS, default=list(SKILLS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[300, 301, 302, 303, 304])
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
    if args.execute_steps < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("step counts must be positive")

    project_root = Path(__file__).resolve().parents[1]
    isaac_runner = project_root / "scripts" / "run_isaac.sh"
    rollout_script = project_root / "scripts" / "run_drawer_pi05_closed_loop.py"
    output_root = args.output_root.expanduser().resolve()
    infrastructure_failures: list[str] = []

    for skill in args.skills:
        for seed in args.seeds:
            output = output_root / skill / f"seed_{seed}.hdf5"
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
    ]
    print("summary:", " ".join(summary_command), flush=True)
    if args.dry_run:
        return 0
    if infrastructure_failures:
        raise RuntimeError("evaluation infrastructure failures: " + "; ".join(infrastructure_failures))
    return subprocess.run(summary_command, cwd=project_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
