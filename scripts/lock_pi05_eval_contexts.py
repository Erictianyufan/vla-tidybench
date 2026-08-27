#!/usr/bin/env python3
"""Generate a deterministic content lock for formal held-out Isaac contexts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_tidybench.evaluation_contexts import build_context_lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    output = args.output or args.manifest.with_suffix(".lock.json")
    output = output.expanduser().resolve()
    if output.exists() and not args.replace:
        parser.error(f"context lock already exists: {output}; pass --replace")
    try:
        lock = build_context_lock(args.manifest, args.data_root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**lock, "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
