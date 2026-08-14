#!/usr/bin/env python3
"""Fail a release if tracked files contain common secrets or large artifacts."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
MAX_TRACKED_BYTES = 25 * 1024 * 1024
MAX_PREVIEW_BYTES = 10 * 1024 * 1024

ALLOWED_DEMO_VIDEOS = {
    Path("docs/media/pi05-dls-recovery-open-success.mp4"),
    Path("docs/media/vla-tidybench-final-project.mp4"),
    Path("docs/media/vla-tidybench-new-scene-preview.mp4"),
}

BLOCKED_SUFFIXES = {
    ".avi",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".key",
    ".mkv",
    ".mov",
    ".mp4",
    ".npz",
    ".orbax",
    ".p12",
    ".pem",
    ".pt",
    ".pth",
    ".safetensors",
    ".tfrecord",
}

SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "assigned credential": re.compile(
        r"(?i)\b(?:password|passwd|api[_-]?key|access[_-]?token|client[_-]?secret)"
        r"\s*[:=]\s*['\"]?[^\s'\"]{8,}"
    ),
    "credential in URL": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
}


def run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def tracked_files() -> list[Path]:
    result = run_git("ls-files", "-z")
    if result.returncode != 0:
        raise SystemExit(f"git ls-files failed: {result.stderr.strip()}")
    return [ROOT / value for value in result.stdout.split("\0") if value]


def inspect_file(path: Path) -> list[str]:
    findings: list[str] = []
    relative = path.relative_to(ROOT)
    suffix = path.suffix.lower()
    size = path.stat().st_size

    if suffix in BLOCKED_SUFFIXES and relative not in ALLOWED_DEMO_VIDEOS:
        findings.append(f"blocked artifact type: {relative}")
    if size > MAX_TRACKED_BYTES:
        findings.append(f"tracked file exceeds 25 MiB: {relative} ({size} bytes)")
    if suffix == ".gif" and size > MAX_PREVIEW_BYTES:
        findings.append(f"preview GIF exceeds 10 MiB: {relative} ({size} bytes)")
    if path.name.startswith(".env") and path.name != ".env.example":
        findings.append(f"environment file is tracked: {relative}")

    if path.resolve() == SELF or size > MAX_TRACKED_BYTES:
        return findings

    try:
        raw = path.read_bytes()
        if b"\0" in raw:
            return findings
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return findings

    for label, pattern in SECRET_PATTERNS.items():
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            findings.append(f"possible {label}: {relative}:{line}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()

    findings: list[str] = []
    for path in tracked_files():
        if path.is_file():
            findings.extend(inspect_file(path))

    remote = run_git("remote", "-v")
    if re.search(r"https?://[^\s/:]+:[^\s/@]+@", remote.stdout):
        findings.append("a Git remote contains embedded credentials")

    if args.require_clean:
        status = run_git("status", "--porcelain")
        if status.stdout.strip():
            findings.append("Git worktree is not clean")

    if findings:
        print("Pre-publication audit FAILED:")
        for finding in findings:
            print(f"- {finding}")
        return 1

    print("Pre-publication audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
