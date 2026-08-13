#!/usr/bin/env python3
"""Render the two recorded RGB observations from an Isaac HDF5 episode to MP4."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    with h5py.File(args.dataset, "r") as dataset:
        episode_name = sorted(dataset["data"].keys())[0]
        obs = dataset["data"][episode_name]["obs"]
        table = obs["table_cam"][:]
        wrist = obs["wrist_cam"][:]
    if table.shape != wrist.shape or table.ndim != 4 or table.shape[-1] not in (3, 4):
        raise ValueError(f"unexpected camera shapes: table={table.shape}, wrist={wrist.shape}")
    frames = np.concatenate((table[..., :3], wrist[..., :3]), axis=2).astype(np.uint8, copy=False)
    height, width = frames.shape[1:3]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    process.stdin.write(frames.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"rendered {len(frames)} frames ({len(frames) / args.fps:.1f}s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
