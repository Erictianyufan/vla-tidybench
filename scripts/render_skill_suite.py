#!/usr/bin/env python3
"""Render four verified atomic drawer trajectories as an honest demo reel."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SKILLS = (
    ("OPEN", "drawer_open_smoke.hdf5", "Open the top drawer"),
    ("PICK", "drawer_pick_smoke.hdf5", "Pick up the tomato soup can"),
    ("PLACE", "drawer_place_smoke.hdf5", "Put the tomato soup can into the top drawer"),
    ("CLOSE", "drawer_close_smoke.hdf5", "Close the top drawer"),
)


def load_frames(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as dataset:
        episode = dataset["data"][sorted(dataset["data"].keys())[0]]
        obs = episode["obs"]
        table = obs["table_cam"][:][..., :3]
        wrist = obs["wrist_cam"][:][..., :3]
    return np.concatenate((table, wrist), axis=2).astype(np.uint8, copy=False)


def labelled(frames: np.ndarray, skill: str, prompt: str, font: ImageFont.FreeTypeFont) -> list[np.ndarray]:
    output = []
    for frame in frames:
        canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + 64), "#10151d")
        canvas.paste(Image.fromarray(frame), (0, 64))
        draw = ImageDraw.Draw(canvas)
        draw.text((14, 8), f"VLA-TidyBench  |  {skill}", font=font, fill="#64d8ff")
        draw.text((14, 34), prompt, font=font, fill="white")
        output.append(np.asarray(canvas))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    ffmpeg = os.environ.get("FFMPEG_BIN", "ffmpeg")
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 18)
    all_frames: list[np.ndarray] = []
    for skill, filename, prompt in SKILLS:
        frames = load_frames(args.data_root / filename)
        section = labelled(frames, skill, prompt, font)
        all_frames.extend([section[0]] * args.fps)
        all_frames.extend(section)
        all_frames.extend([section[-1]] * (args.fps // 2))
    first = all_frames[0]
    height, width = first.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in all_frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"rendered {len(all_frames)} frames ({len(all_frames) / args.fps:.1f}s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
