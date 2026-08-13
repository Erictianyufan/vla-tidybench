#!/usr/bin/env python3
"""Render a two-camera preview from any pi0.5 closed-loop recording."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    with h5py.File(args.recording, "r") as source:
        table = np.asarray(source["table_cam"])
        wrist = np.asarray(source["wrist_cam"])
        success = bool(source.attrs["success"])
        policy = str(source.attrs["policy"])
        prompt = str(source.attrs["prompt"])
    title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    frames: list[np.ndarray] = []
    for index in range(len(table)):
        canvas = Image.new("RGB", (1280, 720), "#10151d")
        left = Image.fromarray(table[index]).resize((580, 580), Image.Resampling.NEAREST)
        right = Image.fromarray(wrist[index]).resize((580, 580), Image.Resampling.NEAREST)
        canvas.paste(left, (40, 100))
        canvas.paste(right, (660, 100))
        draw = ImageDraw.Draw(canvas)
        draw.text((40, 18), "VLA-TidyBench | Real pi0.5 LoRA closed-loop preview", font=title, fill=(96, 216, 255))
        draw.text((40, 55), f'Prompt: "{prompt}"', font=body, fill="white")
        draw.text((1030, 20), "SUCCESS" if success else "ATTEMPT", font=title,
                  fill=(80, 230, 130) if success else (255, 190, 80))
        draw.text((45, 680), "table policy camera", font=body, fill="white")
        draw.text((665, 680), f"wrist policy camera | {policy}", font=body, fill="white")
        frames.append(np.asarray(canvas))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = os.environ.get("FFMPEG_BIN", "ffmpeg")
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "1280x720", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "20",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"rendered {len(frames)} frames -> {args.output}")


if __name__ == "__main__":
    main()
