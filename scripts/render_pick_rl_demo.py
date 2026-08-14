#!/usr/bin/env python3
"""Render baseline and residual-SAC PICK rollouts into a concise demo."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS = 1280, 720, 20


def rollout_frames(path: Path, heading: str, accent: str) -> list[np.ndarray]:
    title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25)
    body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    frames: list[np.ndarray] = []
    with h5py.File(path, "r") as source:
        hero = source["hero_cam"]
        table = source["table_cam"]
        wrist = source["wrist_cam"]
        target = source["target_positions"]
        rewards = source["rewards"]
        success = bool(source.attrs["success"])
        bias = float(source.attrs["calibration_bias"])
        for index in range(len(hero)):
            canvas = Image.fromarray(hero[index]).convert("RGB")
            canvas.paste(Image.fromarray(table[index]).resize((220, 220)), (815, 455))
            canvas.paste(Image.fromarray(wrist[index]).resize((220, 220)), (1045, 455))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, WIDTH, 82), fill="#0c121c")
            draw.text((20, 10), heading, font=title, fill=accent)
            draw.text((20, 47), 'Task: "pick up the YCB tomato soup can"', font=body, fill="white")
            draw.text((920, 12), "SUCCESS" if success else "BASELINE FAIL", font=title,
                      fill="#50e682" if success else "#ffbd55")
            draw.text((890, 49), f"bias={bias:.3f} | z={target[index, 2]:.3f} m | r={rewards[index]:+.2f}",
                      font=body, fill="white")
            draw.rectangle((811, 451, 1039, 679), outline=accent, width=3)
            draw.rectangle((1041, 451, 1269, 679), outline=accent, width=3)
            draw.text((820, 680), "table camera", font=body, fill="white")
            draw.text((1050, 680), "wrist camera", font=body, fill="white")
            frames.append(np.asarray(canvas))
    return frames


def card() -> list[np.ndarray]:
    title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 42)
    body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#0e1520")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 18, HEIGHT), fill="#43c7ef")
    draw.text((70, 100), "PICK residual-RL experiment", font=title, fill="#64d8ff")
    lines = [
        "Scenario: recover dropped groceries beside a home cabinet",
        "Frozen pi0.5 proposal + DLS nominal controller",
        "Residual SAC learns to recover a fixed calibration bias",
        "Actor input has no object pose; simulator truth is reward-only",
    ]
    for index, line in enumerate(lines):
        draw.text((75, 215 + index * 62), line, font=body, fill="white")
    return [np.asarray(image)] * (2 * FPS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = card()
    frames += rollout_frames(args.baseline, "Zero-residual baseline | calibration fault", "#ffbd55")
    frames += rollout_frames(args.rl, "Frozen VLA + DLS + Residual SAC", "#64d8ff")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = os.environ.get("FFMPEG_BIN", "ffmpeg")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"rendered {len(frames)} frames -> {args.output}")


if __name__ == "__main__":
    main()
