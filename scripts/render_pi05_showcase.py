#!/usr/bin/env python3
"""Render a multi-camera pi0.5 closed-loop recording into a demo MP4."""

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
        hero = np.asarray(source["hero_cam"])
        table = np.asarray(source["table_cam"])
        wrist = np.asarray(source["wrist_cam"])
        success = bool(source.attrs["success"])
        policy = str(source.attrs["policy"])
        prompt = str(source.attrs["prompt"])
        mean_infer = float(source.attrs["mean_infer_ms"])
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    frames: list[np.ndarray] = []
    for index in range(len(hero)):
        canvas = Image.fromarray(hero[index]).convert("RGB")
        table_image = Image.fromarray(table[index]).resize((220, 220), Image.Resampling.LANCZOS)
        wrist_image = Image.fromarray(wrist[index]).resize((220, 220), Image.Resampling.LANCZOS)
        # Keep the hero view unobstructed around the Franka and cabinet. The
        # two policy-camera insets occupy the otherwise empty right margin.
        canvas.paste(table_image, (815, 455))
        canvas.paste(wrist_image, (1045, 455))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 1280, 72), fill=(12, 18, 28))
        is_preview = policy == "scripted-teacher-camera-preview"
        heading = "VLA-TidyBench | New scene + three-camera preview" if is_preview else "VLA-TidyBench | Real pi0.5 LoRA closed loop"
        draw.text((22, 10), heading, font=font, fill=(96, 216, 255))
        draw.text((22, 42), f'Prompt: "{prompt}"', font=small, fill="white")
        draw.text((1000, 16), "SUCCESS" if success else "ATTEMPT", font=font, fill=(80, 230, 130) if success else (255, 190, 80))
        draw.rectangle((811, 451, 1039, 679), outline=(96, 216, 255), width=3)
        draw.rectangle((1041, 451, 1269, 679), outline=(96, 216, 255), width=3)
        draw.text((820, 680), "table camera", font=small, fill="white")
        draw.text((1050, 680), "wrist camera", font=small, fill="white")
        footer = policy if is_preview else f"warm inference ~{mean_infer:.0f} ms | {policy}"
        draw.text((760, 78), footer, font=small, fill="white")
        frames.append(np.asarray(canvas))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = os.environ.get("FFMPEG_BIN", "ffmpeg")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1280x720", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
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
