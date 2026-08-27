#!/usr/bin/env python3
"""Build the historical cost-bounded VLA-TidyBench smoke reel."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS = 1280, 720, 20


def card(title: str, lines: list[str], seconds: float = 3.0) -> list[np.ndarray]:
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 42)
    body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#0e1520")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 18, HEIGHT), fill="#43c7ef")
    draw.text((70, 95), title, font=title_font, fill="#64d8ff")
    y = 190
    for line in lines:
        draw.text((75, y), line, font=body_font, fill="white")
        y += 58
    draw.text((75, 650), "VLA-TidyBench | Isaac Sim + Franka + OpenPI pi0.5", font=body_font, fill="#9caec2")
    frame = np.asarray(image)
    return [frame] * int(seconds * FPS)


def recording_frames(path: Path, *, label: str, outcome: str, limit: int | None = None) -> list[np.ndarray]:
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    with h5py.File(path, "r") as source:
        hero = source["hero_cam"]
        table = source["table_cam"]
        wrist = source["wrist_cam"]
        count = min(len(hero), limit or len(hero))
        output: list[np.ndarray] = []
        for index in range(count):
            canvas = Image.fromarray(hero[index]).convert("RGB")
            canvas.paste(Image.fromarray(table[index]).resize((220, 220)), (815, 455))
            canvas.paste(Image.fromarray(wrist[index]).resize((220, 220)), (1045, 455))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, WIDTH, 78), fill="#0c121c")
            draw.text((20, 9), label, font=title_font, fill="#64d8ff")
            draw.text((20, 45), 'Prompt: "open the top drawer"', font=body_font, fill="white")
            draw.text((1010, 18), outcome, font=title_font, fill="#ffbd55" if "NOT" in outcome else "#50e682")
            draw.rectangle((811, 451, 1039, 679), outline="#64d8ff", width=3)
            draw.rectangle((1041, 451, 1269, 679), outline="#64d8ff", width=3)
            draw.text((820, 680), "table camera", font=body_font, fill="white")
            draw.text((1050, 680), "wrist camera", font=body_font, fill="white")
            output.append(np.asarray(canvas))
    return output


def decode_video(path: Path) -> list[np.ndarray]:
    import imageio.v3 as iio

    frames = []
    for frame in iio.imiter(path, plugin="FFMPEG"):
        image = Image.fromarray(frame).convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        frames.append(np.asarray(image))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--policy-attempt", type=Path, required=True)
    parser.add_argument("--skills", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames: list[np.ndarray] = []
    frames += card(
        "VLA-TidyBench",
        [
            "Simulation-first embodied AI project",
            "Franka drawer manipulation in a furnished Isaac Sim scene",
            "Data -> pi0.5 LoRA -> policy server -> closed-loop deployment",
        ],
    )
    frames += card(
        "Historical minimal-data smoke",
        [
            "8 successful OPEN demonstrations | 1,092 RGB frames",
            "OpenPI pi0.5 LoRA | 500 steps | 2 x RTX 4090",
            "500-step smoke loss: 0.0316 | warm inference: about 96 ms",
        ],
    )
    frames += recording_frames(
        args.policy_attempt,
        label="Real pi0.5 LoRA closed-loop rollout",
        outcome="OPEN GATE NOT PASSED",
        limit=120,
    )
    frames += card(
        "Observed model result",
        [
            "The cost-bounded smoke training and deployment chain executed successfully.",
            "The small-data policy moved near the cabinet but did not establish a stable handle contact.",
            "This unsuccessful rollout is retained instead of being presented as a policy success.",
        ],
    )
    frames += recording_frames(
        args.teacher,
        label="Scripted teacher | scene and camera validation",
        outcome="OPEN SUCCESS",
    )
    frames += card(
        "Verified four-skill suite",
        [
            "OPEN | PICK | PLACE | CLOSE",
            "Truth-guided teacher + task-space waypoints + DLS inverse kinematics",
            "The following reel is a teacher demonstration, not pi0.5 inference.",
        ],
    )
    frames += decode_video(args.skills)
    frames += card(
        "Historical smoke status",
        [
            "Smoke completed: simulator, data, LoRA, policy bridge, closed loop, video",
            "Not a result from the formal three-stage 360-episode experiment",
            "Formal claims require a format-v3 deployment and locked Isaac evaluation",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = os.environ.get("FFMPEG_BIN", "ffmpeg")
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-crf", "20",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"rendered {len(frames)} frames ({len(frames) / FPS:.1f}s) -> {args.output}")


if __name__ == "__main__":
    main()
