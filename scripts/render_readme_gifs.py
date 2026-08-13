#!/usr/bin/env python3
"""Create the README 2x2 skill GIF and a compact final-video GIF."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SKILLS = ("OPEN", "PICK", "PLACE", "CLOSE")
PROMPTS = {
    "OPEN": "Open the top drawer",
    "PICK": "Pick up the red object",
    "PLACE": "Put the object in the drawer",
    "CLOSE": "Close the top drawer",
}


def panel(hero: np.ndarray, table: np.ndarray, wrist: np.ndarray, skill: str) -> Image.Image:
    canvas = Image.fromarray(hero).convert("RGB").resize((480, 270), Image.Resampling.LANCZOS)
    table_im = Image.fromarray(table).resize((92, 92), Image.Resampling.LANCZOS)
    wrist_im = Image.fromarray(wrist).resize((92, 92), Image.Resampling.LANCZOS)
    canvas.paste(table_im, (282, 167))
    canvas.paste(wrist_im, (378, 167))
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 21)
    body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    draw.rectangle((0, 0, 480, 54), fill=(10, 16, 25))
    draw.text((12, 5), skill, font=title, fill=(96, 216, 255))
    draw.text((12, 31), PROMPTS[skill], font=body, fill="white")
    draw.rectangle((279, 164, 377, 262), outline=(96, 216, 255), width=2)
    draw.rectangle((375, 164, 473, 262), outline=(96, 216, 255), width=2)
    return canvas


def skill_grid(paths: list[Path], output: Path, frames: int = 48) -> None:
    sources = [h5py.File(path, "r") for path in paths]
    try:
        output_frames = []
        for frame_id in range(frames):
            grid = Image.new("RGB", (960, 540), "#0b111a")
            for slot, (skill, source) in enumerate(zip(SKILLS, sources, strict=True)):
                length = len(source["hero_cam"])
                index = min(round(frame_id * (length - 1) / max(frames - 1, 1)), length - 1)
                view = panel(source["hero_cam"][index], source["table_cam"][index], source["wrist_cam"][index], skill)
                grid.paste(view, ((slot % 2) * 480, (slot // 2) * 270))
            output_frames.append(grid.quantize(colors=96, method=Image.Quantize.MEDIANCUT))
        output.parent.mkdir(parents=True, exist_ok=True)
        output_frames[0].save(output, save_all=True, append_images=output_frames[1:], duration=167, loop=0, optimize=True, disposal=2)
    finally:
        for source in sources:
            source.close()


def final_video_gif(video: Path, output: Path, frames: int = 72) -> None:
    meta = iio.immeta(video, plugin="FFMPEG")
    raw_total = float(meta.get("nframes", 0))
    total = int(raw_total) if math.isfinite(raw_total) else 0
    if total <= 0 or total > 1000000:
        total = int(float(meta["duration"]) * float(meta["fps"]))
    indices = np.linspace(0, total - 1, frames, dtype=int)
    images = []
    for index in indices:
        frame = iio.imread(video, index=int(index), plugin="FFMPEG")
        image = Image.fromarray(frame).convert("RGB").resize((800, 450), Image.Resampling.LANCZOS)
        images.append(image.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:], duration=167, loop=0, optimize=True, disposal=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", type=Path, required=True)
    parser.add_argument("--pick", type=Path, required=True)
    parser.add_argument("--place", type=Path, required=True)
    parser.add_argument("--close", type=Path, required=True)
    parser.add_argument("--final-video", type=Path, required=True)
    parser.add_argument("--skill-gif", type=Path, required=True)
    parser.add_argument("--final-gif", type=Path, required=True)
    args = parser.parse_args()
    skill_grid([args.open, args.pick, args.place, args.close], args.skill_gif)
    final_video_gif(args.final_video, args.final_gif)
    print(args.skill_gif)
    print(args.final_gif)


if __name__ == "__main__":
    main()
