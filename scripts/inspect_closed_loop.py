#!/usr/bin/env python3
"""Create a contact sheet and action summary from a closed-loop recording."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with h5py.File(args.recording, "r") as source:
        if "hero_cam" in source:
            hero = source["hero_cam"]
            ids = np.linspace(0, len(hero) - 1, 5, dtype=int)
            canvas = Image.new("RGB", (640 * len(ids), 400), "#10151d")
            draw = ImageDraw.Draw(canvas)
            for column, index in enumerate(ids):
                frame = Image.fromarray(hero[index]).resize((640, 360), Image.Resampling.LANCZOS)
                canvas.paste(frame, (column * 640, 40))
                draw.text((column * 640 + 6, 8), f"hero step {index}", fill="white")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(args.output)
            print(args.output)
            return
        table = source["table_cam"]
        wrist = source["wrist_cam"]
        actions = np.asarray(source["actions"])
        ids = np.linspace(0, len(table) - 1, 5, dtype=int)
        canvas = Image.new("RGB", (400 * len(ids), 240), "#10151d")
        draw = ImageDraw.Draw(canvas)
        for column, index in enumerate(ids):
            pair = np.concatenate((table[index], wrist[index]), axis=1)
            canvas.paste(Image.fromarray(pair), (column * 400, 40))
            draw.text((column * 400 + 6, 8), f"step {index}", fill="white")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print("mean", actions.mean(axis=0).tolist())
    print("min", actions.min(axis=0).tolist())
    print("max", actions.max(axis=0).tolist())
    print(args.output)


if __name__ == "__main__":
    main()
