#!/usr/bin/env python3
"""Render a lightweight, same-name GIF preview for every demo MP4."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


# Long videos use fewer frames and colors so every README preview stays below
# the repository's 10 MiB publication limit.
PROFILES: dict[str, tuple[int, int, int]] = {
    "pi05-dls-recovery-open-success": (8, 640, 96),
    "pi05-four-skill-minimal-success": (6, 600, 80),
    "pick-residual-sac-demo": (8, 640, 96),
    "pi05-continuous-medicine-demo": (6, 600, 80),
    "vla-tidybench-final-project": (4, 540, 64),
    "vla-tidybench-new-scene-preview": (8, 640, 96),
}


def ffmpeg_executable() -> str:
    configured = os.environ.get("FFMPEG_BIN")
    if configured:
        return configured

    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError("install imageio-ffmpeg or set FFMPEG_BIN") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def render_preview(ffmpeg: str, video: Path, output: Path) -> None:
    fps, width, colors = PROFILES.get(video.stem, (6, 600, 80))
    filter_graph = (
        f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];"
        f"[s0]palettegen=max_colors={colors}:stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
    )
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-filter_complex",
            filter_graph,
            "-loop",
            "0",
            str(output),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-dir", type=Path, default=Path("docs/media"))
    args = parser.parse_args()

    videos = sorted(args.media_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no MP4 files found in {args.media_dir}")

    ffmpeg = ffmpeg_executable()
    for video in videos:
        output = video.with_suffix(".gif")
        render_preview(ffmpeg, video, output)
        print(f"{video.name} -> {output.name} ({output.stat().st_size / 1024**2:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
