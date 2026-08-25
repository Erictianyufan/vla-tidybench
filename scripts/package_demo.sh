#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/raw-demo.mp4" >&2
  exit 2
fi

input=$1
if [[ ! -f "$input" ]]; then
  echo "input video does not exist: $input" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
artifact_dir="$repo_root/artifacts/demo"
media_dir="$repo_root/docs/media"
output_mp4="$artifact_dir/vla-tidybench-demo.mp4"
preview_gif="$media_dir/demo-preview.gif"

mkdir -p "$artifact_dir" "$media_dir"

if [[ -n "${FFMPEG_BIN:-}" ]]; then
  ffmpeg_bin=$FFMPEG_BIN
elif command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg_bin=$(command -v ffmpeg)
elif [[ -x "${OPENPI_ROOT:-${HOME}/openpi}/.venv/bin/python" ]]; then
  ffmpeg_bin=$(
    "${OPENPI_ROOT:-${HOME}/openpi}/.venv/bin/python" -c \
      'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'
  )
else
  echo "ffmpeg was not found; install it or set FFMPEG_BIN" >&2
  exit 1
fi

"$ffmpeg_bin" -hide_banner -loglevel warning -y -i "$input" \
  -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2" \
  -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart -an \
  "$output_mp4"

preview_start=${PREVIEW_START:-00:00:20}
preview_duration=${PREVIEW_DURATION:-12}
"$ffmpeg_bin" -hide_banner -loglevel warning -y -ss "$preview_start" -t "$preview_duration" \
  -i "$output_mp4" \
  -vf "fps=8,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer" \
  "$preview_gif"

(
  cd "$artifact_dir"
  sha256sum "$(basename "$output_mp4")" > SHA256SUMS
)

echo "release video: $output_mp4"
echo "README preview: $preview_gif"
echo "checksums: $artifact_dir/SHA256SUMS"
