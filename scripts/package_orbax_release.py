"""Package a complete Orbax checkpoint as GitHub-Release-sized tar parts.

Each part is below GitHub's per-asset size limit. Reassemble in part-number order
before extracting; the emitted JSON records the SHA-256 of every part and file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


class SplitWriter:
    def __init__(self, output_dir: Path, prefix: str, part_size: int) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.part_size = part_size
        self.parts: list[dict[str, object]] = []
        self._file = None
        self._hash = hashlib.sha256()
        self._size = 0

    def _open_part(self) -> None:
        name = f"{self.prefix}.tar.part{len(self.parts) + 1:02d}"
        self._file = (self.output_dir / name).open("xb")
        self._hash = hashlib.sha256()
        self._size = 0
        self.parts.append({"name": name})

    def _close_part(self) -> None:
        if self._file is None:
            return
        self._file.close()
        self.parts[-1]["size"] = self._size
        self.parts[-1]["sha256"] = self._hash.hexdigest()
        self._file = None

    def write(self, data: bytes) -> int:
        view = memoryview(data)
        original_size = len(view)
        while view:
            if self._file is None:
                self._open_part()
            count = min(len(view), self.part_size - self._size)
            chunk = view[:count]
            self._file.write(chunk)
            self._hash.update(chunk)
            self._size += count
            view = view[count:]
            if self._size == self.part_size:
                self._close_part()
        return original_size

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        self._close_part()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(checkpoint: Path, output_dir: Path, part_size: int) -> Path:
    if not checkpoint.is_dir():
        raise ValueError(f"Not a checkpoint directory: {checkpoint}")
    if part_size <= 0 or part_size >= 2_000_000_000:
        raise ValueError("Part size must be between 1 and 1,999,999,999 bytes")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"pi05-home-expert-step-{checkpoint.name}"
    if any(output_dir.glob(f"{prefix}.tar.part*")):
        raise FileExistsError("Target already contains checkpoint parts; refusing to overwrite")

    files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
    if not files:
        raise ValueError("Checkpoint contains no files")
    writer = SplitWriter(output_dir, prefix, part_size)
    try:
        with tarfile.open(fileobj=writer, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            archive.add(checkpoint, arcname=checkpoint.name, recursive=False)
            for path in files:
                name = f"{checkpoint.name}/{path.relative_to(checkpoint).as_posix()}"
                archive.add(path, arcname=name, recursive=False)
    finally:
        writer.close()

    manifest = {
        "format": "split-tar-v1",
        "checkpoint_directory": checkpoint.name,
        "part_size_limit": part_size,
        "parts": writer.parts,
        "files": [
            {
                "path": path.relative_to(checkpoint).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }
    manifest_path = output_dir / f"{prefix}.manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--part-size", type=int, default=1_800_000_000)
    args = parser.parse_args()
    print(package(args.checkpoint, args.output_dir, args.part_size))


if __name__ == "__main__":
    main()
