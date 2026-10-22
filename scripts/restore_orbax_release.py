"""Verify and restore the split-tar Orbax checkpoint from a release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class JoinedParts:
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self._index = 0
        self._file = None

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            raise ValueError("Streamed tar reader requires a finite read size")
        chunks = []
        remaining = size
        while remaining and self._index < len(self.paths):
            if self._file is None:
                self._file = self.paths[self._index].open("rb")
            chunk = self._file.read(remaining)
            if chunk:
                chunks.append(chunk)
                remaining -= len(chunk)
            else:
                self._file.close()
                self._file = None
                self._index += 1
        return b"".join(chunks)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def restore(manifest_path: Path, destination: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "split-tar-v1":
        raise ValueError("Unsupported checkpoint archive format")
    checkpoint_name = manifest["checkpoint_directory"]
    if PurePosixPath(checkpoint_name).name != checkpoint_name:
        raise ValueError("Unsafe checkpoint directory name")
    target = destination / checkpoint_name
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    parts = []
    for record in manifest["parts"]:
        name = record["name"]
        if PurePosixPath(name).name != name:
            raise ValueError("Unsafe part name")
        path = manifest_path.parent / name
        if path.stat().st_size != record["size"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Part integrity check failed: {name}")
        parts.append(path)

    destination.mkdir(parents=True, exist_ok=True)
    reader = JoinedParts(parts)
    try:
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not relative.parts
                    or relative.parts[0] != checkpoint_name
                ):
                    raise ValueError(f"Unsafe tar member: {member.name}")
                path = destination.joinpath(*relative.parts)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as output:
                        source = archive.extractfile(member)
                        if source is None:
                            raise ValueError(f"Missing tar member body: {member.name}")
                        shutil.copyfileobj(source, output)
                else:
                    raise ValueError(f"Unsupported tar member type: {member.name}")
    finally:
        reader.close()

    for record in manifest["files"]:
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest file path")
        path = target.joinpath(*relative.parts)
        if path.stat().st_size != record["size"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Restored checkpoint verification failed: {path}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(restore(args.manifest, args.destination))


if __name__ == "__main__":
    main()
