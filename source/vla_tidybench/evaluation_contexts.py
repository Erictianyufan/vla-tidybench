"""Content locks for held-out simulator context files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_context_lock(manifest_path: Path, data_root: Path) -> dict[str, object]:
    manifest_path = manifest_path.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != 1 or manifest.get("split") != "validation":
        raise ValueError(f"expected a schema-1 validation manifest: {manifest_path}")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("validation manifest must contain source files")
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("validation manifest sources must be objects")
        relative = str(source.get("file", ""))
        path = (data_root / relative).resolve()
        if not relative or not path.is_relative_to(data_root) or not path.is_file():
            raise ValueError(f"invalid or missing validation context source: {path}")
        if relative in seen:
            raise ValueError(f"duplicate validation context source: {relative}")
        seen.add(relative)
        indices = [int(index) for index in source.get("episode_indices", [])]
        if not indices or len(set(indices)) != len(indices) or any(index < 0 for index in indices):
            raise ValueError(f"invalid validation episode indices for {relative}")
        records.append(
            {
                "file": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "episode_indices": indices,
            }
        )
    return {
        "schema_version": 1,
        "context_manifest": manifest_path.name,
        "context_manifest_sha256": sha256_file(manifest_path),
        "context_count": sum(len(record["episode_indices"]) for record in records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "sources": records,
    }


def validate_context_lock(lock_path: Path, manifest_path: Path, data_root: Path) -> dict[str, object]:
    lock_path = lock_path.expanduser().resolve()
    try:
        recorded = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read evaluation context lock {lock_path}: {error}") from error
    if not isinstance(recorded, dict):
        raise ValueError(f"evaluation context lock must be a JSON object: {lock_path}")
    actual = build_context_lock(manifest_path, data_root)
    if recorded != actual:
        raise ValueError(f"evaluation context lock does not match manifest/data content: {lock_path}")
    return actual
