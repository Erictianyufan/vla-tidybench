from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest
from vla_tidybench.evaluation_contexts import build_context_lock, validate_context_lock


def write_manifest(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    with h5py.File(raw / "drawer_open_formal.hdf5", "w") as dataset:
        dataset.attrs["format_version"] = 1
        data = dataset.create_group("data")
        for index in range(12):
            data.create_group(f"demo_{index}")
    manifest = tmp_path / "main_validation.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split": "validation",
                "sources": [
                    {
                        "file": "drawer_open_formal.hdf5",
                        "episode_indices": [3, 7, 11],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, raw


def test_context_lock_is_portable_and_content_bound(tmp_path: Path) -> None:
    manifest, raw = write_manifest(tmp_path)
    lock = build_context_lock(manifest, raw)
    lock_path = tmp_path / "main_validation.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    validated = validate_context_lock(lock_path, manifest, raw)

    assert validated["context_manifest"] == manifest.name
    assert validated["context_count"] == 3
    assert validated["total_bytes"] > 0
    assert validated["sources"][0]["file"] == "drawer_open_formal.hdf5"
    assert validated["sources"][0]["episode_names"] == ["demo_3", "demo_7", "demo_11"]


def test_context_lock_rejects_same_path_after_content_mutation(tmp_path: Path) -> None:
    manifest, raw = write_manifest(tmp_path)
    lock_path = tmp_path / "main_validation.lock.json"
    lock_path.write_text(json.dumps(build_context_lock(manifest, raw)), encoding="utf-8")
    with (raw / "drawer_open_formal.hdf5").open("ab") as output:
        output.write(b"tampered-context")

    with pytest.raises(ValueError, match="does not match"):
        validate_context_lock(lock_path, manifest, raw)


def test_context_lock_rejects_manifest_path_escape(tmp_path: Path) -> None:
    manifest, raw = write_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["sources"][0]["file"] = "../outside.hdf5"
    (tmp_path / "outside.hdf5").touch()
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid or missing"):
        build_context_lock(manifest, raw)
