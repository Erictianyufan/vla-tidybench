import json
import runpy
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
package = runpy.run_path(str(_SCRIPTS / "package_orbax_release.py"))["package"]
restore = runpy.run_path(str(_SCRIPTS / "restore_orbax_release.py"))["restore"]


def test_split_checkpoint_round_trip_and_integrity(tmp_path) -> None:
    checkpoint = tmp_path / "1000"
    nested = checkpoint / "params" / "ocdbt.process_0"
    nested.mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("metadata", encoding="utf-8")
    (nested / "weights").write_bytes(bytes(range(256)) * 3)

    release = tmp_path / "release"
    manifest_path = package(checkpoint, release, part_size=512)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["parts"]) > 1
    assert all(part["size"] <= 512 for part in manifest["parts"])

    restored = restore(manifest_path, tmp_path / "restored")
    assert (restored / "_CHECKPOINT_METADATA").read_text(encoding="utf-8") == "metadata"
    assert (restored / "params" / "ocdbt.process_0" / "weights").read_bytes() == bytes(range(256)) * 3

    first_part = release / manifest["parts"][0]["name"]
    first_part.write_bytes(first_part.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="integrity check failed"):
        restore(manifest_path, tmp_path / "tampered")
