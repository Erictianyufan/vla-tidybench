from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_drawer_manifest_has_four_atomic_skills() -> None:
    config = json.loads((ROOT / "configs/data/drawer_m2_smoke.json").read_text(encoding="utf-8"))
    assert config["fps"] == 20
    assert len(config["sources"]) == 4
    prompts = {source["prompt"] for source in config["sources"]}
    assert prompts == {
        "open the top drawer",
        "pick up the red object",
        "put the red object into the top drawer",
        "close the top drawer",
    }
    assert all(source["episode_indices"] == [0] for source in config["sources"])
