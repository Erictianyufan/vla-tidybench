from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plan_pi05_data_splits", ROOT / "scripts/plan_pi05_data_splits.py")
assert SPEC is not None and SPEC.loader is not None
planning = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = planning
SPEC.loader.exec_module(planning)


def write_source(path: Path, count: int, *, unsuccessful: int | None = None) -> None:
    with h5py.File(path, "w") as output:
        output.attrs["format_version"] = 1
        data = output.create_group("data")
        for index in range(count):
            episode = data.create_group(f"demo_{index}")
            episode.attrs["success"] = index != unsuccessful
            episode.create_dataset("actions", shape=(1, 7), dtype="f4")
            obs = episode.create_group("obs")
            obs.create_dataset("table_cam", shape=(1, 200, 200, 3), dtype="u1")
            obs.create_dataset("wrist_cam", shape=(1, 200, 200, 3), dtype="u1")
            obs.create_dataset("joint_pos", shape=(1, 9), dtype="f4")
            obs.create_dataset("joint_vel", shape=(1, 9), dtype="f4")


def write_config(path: Path, sources: list[tuple[str, str]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo_id": "local/unused",
                "fps": 20,
                "sources": [{"file": file, "prompt": prompt} for file, prompt in sources],
            }
        ),
        encoding="utf-8",
    )


def test_episode_split_and_hard_replay_mix_are_disjoint_and_deterministic(tmp_path: Path) -> None:
    prompts = ("open the top drawer", "close the top drawer")
    main_sources = []
    hard_sources = []
    for index, prompt in enumerate(prompts):
        main_file = f"main_{index}.hdf5"
        hard_file = f"hard_{index}.hdf5"
        write_source(tmp_path / main_file, 10)
        write_source(tmp_path / hard_file, 5)
        main_sources.append((main_file, prompt))
        hard_sources.append((hard_file, prompt))
    main_config = tmp_path / "main.json"
    hard_config = tmp_path / "hard.json"
    write_config(main_config, main_sources)
    write_config(hard_config, hard_sources)

    _, nominal = planning.load_source_config(main_config, tmp_path, role="nominal")
    _, hard = planning.load_source_config(hard_config, tmp_path, role="hard_recovery")
    train, validation = planning.split_nominal(
        nominal,
        seed=2026,
        validation_fraction=0.2,
        min_train_per_prompt=8,
        min_validation_per_prompt=2,
    )
    repeated = planning.split_nominal(
        nominal,
        seed=2026,
        validation_fraction=0.2,
        min_train_per_prompt=8,
        min_validation_per_prompt=2,
    )
    hard_train, hard_validation = planning.split_nominal(
        hard,
        seed=2026,
        validation_fraction=0.2,
        min_train_per_prompt=4,
        min_validation_per_prompt=1,
        namespace="hard_recovery",
    )
    mixed = planning.hard_mix(
        train,
        hard_train,
        seed=2026,
        nominal_replay_ratio=1.0,
        min_hard_per_prompt=2,
    )

    assert [episode.key for episode in train] == [episode.key for episode in repeated[0]]
    assert not ({episode.key for episode in train} & {episode.key for episode in validation})
    assert len(train) == 16
    assert len(validation) == 4
    assert len(hard_train) == 8
    assert len(hard_validation) == 2
    assert not ({episode.key for episode in hard_train} & {episode.key for episode in hard_validation})
    assert len(mixed) == 16
    assert sum(episode.role == "hard_recovery" for episode in mixed) == 8
    assert sum(episode.role == "nominal" for episode in mixed) == 8


def test_unsuccessful_episode_cannot_enter_supervised_manifest(tmp_path: Path) -> None:
    write_source(tmp_path / "failed.hdf5", 2, unsuccessful=1)
    config = tmp_path / "failed.json"
    write_config(config, [("failed.hdf5", "open the top drawer")])
    try:
        planning.load_source_config(config, tmp_path, role="hard_recovery")
    except ValueError as error:
        assert "successful/recovered episodes only" in str(error)
    else:
        raise AssertionError("unsuccessful episode was accepted")


def test_formal_prompts_match_the_medicine_bottle_scene() -> None:
    expected = {
        "open the top drawer",
        "pick up the medicine bottle",
        "put the medicine bottle into the top drawer",
        "close the top drawer",
    }
    for name in ("drawer_four_skill_formal.json", "drawer_four_skill_hard_recovery.json"):
        config = json.loads((ROOT / "configs" / "data" / name).read_text(encoding="utf-8"))
        assert {source["prompt"] for source in config["sources"]} == expected
