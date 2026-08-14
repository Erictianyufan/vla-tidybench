#!/usr/bin/env python3
"""Validate low-cost continuation contracts for Mimic, OOD and residual RL."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> int:
    mimic = load("configs/mimic/drawer_smoke.json")
    ood = load("configs/eval/drawer_ood_smoke.json")
    rl = load("configs/rl/open_residual_sac.json")
    pick_rl = load("configs/rl/pick_residual_sac.json")

    assert mimic["generation_num_trials"] > 0 and mimic["num_envs"] > 0
    assert mimic["export_success_only"] is True
    seeds = [seed for bucket in ood["buckets"].values() for seed in bucket["seeds"]]
    assert len(seeds) == len(set(seeds))
    assert rl["base_policy_frozen"] is True
    assert rl["actor_privileged_fields"] == []
    assert rl["residual"]["dimensions"] == 6
    assert rl["residual"]["gripper_residual"] is False
    assert rl["release_gate"]["fallback"] == rl["base_policy"]
    assert pick_rl["skill"] == "pick_tomato_soup_can"
    assert pick_rl["base_controller_frozen"] is True
    assert pick_rl["actor_privileged_fields"] == []
    assert pick_rl["residual"]["dimensions"] == 1
    assert pick_rl["residual"]["adapter_output_dimensions"] == 6
    assert pick_rl["residual"]["gripper_residual"] is False
    print(
        f"extension contracts passed: Mimic={mimic['generation_num_trials']} trials, "
        f"OOD={len(seeds)} smoke episodes, RL=frozen-VLA residual"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
