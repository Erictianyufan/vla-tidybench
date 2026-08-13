from pathlib import Path


def test_scripted_collector_uses_canonical_action_and_separate_dataset():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "collect_scripted_stack.py").read_text(encoding="utf-8")
    launcher = (root / "scripts" / "collect_scripted_stack.sh").read_text(encoding="utf-8")

    assert "torch.zeros((self.env.num_envs, 7)" in source
    assert "action[:, 6] = gripper" in source
    assert "env.sim.reset" not in source
    assert "stack_scripted.hdf5" in launcher
    assert "stack_human.hdf5" not in launcher


def test_merge_uses_explicit_episode_selection_and_preserves_provenance():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "merge_stack_datasets.py").read_text(encoding="utf-8")

    assert 'target_episode.attrs["source_file"]' in source
    assert 'target_episode.attrs["source_episode"]' in source
    assert "refusing to merge unsuccessful episode" in source
    assert 'destination.attrs["format_version"]' in source
