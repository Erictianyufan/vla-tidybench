import pytest

from vla_tidybench.task_graph import DrawerSkill, DrawerTaskGraph


def test_success_path_is_open_pick_place_close() -> None:
    graph = DrawerTaskGraph()
    seen = []
    while graph.current is not DrawerSkill.DONE:
        seen.append((graph.current.value, graph.prompt))
        graph.update(succeeded=True)
    assert [skill for skill, _ in seen] == ["open", "pick", "place", "close"]


def test_failure_is_terminal_and_resettable() -> None:
    graph = DrawerTaskGraph()
    assert graph.update(succeeded=False, exhausted=True) is DrawerSkill.FAILED
    assert graph.update(succeeded=True) is DrawerSkill.FAILED
    with pytest.raises(RuntimeError):
        _ = graph.prompt
    graph.reset()
    assert graph.current is DrawerSkill.OPEN
