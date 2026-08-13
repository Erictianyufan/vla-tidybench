import numpy as np
from vla_tidybench.policy_bridge import ActionChunk, ActionQueue


def test_accepts_current_chunk_and_rejects_wrong_episode():
    queue = ActionQueue(max_age_s=1.0)
    queue.reset("episode-a")
    current = ActionChunk.create("episode-a", 4, np.zeros((3, 7), dtype=np.float32))
    assert queue.accept(current, current_step=4, now_s=current.created_monotonic_s)
    assert len(queue) == 3
    wrong = ActionChunk.create("episode-b", 4, np.zeros((1, 7), dtype=np.float32))
    assert not queue.accept(wrong, current_step=4, now_s=wrong.created_monotonic_s)


def test_rejects_stale_chunk():
    queue = ActionQueue(max_age_s=0.25)
    queue.reset("episode-a")
    chunk = ActionChunk.create("episode-a", 2, np.zeros((1, 7), dtype=np.float32))
    assert not queue.accept(chunk, current_step=3, now_s=chunk.created_monotonic_s + 0.3)

