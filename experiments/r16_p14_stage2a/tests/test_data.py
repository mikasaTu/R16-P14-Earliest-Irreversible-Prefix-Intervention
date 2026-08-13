from __future__ import annotations

from pathlib import Path

import numpy as np

from r16_p14_stage2a.actor_data import TaskArrays, atomic_save_npz


def test_history_does_not_cross_episode_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "task.npz"
    features = np.arange(6 * 95, dtype=np.float32).reshape(6, 95)
    actions = np.arange(6 * 7, dtype=np.float32).reshape(6, 7)
    atomic_save_npz(
        path,
        features=features,
        actions=actions,
        episode_index=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        timestep=np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int32),
        episode_lengths=np.asarray([3, 3], dtype=np.int32),
        metadata=np.asarray("{}"),
    )
    arrays = TaskArrays("put_the_cream_cheese_in_the_bowl", path)
    states, histories, _, targets, masks = arrays.batch_raw(np.asarray([3]))
    assert np.all(states[0] == features[3])
    assert np.all(histories[0] == 0)
    assert np.all(targets[0, :3] == actions[3:6])
    assert masks[0].sum() == 3
