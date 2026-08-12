from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from r16_p14_stage1.checkpoint import find_latest_complete, save_complete_checkpoint
from r16_p14_stage1.data import ChunkArrays, atomic_save_npz
from r16_p14_stage1.model import ChunkedBCMLP


def _cache(path: Path) -> None:
    features = np.arange(5 * 95, dtype=np.float32).reshape(5, 95)
    actions = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    atomic_save_npz(
        path,
        features=features,
        actions=actions,
        episode_index=np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
        timestep=np.asarray([0, 1, 2, 0, 1], dtype=np.int32),
        episode_lengths=np.asarray([3, 2], dtype=np.int32),
        metadata=np.asarray(json.dumps({"test": True})),
    )


def test_chunks_do_not_cross_episode(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    _cache(path)
    arrays = ChunkArrays(path, chunk_length=4)
    np.testing.assert_array_equal(arrays.targets[2], np.repeat(arrays.actions[2:3], 4, axis=0))
    np.testing.assert_array_equal(arrays.masks[2], [1, 0, 0, 0])
    np.testing.assert_array_equal(arrays.targets[3, :2], arrays.actions[3:5])
    np.testing.assert_array_equal(arrays.masks[3], [1, 1, 0, 0])


def test_incomplete_checkpoint_is_ignored(tmp_path: Path) -> None:
    lineage = tmp_path / "lineage"
    incomplete = lineage / "step_99999999"
    incomplete.mkdir(parents=True)
    (incomplete / "checkpoint.pt").write_bytes(b"partial")
    complete = save_complete_checkpoint(lineage, step=12, payload={"step": 12})
    assert find_latest_complete(lineage) == complete


def test_model_shape() -> None:
    model = ChunkedBCMLP(feature_dim=95, action_dim=7, chunk_length=16, hidden_dim=32)
    assert model(torch.zeros(3, 95)).shape == (3, 16, 7)
