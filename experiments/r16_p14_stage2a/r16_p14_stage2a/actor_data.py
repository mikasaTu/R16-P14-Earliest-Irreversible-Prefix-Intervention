from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch

from .envs import feature_from_obs, make_env
from .settings import (
    ACTION_DIM,
    ACTION_HISTORY,
    CHUNK_LENGTH,
    DEFAULT_CACHE_ROOT,
    DEFAULT_LIBERO_CONFIG,
    FEATURE_DIM,
    OBS_HISTORY,
    TASK_NAMES,
    TASK_SPECS,
    TASK_TO_INDEX,
)


STAGE1_CACHE_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r16_p14_libero_stage1/pilot-v1"
)


def _demo_sort_key(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def cache_path_for(task_name: str, cache_root: Path = DEFAULT_CACHE_ROOT) -> Path:
    return Path(cache_root) / f"{task_name}.state_obs_v1.npz"


def existing_cache_path(task_name: str, cache_root: Path = DEFAULT_CACHE_ROOT) -> Path | None:
    stage2 = cache_path_for(task_name, cache_root)
    if stage2.is_file():
        return stage2
    stage1 = STAGE1_CACHE_ROOT / f"{task_name}.state_obs_v1.npz"
    return stage1 if stage1.is_file() else None


def build_feature_cache(
    task_name: str,
    *,
    output_path: Path,
    config_dir: Path,
    demo_count: int = 50,
    max_states_per_demo: int | None = None,
) -> Path:
    spec = TASK_SPECS[task_name]
    if output_path.exists():
        raise FileExistsError(output_path)
    env, _ = make_env(task_name, config_dir=config_dir, horizon=spec.horizon)
    feature_blocks: list[np.ndarray] = []
    action_blocks: list[np.ndarray] = []
    episode_blocks: list[np.ndarray] = []
    timestep_blocks: list[np.ndarray] = []
    lengths: list[int] = []
    try:
        with h5py.File(spec.hdf5_path, "r") as dataset:
            names = sorted(dataset["data"].keys(), key=_demo_sort_key)[:demo_count]
            for episode_index, name in enumerate(names):
                demo = dataset["data"][name]
                states = np.asarray(demo["states"], dtype=np.float64)
                actions = np.asarray(demo["actions"], dtype=np.float32)
                length = min(len(states), len(actions))
                if max_states_per_demo is not None:
                    length = min(length, max_states_per_demo)
                features = np.empty((length, FEATURE_DIM), dtype=np.float32)
                for timestep in range(length):
                    features[timestep] = feature_from_obs(
                        env.regenerate_obs_from_state(states[timestep])
                    )
                feature_blocks.append(features)
                action_blocks.append(actions[:length])
                episode_blocks.append(np.full(length, episode_index, dtype=np.int32))
                timestep_blocks.append(np.arange(length, dtype=np.int32))
                lengths.append(length)
                print(
                    f"CACHE_EPISODE task={task_name} episode={episode_index} states={length}",
                    flush=True,
                )
    finally:
        env.close()
    metadata = {
        "schema_version": 1,
        "task": task_name,
        "source_hdf5": str(spec.hdf5_path),
        "demo_count": len(lengths),
        "max_states_per_demo": max_states_per_demo,
        "feature_keys": ["robot0_proprio-state", "object-state"],
        "feature_dim": FEATURE_DIM,
        "action_dim": ACTION_DIM,
    }
    atomic_save_npz(
        output_path,
        features=np.concatenate(feature_blocks),
        actions=np.concatenate(action_blocks),
        episode_index=np.concatenate(episode_blocks),
        timestep=np.concatenate(timestep_blocks),
        episode_lengths=np.asarray(lengths, dtype=np.int32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return output_path


@dataclass(frozen=True)
class Normalization:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
        }

    @classmethod
    def from_dict(cls, value: dict[str, np.ndarray]) -> "Normalization":
        return cls(
            **{
                key: np.asarray(item, dtype=np.float32)
                for key, item in value.items()
            }
        )


class TaskArrays:
    def __init__(self, task_name: str, path: Path, chunk_length: int = CHUNK_LENGTH):
        self.task_name = task_name
        self.task_id = TASK_TO_INDEX[task_name]
        self.path = path
        with np.load(path, allow_pickle=False) as cache:
            self.features = np.asarray(cache["features"], dtype=np.float32)
            self.actions = np.asarray(cache["actions"], dtype=np.float32)
            self.episode_index = np.asarray(cache["episode_index"], dtype=np.int32)
            self.timestep = np.asarray(cache["timestep"], dtype=np.int32)
            self.episode_lengths = np.asarray(cache["episode_lengths"], dtype=np.int32)
        if self.features.shape[1:] != (FEATURE_DIM,) or self.actions.shape[1:] != (ACTION_DIM,):
            raise ValueError(f"bad cache shapes at {path}")
        self.chunk_length = chunk_length
        self.episode_starts = np.cumsum(np.r_[0, self.episode_lengths[:-1]]).astype(np.int64)
        self.row_episode_starts = self.episode_starts[self.episode_index]
        self.row_episode_ends = self.row_episode_starts + self.episode_lengths[self.episode_index]

    def batch_raw(self, indices: np.ndarray) -> tuple[np.ndarray, ...]:
        batch = len(indices)
        states = np.empty((batch, OBS_HISTORY, FEATURE_DIM), dtype=np.float32)
        action_history = np.zeros((batch, ACTION_HISTORY, ACTION_DIM), dtype=np.float32)
        targets = np.empty((batch, self.chunk_length, ACTION_DIM), dtype=np.float32)
        masks = np.zeros((batch, self.chunk_length), dtype=np.float32)
        for output_index, row in enumerate(indices.tolist()):
            start = int(self.row_episode_starts[row])
            end = int(self.row_episode_ends[row])
            for slot, source in enumerate(range(row - OBS_HISTORY + 1, row + 1)):
                states[output_index, slot] = self.features[max(start, source)]
            history_start = max(start, row - ACTION_HISTORY)
            history = self.actions[history_start:row]
            if len(history):
                action_history[output_index, -len(history) :] = history
            valid = min(self.chunk_length, end - row)
            targets[output_index, :valid] = self.actions[row : row + valid]
            targets[output_index, valid:] = self.actions[end - 1]
            masks[output_index, :valid] = 1.0
        task_ids = np.full(batch, self.task_id, dtype=np.int64)
        return states, action_history, task_ids, targets, masks


class MultiTaskSampler:
    def __init__(
        self,
        tasks: Iterable[str],
        cache_root: Path,
        *,
        chunk_length: int = CHUNK_LENGTH,
    ) -> None:
        self.task_names = tuple(tasks)
        self.arrays = {
            task: TaskArrays(
                task,
                existing_cache_path(task, cache_root)
                or cache_path_for(task, cache_root),
                chunk_length,
            )
            for task in self.task_names
        }
        missing = [task for task, item in self.arrays.items() if not item.path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing task caches: {missing}")
        self.normalization = self._normalization()

    def _normalization(self) -> Normalization:
        # Equal task weight, independent of trajectory length.
        state_means = np.stack([item.features.mean(0) for item in self.arrays.values()])
        state_second = np.stack([(item.features**2).mean(0) for item in self.arrays.values()])
        action_means = np.stack([item.actions.mean(0) for item in self.arrays.values()])
        action_second = np.stack([(item.actions**2).mean(0) for item in self.arrays.values()])
        state_mean = state_means.mean(0)
        action_mean = action_means.mean(0)
        state_std = np.sqrt(np.maximum(state_second.mean(0) - state_mean**2, 1e-8))
        action_std = np.sqrt(np.maximum(action_second.mean(0) - action_mean**2, 1e-8))
        return Normalization(
            state_mean=state_mean.astype(np.float32),
            state_std=np.maximum(state_std, 1e-4).astype(np.float32),
            action_mean=action_mean.astype(np.float32),
            action_std=np.maximum(action_std, 1e-4).astype(np.float32),
        )

    def sample(self, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
        # Task-uniform sampling enforces the same effective task weight.
        task_indices = torch.randint(
            len(self.task_names), (batch_size,), generator=generator
        ).numpy()
        states = np.empty((batch_size, OBS_HISTORY, FEATURE_DIM), dtype=np.float32)
        histories = np.empty((batch_size, ACTION_HISTORY, ACTION_DIM), dtype=np.float32)
        task_ids = np.empty(batch_size, dtype=np.int64)
        targets = np.empty((batch_size, CHUNK_LENGTH, ACTION_DIM), dtype=np.float32)
        masks = np.empty((batch_size, CHUNK_LENGTH), dtype=np.float32)
        for task_index, task_name in enumerate(self.task_names):
            positions = np.flatnonzero(task_indices == task_index)
            if not len(positions):
                continue
            arrays = self.arrays[task_name]
            rows = torch.randint(
                len(arrays.features), (len(positions),), generator=generator
            ).numpy()
            raw = arrays.batch_raw(rows)
            states[positions], histories[positions], task_ids[positions], targets[positions], masks[positions] = raw
        norm = self.normalization
        states = (states - norm.state_mean) / norm.state_std
        histories = (histories - norm.action_mean) / norm.action_std
        targets = (targets - norm.action_mean) / norm.action_std
        return tuple(
            torch.from_numpy(item)
            for item in (states, histories, task_ids, targets, masks)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES, default=list(TASK_NAMES))
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_LIBERO_CONFIG)
    parser.add_argument("--demo-count", type=int, default=50)
    parser.add_argument("--max-states-per-demo", type=int)
    args = parser.parse_args()
    for task in args.tasks:
        existing = existing_cache_path(task, args.cache_root)
        if existing is not None and args.max_states_per_demo is None:
            print(f"CACHE_REUSE task={task} path={existing}", flush=True)
            continue
        output = cache_path_for(task, args.cache_root)
        build_feature_cache(
            task,
            output_path=output,
            config_dir=args.config_dir,
            demo_count=args.demo_count,
            max_states_per_demo=args.max_states_per_demo,
        )
        print(f"CACHE_COMPLETE task={task} path={output}", flush=True)


if __name__ == "__main__":
    main()
