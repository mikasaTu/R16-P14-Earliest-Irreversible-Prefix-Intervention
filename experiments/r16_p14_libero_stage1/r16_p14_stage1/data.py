from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from .envs import feature_from_obs, make_env
from .settings import CHUNK_LENGTH, DEFAULT_CACHE_ROOT, DEFAULT_LIBERO_CONFIG, TASK_SPECS


def _demo_sort_key(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp.open("wb") as file:
        np.savez_compressed(file, **arrays)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)


def build_feature_cache(
    task_name: str,
    *,
    output_path: Path,
    demo_count: int = 50,
    max_states_per_demo: int | None = None,
    config_dir: Path | str = DEFAULT_LIBERO_CONFIG,
) -> Path:
    spec = TASK_SPECS[task_name]
    if not spec.hdf5_path.is_file():
        raise FileNotFoundError(spec.hdf5_path)
    env, _ = make_env(task_name, config_dir=config_dir, horizon=spec.max_episode_steps)
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
                features = np.empty((length, 95), dtype=np.float32)
                for timestep in range(length):
                    obs = env.regenerate_obs_from_state(states[timestep])
                    features[timestep] = feature_from_obs(obs)
                feature_blocks.append(features)
                action_blocks.append(actions[:length])
                episode_blocks.append(np.full(length, episode_index, dtype=np.int32))
                timestep_blocks.append(np.arange(length, dtype=np.int32))
                lengths.append(length)
    finally:
        env.close()

    if not lengths:
        raise RuntimeError(f"no demonstrations found for {task_name}")
    features = np.concatenate(feature_blocks)
    actions = np.concatenate(action_blocks)
    if features.shape[1] != 95 or actions.shape[1] != 7:
        raise ValueError(f"unexpected feature/action shapes: {features.shape}, {actions.shape}")
    metadata = {
        "task": task_name,
        "source_hdf5": str(spec.hdf5_path),
        "demo_count": len(lengths),
        "feature_keys": ["robot0_proprio-state", "object-state"],
        "feature_dim": 95,
        "action_dim": 7,
    }
    atomic_save_npz(
        output_path,
        features=features,
        actions=actions,
        episode_index=np.concatenate(episode_blocks),
        timestep=np.concatenate(timestep_blocks),
        episode_lengths=np.asarray(lengths, dtype=np.int32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return output_path


@dataclass(frozen=True)
class Normalization:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def as_torch(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "feature_mean": torch.as_tensor(self.feature_mean, device=device),
            "feature_std": torch.as_tensor(self.feature_std, device=device),
            "action_mean": torch.as_tensor(self.action_mean, device=device),
            "action_std": torch.as_tensor(self.action_std, device=device),
        }

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
        }

    @classmethod
    def from_dict(cls, value: dict[str, np.ndarray]) -> "Normalization":
        return cls(**{key: np.asarray(item, dtype=np.float32) for key, item in value.items()})


class ChunkArrays:
    def __init__(self, cache_path: Path, chunk_length: int = CHUNK_LENGTH):
        with np.load(cache_path, allow_pickle=False) as cache:
            self.features = np.asarray(cache["features"], dtype=np.float32)
            self.actions = np.asarray(cache["actions"], dtype=np.float32)
            self.episode_index = np.asarray(cache["episode_index"], dtype=np.int32)
            self.timestep = np.asarray(cache["timestep"], dtype=np.int32)
            self.episode_lengths = np.asarray(cache["episode_lengths"], dtype=np.int32)
        self.chunk_length = int(chunk_length)
        self.normalization = self._compute_normalization()
        self.targets, self.masks = self._build_targets()

    def _compute_normalization(self) -> Normalization:
        feature_std = self.features.std(axis=0)
        action_std = self.actions.std(axis=0)
        feature_std = np.maximum(feature_std, 1e-4).astype(np.float32)
        action_std = np.maximum(action_std, 1e-4).astype(np.float32)
        return Normalization(
            feature_mean=self.features.mean(axis=0).astype(np.float32),
            feature_std=feature_std,
            action_mean=self.actions.mean(axis=0).astype(np.float32),
            action_std=action_std,
        )

    def _build_targets(self) -> tuple[np.ndarray, np.ndarray]:
        count, action_dim = self.actions.shape
        targets = np.empty((count, self.chunk_length, action_dim), dtype=np.float32)
        masks = np.zeros((count, self.chunk_length), dtype=np.float32)
        episode_starts = np.cumsum(np.r_[0, self.episode_lengths[:-1]])
        for start, length in zip(episode_starts, self.episode_lengths, strict=True):
            end = int(start + length)
            for index in range(int(start), end):
                valid = min(self.chunk_length, end - index)
                targets[index, :valid] = self.actions[index : index + valid]
                targets[index, valid:] = self.actions[end - 1]
                masks[index, :valid] = 1.0
        return targets, masks

    def normalized_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm = self.normalization
        features = (self.features - norm.feature_mean) / norm.feature_std
        targets = (self.targets - norm.action_mean) / norm.action_std
        return (
            torch.from_numpy(features),
            torch.from_numpy(targets),
            torch.from_numpy(self.masks),
        )


def cache_path_for(task_name: str, cache_root: Path = DEFAULT_CACHE_ROOT) -> Path:
    return Path(cache_root) / f"{task_name}.state_obs_v1.npz"
