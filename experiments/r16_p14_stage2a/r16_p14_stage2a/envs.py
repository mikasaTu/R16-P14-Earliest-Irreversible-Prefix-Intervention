from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from .settings import DEFAULT_LIBERO_CONFIG, FEATURE_KEYS, TASK_SPECS


def configure_libero(config_dir: Path | str = DEFAULT_LIBERO_CONFIG) -> None:
    config_dir = Path(config_dir).resolve()
    if not (config_dir / "config.yaml").is_file():
        raise FileNotFoundError(config_dir / "config.yaml")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)


def get_suite(config_dir: Path | str = DEFAULT_LIBERO_CONFIG):
    configure_libero(config_dir)
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()["libero_goal"]()


def make_env(
    task_name: str,
    *,
    config_dir: Path | str = DEFAULT_LIBERO_CONFIG,
    seed: int = 0,
    horizon: int | None = None,
):
    configure_libero(config_dir)
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    spec = TASK_SPECS[task_name]
    suite = get_suite(config_dir)
    task = suite.get_task(spec.task_id)
    if task.name != spec.name:
        raise RuntimeError(f"task order mismatch: {task.name} != {spec.name}")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = ControlEnv(
        bddl_file_name=str(bddl),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        horizon=horizon or spec.horizon,
        ignore_done=True,
        hard_reset=True,
    )
    env.seed(seed)
    env.reset()
    return env, suite


def feature_from_obs(obs: dict[str, Any]) -> np.ndarray:
    missing = [key for key in FEATURE_KEYS if key not in obs]
    if missing:
        raise KeyError(f"missing registered observation keys: {missing}")
    feature = np.concatenate(
        [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in FEATURE_KEYS]
    )
    if feature.shape != (95,) or not np.isfinite(feature).all():
        raise ValueError(f"invalid state feature: shape={feature.shape}")
    return feature


def current_observation(env) -> dict[str, Any]:
    env._post_process()
    env._update_observables(force=True)
    return env.env._get_observations()


def restore_state(env, state: np.ndarray) -> dict[str, Any]:
    env.sim.reset()
    for robot in env.robots:
        robot.reset(deterministic=True)
    env.set_init_state(np.asarray(state, dtype=np.float64))
    env.env.timestep = 0
    for robot in env.robots:
        controller = robot.controller
        if isinstance(controller, dict):
            for item in controller.values():
                item.reset_goal()
        else:
            controller.reset_goal()
    env._update_observables(force=True)
    return env.env._get_observations()


def joint_qpos(env, joint_name: str) -> np.ndarray:
    return np.asarray(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()


def set_joint_qpos(env, joint_name: str, qpos: np.ndarray) -> None:
    env.sim.data.set_joint_qpos(joint_name, np.asarray(qpos, dtype=np.float64))
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)


def contact_pairs(env) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        left = env.sim.model.geom_id2name(contact.geom1) or str(contact.geom1)
        right = env.sim.model.geom_id2name(contact.geom2) or str(contact.geom2)
        pairs.append(tuple(sorted((left, right))))
    return tuple(sorted(pairs))


def state_sha256(state: np.ndarray) -> str:
    state = np.ascontiguousarray(state, dtype=np.float64)
    return hashlib.sha256(state.tobytes()).hexdigest()


def array_sha256(value: np.ndarray, dtype=np.float32) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def contact_sha256(contacts: Any) -> str:
    return hashlib.sha256(repr(contacts).encode("utf-8")).hexdigest()
