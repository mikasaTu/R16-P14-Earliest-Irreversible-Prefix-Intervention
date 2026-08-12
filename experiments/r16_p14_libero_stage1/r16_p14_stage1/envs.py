from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from .settings import DEFAULT_LIBERO_CONFIG, FEATURE_KEYS, TASK_SPECS


def configure_libero(config_dir: Path | str = DEFAULT_LIBERO_CONFIG) -> None:
    config_dir = Path(config_dir).resolve()
    config_file = config_dir / "config.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(f"LIBERO config is missing: {config_file}")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)


def get_suite(config_dir: Path | str = DEFAULT_LIBERO_CONFIG):
    configure_libero(config_dir)
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()["libero_goal"]()


def make_env(
    task_name: str,
    *,
    config_dir: Path | str = DEFAULT_LIBERO_CONFIG,
    horizon: int | None = None,
    seed: int = 0,
):
    configure_libero(config_dir)
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    spec = TASK_SPECS[task_name]
    suite = get_suite(config_dir)
    task = suite.get_task(spec.task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if not bddl_file.is_file():
        raise FileNotFoundError(f"task BDDL is missing: {bddl_file}")
    env = ControlEnv(
        bddl_file_name=str(bddl_file),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        horizon=horizon or spec.max_episode_steps,
        ignore_done=True,
        hard_reset=True,
    )
    env.seed(seed)
    env.reset()
    return env, suite


def feature_from_obs(obs: dict[str, Any]) -> np.ndarray:
    missing = [key for key in FEATURE_KEYS if key not in obs]
    if missing:
        raise KeyError(f"missing registered state-observation keys: {missing}")
    feature = np.concatenate(
        [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in FEATURE_KEYS]
    )
    if not np.isfinite(feature).all():
        raise ValueError("non-finite state observation")
    return feature


def current_feature(env) -> np.ndarray:
    env._post_process()
    env._update_observables(force=True)
    return feature_from_obs(env.env._get_observations())


def restore_state(env, state: np.ndarray) -> dict[str, Any]:
    """Restore physics and reset controller goals to the restored robot pose."""
    # MuJoCo's flattened state does not include the OSC controller object or
    # robot history buffers.  It also omits warm-start and control workspaces
    # in MjData.  Clear those first, recreate the controller, then overwrite
    # the brief reset pose with the exact target physics state.
    env.sim.reset()
    for robot in env.robots:
        robot.reset(deterministic=True)
    obs = env.set_init_state(np.asarray(state, dtype=np.float64))
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


def state_sha256(state: np.ndarray) -> str:
    state = np.ascontiguousarray(state, dtype=np.float64)
    return hashlib.sha256(state.tobytes()).hexdigest()


def action_chunk_sha256(chunk: np.ndarray) -> str:
    chunk = np.ascontiguousarray(chunk, dtype=np.float32)
    return hashlib.sha256(chunk.tobytes()).hexdigest()


def contact_pairs(env) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        left = env.sim.model.geom_id2name(contact.geom1) or str(contact.geom1)
        right = env.sim.model.geom_id2name(contact.geom2) or str(contact.geom2)
        pairs.append(tuple(sorted((left, right))))
    return tuple(sorted(pairs))


def contact_sha256(env) -> str:
    return hashlib.sha256(repr(contact_pairs(env)).encode("utf-8")).hexdigest()


def shift_free_joint(env, joint_name: str, axis: int, delta: float) -> np.ndarray:
    qpos = np.asarray(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
    if qpos.shape != (7,):
        raise ValueError(f"{joint_name} is not a free joint: qpos shape {qpos.shape}")
    qpos[axis] += float(delta)
    env.sim.data.set_joint_qpos(joint_name, qpos)
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return qpos


def set_free_joint_position(env, joint_name: str, xyz: np.ndarray) -> np.ndarray:
    qpos = np.asarray(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
    if qpos.shape != (7,):
        raise ValueError(f"{joint_name} is not a free joint: qpos shape {qpos.shape}")
    qpos[:3] = np.asarray(xyz, dtype=np.float64)
    env.sim.data.set_joint_qpos(joint_name, qpos)
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return qpos
