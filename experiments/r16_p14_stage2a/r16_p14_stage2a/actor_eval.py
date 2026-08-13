from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .actor_data import Normalization
from .actor_model import ACTConfig, HistoryConditionedStateACT, predict_chunk
from .envs import contact_pairs, feature_from_obs, joint_qpos, make_env, restore_state
from .io_utils import append_jsonl, atomic_write_json
from .mechanics import joint_trajectory, load_demo
from .settings import (
    ACTION_DIM,
    ACTION_HISTORY,
    ARTIFACT_ROOT,
    DEFAULT_LIBERO_CONFIG,
    OBS_HISTORY,
    TASK_NAMES,
    TASK_SPECS,
    TASK_TO_INDEX,
)


def object_key(joint: str | None) -> str:
    return "" if joint is None else joint.removesuffix("_joint0")


def has_pair(env, left: str, right: str) -> bool:
    return any(
        any(left in item for item in pair) and any(right in item for item in pair)
        for pair in contact_pairs(env)
    )


@dataclass
class PhaseTracker:
    task_name: str
    initial_manipulated: np.ndarray | None
    initial_mechanism: float | None
    goal_manipulated: np.ndarray | None
    grasp_or_open: bool = False
    lift_or_transport: bool = False
    pre_release_or_contact: bool = False

    @classmethod
    def create(cls, env, task_name: str) -> "PhaseTracker":
        spec = TASK_SPECS[task_name]
        initial_manipulated = (
            joint_qpos(env, spec.manipulated_joint)
            if spec.manipulated_joint
            else None
        )
        initial_mechanism = (
            float(joint_qpos(env, spec.mechanism_joint).reshape(-1)[0])
            if spec.mechanism_joint
            else None
        )
        goal = None
        if task_name in {
            "open_the_top_drawer_and_put_the_bowl_inside",
            "push_the_plate_to_the_front_of_the_stove",
            "put_the_bowl_on_the_stove",
        }:
            states, _ = load_demo(task_name, 0)
            assert spec.manipulated_joint
            goal = np.asarray(
                joint_trajectory(env, states, spec.manipulated_joint)[-1],
                dtype=np.float64,
            )
        return cls(task_name, initial_manipulated, initial_mechanism, goal)

    def observe(self, env, action: np.ndarray, success: bool) -> None:
        spec = TASK_SPECS[self.task_name]
        if self.task_name == "open_the_middle_drawer_of_the_cabinet":
            assert spec.mechanism_joint and self.initial_mechanism is not None
            motion = abs(
                float(joint_qpos(env, spec.mechanism_joint).reshape(-1)[0])
                - self.initial_mechanism
            )
            self.grasp_or_open |= motion >= 0.005
            self.lift_or_transport |= motion >= 0.030
            self.pre_release_or_contact |= motion >= 0.080 or success
            return
        if self.task_name == "open_the_top_drawer_and_put_the_bowl_inside":
            assert spec.mechanism_joint and self.initial_mechanism is not None
            assert spec.manipulated_joint and self.initial_manipulated is not None
            drawer_motion = abs(
                float(joint_qpos(env, spec.mechanism_joint).reshape(-1)[0])
                - self.initial_mechanism
            )
            object_position = joint_qpos(env, spec.manipulated_joint)
            self.grasp_or_open |= drawer_motion >= 0.030
            self.lift_or_transport |= (
                object_position[2] - self.initial_manipulated[2] >= spec.lift_delta
            )
            if self.goal_manipulated is not None:
                self.pre_release_or_contact |= (
                    np.linalg.norm(object_position[:3] - self.goal_manipulated[:3]) <= 0.10
                )
            self.pre_release_or_contact |= success
            return
        assert spec.manipulated_joint and self.initial_manipulated is not None
        object_position = joint_qpos(env, spec.manipulated_joint)
        displacement = float(
            np.linalg.norm(object_position[:3] - self.initial_manipulated[:3])
        )
        if spec.family == "swept_path_blocker":
            self.grasp_or_open |= displacement >= 0.005
            self.lift_or_transport |= displacement >= 0.040
            if self.goal_manipulated is not None:
                self.pre_release_or_contact |= (
                    np.linalg.norm(object_position[:3] - self.goal_manipulated[:3]) <= 0.06
                )
        else:
            lift = float(object_position[2] - self.initial_manipulated[2])
            self.grasp_or_open |= lift >= 0.008
            self.lift_or_transport |= lift >= spec.lift_delta
            if spec.target_joint:
                target = joint_qpos(env, spec.target_joint)
                self.pre_release_or_contact |= (
                    np.linalg.norm(object_position[:2] - target[:2])
                    <= spec.placement_xy_tolerance
                    or has_pair(env, object_key(spec.manipulated_joint), object_key(spec.target_joint))
                )
            elif self.goal_manipulated is not None:
                self.pre_release_or_contact |= (
                    np.linalg.norm(object_position[:3] - self.goal_manipulated[:3]) <= 0.10
                )
        self.pre_release_or_contact |= success


def load_actor(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = HistoryConditionedStateACT(ACTConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    normalization = Normalization.from_dict(payload["normalization"])
    return payload, model, normalization


def rollout(
    env,
    *,
    task_name: str,
    init_state: np.ndarray,
    model: HistoryConditionedStateACT,
    normalization: Normalization,
    device: torch.device,
    max_steps: int,
) -> dict[str, Any]:
    env.reset()
    obs = restore_state(env, init_state)
    initial_feature = feature_from_obs(obs)
    state_history = [initial_feature.copy() for _ in range(OBS_HISTORY)]
    action_history = [np.zeros(ACTION_DIM, dtype=np.float32) for _ in range(ACTION_HISTORY)]
    phase = PhaseTracker.create(env, task_name)
    actions: list[np.ndarray] = []
    chunk_hashes: list[str] = []
    success = bool(env.check_success())
    step = 0
    while step < max_steps and not success:
        chunk = predict_chunk(
            model,
            normalization,
            np.asarray(state_history),
            np.asarray(action_history),
            TASK_TO_INDEX[task_name],
            device,
        )
        chunk_hashes.append(
            __import__("hashlib").sha256(
                np.ascontiguousarray(chunk, dtype=np.float32).tobytes()
            ).hexdigest()
        )
        action = chunk[0]
        obs, _, _, _ = env.step(action)
        step += 1
        success = bool(env.check_success())
        actions.append(action.copy())
        state_history = state_history[1:] + [feature_from_obs(obs)]
        action_history = action_history[1:] + [action.copy()]
        phase.observe(env, action, success)
    action_array = np.asarray(actions, dtype=np.float32)
    smoothness = (
        float(np.linalg.norm(np.diff(action_array[:, :6], axis=0), axis=1).mean())
        if len(action_array) > 1
        else 0.0
    )
    path_length = (
        float(np.linalg.norm(action_array[:, :3], axis=1).sum())
        if len(action_array)
        else 0.0
    )
    return {
        "success": bool(success),
        "safe_success": bool(success),
        "episode_length": step,
        "policy_calls": len(chunk_hashes),
        "chunk_smoothness": smoothness,
        "action_space_path_length": path_length,
        "grasp_or_open_phase_reached": bool(phase.grasp_or_open),
        "lift_or_transport_phase_reached": bool(phase.lift_or_transport),
        "pre_release_or_contact_phase_reached": bool(phase.pre_release_or_contact),
        "failure_after_target_phase": bool(
            not success and phase.pre_release_or_contact
        ),
        "first_chunk_hash": chunk_hashes[0] if chunk_hashes else None,
        "last_chunk_hash": chunk_hashes[-1] if chunk_hashes else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_ROOT / "actor/eval")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_LIBERO_CONFIG)
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES, default=list(TASK_NAMES))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    payload, model, normalization = load_actor(args.checkpoint, device)
    if int(payload["seed"]) != args.seed:
        raise ValueError("checkpoint seed mismatch")
    output_path = args.output_dir / f"seed_{args.seed}.episodes.jsonl"
    if output_path.exists():
        if output_path.stat().st_size == 0:
            # A serialization error can create the append target before the
            # first line is committed. Only a provably empty partial file is
            # safe to remove on retry; non-empty evidence stays fail-closed.
            output_path.unlink()
        else:
            raise FileExistsError(output_path)
    records: list[dict[str, Any]] = []
    for task_name in args.tasks:
        spec = TASK_SPECS[task_name]
        env, suite = make_env(task_name, config_dir=args.config_dir, seed=0)
        init_states = np.asarray(suite.get_task_init_states(spec.task_id))
        try:
            for episode in range(args.episodes):
                result = rollout(
                    env,
                    task_name=task_name,
                    init_state=init_states[episode % len(init_states)],
                    model=model,
                    normalization=normalization,
                    device=device,
                    max_steps=spec.horizon,
                )
                record = {
                    "run_id": payload["run_id"],
                    "actor": "HistoryConditionedStateACT",
                    "policy_seed": args.seed,
                    "task": task_name,
                    "environment_seed": episode % len(init_states),
                    "execution_horizon": 1,
                    "actual_chunk_length": 16,
                    "checkpoint": str(args.checkpoint),
                    **result,
                }
                append_jsonl(output_path, record)
                records.append(record)
                print(
                    f"ACT_EVAL_EPISODE seed={args.seed} task={task_name} "
                    f"episode={episode} success={int(result['success'])} "
                    f"late={int(result['pre_release_or_contact_phase_reached'])} "
                    f"steps={result['episode_length']}",
                    flush=True,
                )
        finally:
            env.close()
    groups: dict[str, Any] = {}
    for task_name in args.tasks:
        selected = [record for record in records if record["task"] == task_name]
        failures = [record for record in selected if not record["success"]]
        groups[task_name] = {
            "episodes": len(selected),
            "success_rate": sum(record["success"] for record in selected) / len(selected),
            "grasp_or_open_phase_reach_rate": sum(record["grasp_or_open_phase_reached"] for record in selected) / len(selected),
            "lift_or_transport_phase_reach_rate": sum(record["lift_or_transport_phase_reached"] for record in selected) / len(selected),
            "pre_release_or_contact_phase_reach_rate": sum(record["pre_release_or_contact_phase_reached"] for record in selected) / len(selected),
            "failed_episode_count": len(failures),
            "failure_after_target_phase_fraction": (
                sum(record["failure_after_target_phase"] for record in failures) / len(failures)
                if failures
                else None
            ),
            "mean_chunk_smoothness": float(np.mean([record["chunk_smoothness"] for record in selected])),
            "mean_policy_calls": float(np.mean([record["policy_calls"] for record in selected])),
            "mean_episode_length": float(np.mean([record["episode_length"] for record in selected])),
        }
    summary = {
        "schema_version": 1,
        "run_id": payload["run_id"],
        "actor": "HistoryConditionedStateACT",
        "policy_seed": args.seed,
        "parameter_count": payload["parameter_count"],
        "episode_count": len(records),
        "groups": groups,
    }
    atomic_write_json(args.output_dir / f"seed_{args.seed}.summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
