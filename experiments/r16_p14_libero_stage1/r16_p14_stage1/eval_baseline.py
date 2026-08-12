from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .envs import (
    action_chunk_sha256,
    contact_pairs,
    feature_from_obs,
    make_env,
    restore_state,
    state_sha256,
)
from .io_utils import append_jsonl, atomic_write_json
from .model import ChunkedBCMLP, predict_chunk
from .policy_io import load_policy
from .settings import (
    DEFAULT_CKPT_ROOT,
    DEFAULT_LIBERO_CONFIG,
    DEFAULT_LOG_ROOT,
    DEVELOPMENT_TASKS,
    EXECUTION_HORIZONS,
    TASK_SPECS,
    TRAIN_SEEDS,
)


@dataclass
class RolloutTrace:
    chunks: list[np.ndarray]
    actions: list[np.ndarray]
    state_hashes: list[str]
    contact_hashes: list[str]
    success: bool
    episode_length: int
    policy_calls: int
    contact_steps: int
    gripper_events: int

    def parity_signature(self) -> dict[str, Any]:
        return {
            "chunk_hashes": [action_chunk_sha256(chunk) for chunk in self.chunks],
            "action_sha256": hashlib.sha256(
                np.ascontiguousarray(np.asarray(self.actions), dtype=np.float32).tobytes()
            ).hexdigest(),
            "state_hashes": self.state_hashes,
            "contact_hashes": self.contact_hashes,
            "success": self.success,
            "episode_length": self.episode_length,
            "policy_calls": self.policy_calls,
        }


class DetailLogger:
    def __init__(self, path: Path | None):
        self.path = path

    def query(self, *, step: int, state: np.ndarray, chunk: np.ndarray, feature: np.ndarray) -> None:
        if self.path is None:
            return
        append_jsonl(
            self.path,
            {
                "event": "policy_query",
                "timestamp_ns": time.time_ns(),
                "step": step,
                "state": np.asarray(state, dtype=np.float64).tolist(),
                "state_hash": state_sha256(state),
                "feature": np.asarray(feature, dtype=np.float32).tolist(),
                "action_chunk": np.asarray(chunk, dtype=np.float32).tolist(),
                "action_chunk_hash": action_chunk_sha256(chunk),
            },
        )

    def action(
        self,
        *,
        step: int,
        prefix_index: int,
        action: np.ndarray,
        state: np.ndarray,
        contacts: tuple[tuple[str, str], ...],
        success: bool,
    ) -> None:
        if self.path is None:
            return
        append_jsonl(
            self.path,
            {
                "event": "executed_action",
                "timestamp_ns": time.time_ns(),
                "step": step,
                "executed_prefix_index": prefix_index,
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "state_hash": state_sha256(state),
                "contacts": contacts,
                "success": bool(success),
            },
        )


def rollout_episode(
    env,
    *,
    init_state: np.ndarray,
    model: ChunkedBCMLP,
    normalization,
    device: torch.device,
    execution_horizon: int,
    max_steps: int,
    detail_log: Path | None = None,
) -> RolloutTrace:
    obs = restore_state(env, np.asarray(init_state, dtype=np.float64))
    chunks: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    state_hashes: list[str] = []
    contact_hashes: list[str] = []
    contact_steps = 0
    gripper_events = 0
    previous_gripper: float | None = None
    logger = DetailLogger(detail_log)
    step = 0
    success = bool(env.check_success())
    while step < max_steps and not success:
        feature = feature_from_obs(obs)
        chunk = predict_chunk(model, normalization, feature, device)
        chunks.append(chunk.copy())
        query_state = env.get_sim_state().copy()
        logger.query(step=step, state=query_state, chunk=chunk, feature=feature)
        for prefix_index, action in enumerate(chunk[:execution_horizon]):
            if step >= max_steps or success:
                break
            obs, _, _, _ = env.step(action)
            step += 1
            success = bool(env.check_success())
            state = env.get_sim_state().copy()
            contacts = contact_pairs(env)
            actions.append(action.copy())
            state_hashes.append(state_sha256(state))
            contact_hashes.append(hashlib.sha256(repr(contacts).encode("utf-8")).hexdigest())
            if contacts:
                contact_steps += 1
            gripper = float(action[-1])
            if previous_gripper is not None and np.sign(gripper) != np.sign(previous_gripper):
                gripper_events += 1
            previous_gripper = gripper
            logger.action(
                step=step,
                prefix_index=prefix_index,
                action=action,
                state=state,
                contacts=contacts,
                success=success,
            )
    return RolloutTrace(
        chunks=chunks,
        actions=actions,
        state_hashes=state_hashes,
        contact_hashes=contact_hashes,
        success=success,
        episode_length=step,
        policy_calls=len(chunks),
        contact_steps=contact_steps,
        gripper_events=gripper_events,
    )


def run_parity(
    *,
    task_name: str,
    config_dir: Path,
    init_state: np.ndarray,
    model: ChunkedBCMLP,
    normalization,
    device: torch.device,
    execution_horizon: int,
    max_steps: int,
    detail_path: Path,
) -> dict[str, Any]:
    if detail_path.exists():
        detail_path.unlink()
    disabled_env, _ = make_env(task_name, config_dir=config_dir)
    enabled_env, _ = make_env(task_name, config_dir=config_dir)
    try:
        disabled = rollout_episode(
            disabled_env,
            init_state=init_state,
            model=model,
            normalization=normalization,
            device=device,
            execution_horizon=execution_horizon,
            max_steps=max_steps,
        )
        enabled = rollout_episode(
            enabled_env,
            init_state=init_state,
            model=model,
            normalization=normalization,
            device=device,
            execution_horizon=execution_horizon,
            max_steps=max_steps,
            detail_log=detail_path,
        )
    finally:
        disabled_env.close()
        enabled_env.close()
    left = disabled.parity_signature()
    right = enabled.parity_signature()
    return {"passed": left == right, "disabled": left, "enabled": right}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEVELOPMENT_TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TRAIN_SEEDS))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(EXECUTION_HORIZONS))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--libero-config", type=Path, default=DEFAULT_LIBERO_CONFIG)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--parity-steps", type=int, default=64)
    parser.add_argument("--max-episode-steps", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "2"))))
    output_dir = args.log_root / args.run_id / "baseline_eval"
    output_path = output_dir / "episodes.jsonl"
    if output_path.exists():
        raise FileExistsError(f"refusing to append to existing evaluation: {output_path}")
    records: list[dict[str, Any]] = []
    parity_result: dict[str, Any] | None = None
    for task_name in args.tasks:
        spec = TASK_SPECS[task_name]
        rollout_budget = args.max_episode_steps or spec.max_episode_steps
        env, suite = make_env(task_name, config_dir=args.libero_config, horizon=spec.max_episode_steps)
        init_states = np.asarray(suite.get_task_init_states(spec.task_id))
        try:
            for seed in args.seeds:
                model, normalization, checkpoint_dir = load_policy(
                    args.checkpoint_root, args.run_id, task_name, seed, device
                )
                if args.parity and parity_result is None:
                    parity_result = run_parity(
                        task_name=task_name,
                        config_dir=args.libero_config,
                        init_state=init_states[0],
                        model=model,
                        normalization=normalization,
                        device=device,
                        execution_horizon=8,
                        max_steps=min(rollout_budget, args.parity_steps),
                        detail_path=output_dir / "instrumentation_parity.enabled.jsonl",
                    )
                    atomic_write_json(output_dir / "instrumentation_parity.json", parity_result)
                    if not parity_result["passed"]:
                        raise RuntimeError("instrumentation parity failed")
                    print("INSTRUMENTATION_PARITY_OK", flush=True)
                for execution_horizon in args.horizons:
                    for episode in range(args.episodes):
                        init_id = episode % len(init_states)
                        trace = rollout_episode(
                            env,
                            init_state=init_states[init_id],
                            model=model,
                            normalization=normalization,
                            device=device,
                            execution_horizon=execution_horizon,
                            max_steps=rollout_budget,
                        )
                        action_array = np.asarray(trace.actions, dtype=np.float32)
                        record = {
                            "run_id": args.run_id,
                            "task": task_name,
                            "policy_seed": seed,
                            "environment_seed": init_id,
                            "execution_horizon": execution_horizon,
                            "actual_chunk_length": model.chunk_length,
                            "success": trace.success,
                            "safe_success": trace.success,
                            "episode_length": trace.episode_length,
                            "executed_chunk_count": trace.policy_calls,
                            "policy_calls": trace.policy_calls,
                            "mean_action_magnitude": float(
                                np.linalg.norm(action_array[:, :6], axis=1).mean()
                            ) if len(action_array) else 0.0,
                            "contact_steps": trace.contact_steps,
                            "gripper_events": trace.gripper_events,
                            "failure_type": None if trace.success else "timeout_or_policy_failure",
                            "checkpoint": str(checkpoint_dir),
                        }
                        append_jsonl(output_path, record)
                        records.append(record)
                        if len(records) == 1:
                            atomic_write_json(
                                output_dir / "first_completed_rollout.json", record
                            )
                        print(
                            f"EVAL_EPISODE task={task_name} seed={seed} horizon={execution_horizon} "
                            f"episode={episode} success={int(trace.success)} steps={trace.episode_length}",
                            flush=True,
                        )
        finally:
            env.close()
    groups: dict[str, dict[str, float]] = {}
    for record in records:
        key = f"{record['task']}|seed={record['policy_seed']}|h={record['execution_horizon']}"
        group = groups.setdefault(key, {"episodes": 0.0, "successes": 0.0, "steps": 0.0})
        group["episodes"] += 1
        group["successes"] += float(record["success"])
        group["steps"] += float(record["episode_length"])
    for group in groups.values():
        group["success_rate"] = group["successes"] / group["episodes"]
        group["mean_episode_length"] = group["steps"] / group["episodes"]
    summary = {
        "run_id": args.run_id,
        "episode_count": len(records),
        "instrumentation_parity": parity_result,
        "groups": groups,
    }
    summary_path = output_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    print(f"BASELINE_EVAL_COMPLETE summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
