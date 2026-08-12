from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
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
from .model import predict_chunk
from .perturbations import (
    apply_controlled_perturbation,
    object_position,
    repair_primitive,
    safety_violation,
)
from .policy_io import load_policy
from .settings import (
    CHUNK_LENGTH,
    DEFAULT_CKPT_ROOT,
    DEFAULT_LIBERO_CONFIG,
    DEFAULT_LOG_ROOT,
    TASK_SPECS,
)


OPERATORS = (
    "nominal_continue",
    "trim_and_replan",
    "hold_and_replan",
    "bounded_rollback_and_replan",
    "cause_specific_local_repair",
)
SEVERITIES = ("low", "medium", "high")


@dataclass(frozen=True)
class Candidate:
    demo_id: int
    anchor_step: int
    anchor_rank: int
    insertion_prefix: int
    severity: str

    @property
    def candidate_id(self) -> str:
        return (
            f"demo{self.demo_id:02d}_t{self.anchor_step:04d}_a{self.anchor_rank}_"
            f"p{self.insertion_prefix:02d}_{self.severity}"
        )


def _load_demo(task_name: str, demo_id: int) -> tuple[np.ndarray, np.ndarray]:
    spec = TASK_SPECS[task_name]
    with h5py.File(spec.hdf5_path, "r") as dataset:
        demo = dataset[f"data/demo_{demo_id}"]
        return (
            np.asarray(demo["states"], dtype=np.float64),
            np.asarray(demo["actions"], dtype=np.float32),
        )


def _joint_trajectory(env, states: np.ndarray, joint_name: str) -> np.ndarray:
    address = env.sim.model.get_joint_qpos_addr(joint_name)
    qpos = states[:, 1 : 1 + env.sim.model.nq]
    if isinstance(address, (int, np.integer)):
        return qpos[:, int(address)]
    return qpos[:, int(address[0]) : int(address[1])]


def phase_anchors(
    env,
    task_name: str,
    states: np.ndarray,
    actions: np.ndarray,
) -> list[int]:
    if task_name == "open_the_middle_drawer_of_the_cabinet":
        drawer = np.asarray(
            _joint_trajectory(env, states, "wooden_cabinet_1_middle_level")
        ).reshape(-1)
        indices = np.flatnonzero(np.abs(drawer - drawer[0]) > 0.005)
        if len(indices) == 0:
            return []
        event = int(indices[0])
    elif task_name == "put_the_bowl_on_the_plate":
        bowl = _joint_trajectory(env, states, "akita_black_bowl_1_joint0")
        lifted = np.flatnonzero(bowl[:, 2] - bowl[0, 2] > 0.03)
        if len(lifted) == 0:
            return []
        releases = np.flatnonzero(
            (np.arange(len(actions)) >= int(lifted[0]))
            & (actions[:, -1] < 0)
            & np.r_[False, actions[:-1, -1] > 0]
        )
        event = int(releases[-1]) if len(releases) else max(0, len(actions) - 48)
    elif task_name == "put_the_wine_bottle_on_the_rack":
        bottle = _joint_trajectory(env, states, "wine_bottle_1_joint0")
        lifted = np.flatnonzero(bottle[:, 2] - bottle[0, 2] > 0.05)
        if len(lifted) == 0:
            return []
        releases = np.flatnonzero(
            (np.arange(len(actions)) >= int(lifted[0]))
            & (actions[:, -1] < 0)
            & np.r_[False, actions[:-1, -1] > 0]
        )
        release = int(releases[-1]) if len(releases) else 0
        # Some official wine-rack demonstrations include a long post-release
        # settling/pushing phase.  Keep every branch within the preregistered
        # 96-step budget while remaining in the final contact-alignment phase.
        event = max(release, len(actions) - 64)
    else:
        return []
    proposed = [event - 8, event, event + 6]
    return sorted({max(0, min(len(states) - CHUNK_LENGTH - 1, item)) for item in proposed})


def candidate_schedule(
    env,
    task_name: str,
    *,
    demo_count: int,
    insertion_prefixes: Iterable[int],
    max_candidates: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    prefixes = tuple(int(item) for item in insertion_prefixes)
    for demo_id in range(demo_count):
        states, actions = _load_demo(task_name, demo_id)
        for anchor_rank, anchor_step in enumerate(
            phase_anchors(env, task_name, states, actions)
        ):
            for prefix_rank, insertion_prefix in enumerate(prefixes):
                severity = SEVERITIES[(demo_id + anchor_rank + prefix_rank) % len(SEVERITIES)]
                candidates.append(
                    Candidate(
                        demo_id=demo_id,
                        anchor_step=anchor_step,
                        anchor_rank=anchor_rank,
                        insertion_prefix=insertion_prefix,
                        severity=severity,
                    )
                )
                if len(candidates) >= max_candidates:
                    return candidates
    return candidates


def _initial_phase(env, task_name: str) -> dict[str, bool]:
    lifted = False
    if task_name == "put_the_bowl_on_the_plate":
        lifted = object_position(env, "akita_black_bowl_1_joint0")[2] > 0.94
    elif task_name == "put_the_wine_bottle_on_the_rack":
        lifted = object_position(env, "wine_bottle_1_joint0")[2] > 0.96
    return {"object_lifted": bool(lifted)}


def _inverse_rollback(recent_actions: list[np.ndarray], count: int = 3) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for source in reversed(recent_actions[-count:]):
        action = np.asarray(source, dtype=np.float32).copy()
        action[:6] *= -1.0
        output.append(np.clip(action, -1.0, 1.0))
    return output


def _hold_action(task_name: str) -> np.ndarray:
    gripper = 1.0 if task_name != "open_the_middle_drawer_of_the_cabinet" else -1.0
    return np.asarray([0, 0, 0, 0, 0, 0, gripper], dtype=np.float32)


def run_branch(
    env,
    *,
    task_name: str,
    snapshot: np.ndarray,
    original_chunk: np.ndarray,
    prefix_k: int,
    operator: str,
    model,
    normalization,
    device: torch.device,
    branch_budget: int,
    execution_horizon: int,
    expert_suffix: np.ndarray | None = None,
) -> dict[str, Any]:
    obs = restore_state(env, snapshot)
    recent_actions = [np.asarray(item, dtype=np.float32).copy() for item in original_chunk[:prefix_k]]
    if operator == "nominal_continue":
        queue = [item.copy() for item in original_chunk[prefix_k:]]
    elif operator == "trim_and_replan":
        queue = []
    elif operator == "hold_and_replan":
        queue = [_hold_action(task_name)]
    elif operator == "bounded_rollback_and_replan":
        queue = _inverse_rollback(recent_actions)
    elif operator == "cause_specific_local_repair":
        queue = repair_primitive(task_name, recent_actions)
    elif operator == "privileged_scripted_recovery":
        queue = repair_primitive(task_name, recent_actions)
        queue.extend([] if expert_suffix is None else [item.copy() for item in expert_suffix])
    else:
        raise KeyError(operator)

    phase = _initial_phase(env, task_name)
    success = bool(env.check_success())
    violation = False
    violation_type: str | None = None
    executed: list[np.ndarray] = []
    policy_calls = 0
    path_length = 0.0
    contact_steps = 0
    while len(executed) < branch_budget and not success and not violation:
        if not queue:
            if operator == "privileged_scripted_recovery":
                break
            feature = feature_from_obs(obs)
            predicted = predict_chunk(model, normalization, feature, device)
            queue.extend(item.copy() for item in predicted[:execution_horizon])
            policy_calls += 1
        action = np.asarray(queue.pop(0), dtype=np.float32)
        obs, _, _, _ = env.step(action)
        executed.append(action.copy())
        recent_actions.append(action.copy())
        path_length += float(np.linalg.norm(action[:3]))
        if contact_pairs(env):
            contact_steps += 1
        violation, violation_type = safety_violation(
            env, task_name, action=action, phase=phase
        )
        success = bool(env.check_success())
    return {
        "operator": operator,
        "safe": not violation,
        "task_recoverable": success,
        "safe_success": bool(success and not violation),
        "violation": violation,
        "violation_type": violation_type,
        "steps": len(executed),
        "rework_steps": len(executed),
        "extra_path_length": path_length,
        "policy_calls": policy_calls,
        "rollback_count": int(operator in {"bounded_rollback_and_replan", "cause_specific_local_repair"}),
        "contact_steps": contact_steps,
        "final_state_hash": state_sha256(env.get_sim_state()),
    }


def replay_suffix_determinism(
    env,
    *,
    snapshot: np.ndarray,
    suffix: np.ndarray,
) -> dict[str, Any]:
    outputs: list[tuple[np.ndarray, bool, str]] = []
    for _ in range(3):
        restore_state(env, snapshot)
        contacts = []
        for action in suffix:
            env.step(action)
            contacts.append(contact_pairs(env))
        outputs.append(
            (
                env.get_sim_state().copy(),
                bool(env.check_success()),
                hashlib.sha256(repr(contacts).encode("utf-8")).hexdigest(),
            )
        )
    max_error = max(float(np.max(np.abs(outputs[0][0] - item[0]))) for item in outputs[1:])
    event_match = all(item[1:] == outputs[0][1:] for item in outputs[1:])
    return {
        "repeats": 3,
        "max_state_abs_error": max_error,
        "outcome_and_contact_match": event_match,
        "passed": bool(max_error < 1e-9 and event_match),
    }


def audit_candidate(
    env,
    *,
    task_name: str,
    candidate: Candidate,
    model,
    normalization,
    device: torch.device,
    branch_budget: int,
    execution_horizon: int,
    prefix_stride: int,
) -> dict[str, Any]:
    states, expert_actions = _load_demo(task_name, candidate.demo_id)
    anchor_state = states[candidate.anchor_step]
    obs = restore_state(env, anchor_state)
    feature = feature_from_obs(obs)
    chunk = predict_chunk(model, normalization, feature, device)
    snapshots = [env.get_sim_state().copy()]
    applied = None
    for index, action in enumerate(chunk):
        env.step(action)
        if index + 1 == candidate.insertion_prefix:
            applied = apply_controlled_perturbation(env, task_name, candidate.severity)
        snapshots.append(env.get_sim_state().copy())
    if applied is None:
        raise ValueError("perturbation insertion prefix was outside the action chunk")

    replay = replay_suffix_determinism(
        env,
        snapshot=snapshots[candidate.insertion_prefix],
        suffix=chunk[candidate.insertion_prefix :],
    )
    prefix_values = list(
        range(candidate.insertion_prefix, len(chunk) + 1, max(1, prefix_stride))
    )
    if prefix_values[-1] != len(chunk):
        prefix_values.append(len(chunk))
    prefix_records = []
    recoverable_by_k: dict[int, bool] = {}
    physical_by_k: dict[int, bool] = {}
    for prefix_k in prefix_values:
        branches = {}
        expert_start = min(len(expert_actions), candidate.anchor_step + prefix_k)
        expert_suffix = expert_actions[expert_start : expert_start + branch_budget]
        for operator in OPERATORS:
            branches[operator] = run_branch(
                env,
                task_name=task_name,
                snapshot=snapshots[prefix_k],
                original_chunk=chunk,
                prefix_k=prefix_k,
                operator=operator,
                model=model,
                normalization=normalization,
                device=device,
                branch_budget=branch_budget,
                execution_horizon=execution_horizon,
            )
        physical = run_branch(
            env,
            task_name=task_name,
            snapshot=snapshots[prefix_k],
            original_chunk=chunk,
            prefix_k=prefix_k,
            operator="privileged_scripted_recovery",
            model=model,
            normalization=normalization,
            device=device,
            branch_budget=branch_budget,
            execution_horizon=execution_horizon,
            expert_suffix=expert_suffix,
        )
        policy_recoverable = any(
            branches[name]["safe_success"] for name in OPERATORS if name != "nominal_continue"
        )
        recoverable_by_k[prefix_k] = policy_recoverable
        physical_by_k[prefix_k] = bool(physical["safe_success"])
        prefix_records.append(
            {
                "prefix_k": prefix_k,
                "policy_recoverable": policy_recoverable,
                "physical_recoverable": bool(physical["safe_success"]),
                "branches": branches,
                "physical_branch": physical,
                "snapshot_hash": state_sha256(snapshots[prefix_k]),
            }
        )

    def irreversible_boundary(values: dict[int, bool]) -> int:
        keys = sorted(values)
        for key in keys:
            if not any(values[later] for later in keys if later >= key):
                return key
        return len(chunk) + 1

    k_irrev_policy = irreversible_boundary(recoverable_by_k)
    k_irrev_physical = irreversible_boundary(physical_by_k)
    return {
        "candidate_id": candidate.candidate_id,
        "task": task_name,
        "demo_id": candidate.demo_id,
        "anchor_step": candidate.anchor_step,
        "chunk_length": len(chunk),
        "action_chunk_hash": action_chunk_sha256(chunk),
        "insertion_prefix": candidate.insertion_prefix,
        "severity": candidate.severity,
        "cause_type": TASK_SPECS[task_name].cause,
        "perturbation": applied.to_dict(),
        "replay_determinism": replay,
        "prefix_stride": prefix_stride,
        "prefixes": prefix_records,
        "k_irrev_policy": k_irrev_policy,
        "k_irrev_physical": k_irrev_physical,
        "intervention_window_policy": k_irrev_policy - candidate.insertion_prefix,
        "intervention_window_physical": k_irrev_physical - candidate.insertion_prefix,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    usable = []
    replay_passes = []
    windows = []
    retentions = []
    no_intervention_violations = 0
    oracle_violations = 0
    paired_rework_reductions = []
    operator_wins = {name: 0 for name in OPERATORS if name != "nominal_continue"}
    for record in records:
        replay_passes.append(bool(record["replay_determinism"]["passed"]))
        by_k = {item["prefix_k"]: item for item in record["prefixes"]}
        insertion = record["insertion_prefix"]
        trigger = by_k.get(insertion)
        if trigger is None:
            continue
        nominal = trigger["branches"]["nominal_continue"]
        unsafe = bool(nominal["violation"] or not nominal["task_recoverable"])
        if not unsafe:
            continue
        usable.append(record)
        no_intervention_violations += 1
        windows.append(record["intervention_window_policy"])
        latest_safe_candidates = [
            item
            for item in record["prefixes"]
            if item["prefix_k"] < record["k_irrev_policy"]
            and item["policy_recoverable"]
        ]
        chosen = max(latest_safe_candidates, key=lambda item: item["prefix_k"], default=trigger)
        retentions.append(
            chosen["prefix_k"] / record["chunk_length"]
            if latest_safe_candidates
            else 0.0
        )
        intervention_results = {
            name: chosen["branches"][name]
            for name in OPERATORS
            if name != "nominal_continue"
        }
        successful = {
            name: value for name, value in intervention_results.items() if value["safe_success"]
        }
        if successful:
            best_name, best_result = min(
                successful.items(), key=lambda item: (item[1]["rework_steps"], item[0])
            )
            operator_wins[best_name] += 1
            oracle_violations += int(not best_result["safe_success"])
        else:
            oracle_violations += 1
        full = chosen["branches"]["trim_and_replan"]
        if full["safe_success"] and successful and full["rework_steps"]:
            paired_rework_reductions.append(
                (full["rework_steps"] - best_result["rework_steps"])
                / full["rework_steps"]
            )
    count = len(usable)
    violation_reduction = None
    if no_intervention_violations:
        violation_reduction = (
            no_intervention_violations - oracle_violations
        ) / no_intervention_violations
    rework_reduction = (
        statistics.median(paired_rework_reductions)
        if paired_rework_reductions
        else None
    )
    return {
        "candidate_count": len(records),
        "usable_unsafe_or_near_unsafe_chunks": count,
        "replay_determinism_rate": sum(replay_passes) / len(replay_passes) if replay_passes else 0.0,
        "median_intervention_window": statistics.median(windows) if windows else None,
        "median_safe_prefix_retention": statistics.median(retentions) if retentions else None,
        "no_intervention_violation_count": no_intervention_violations,
        "oracle_violation_count": oracle_violations,
        "relative_violation_reduction": violation_reduction,
        "relative_rework_reduction_vs_same_trigger_full_replan": rework_reduction,
        "paired_rework_comparison_count": len(paired_rework_reductions),
        "operator_wins": operator_wins,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--model-seed", type=int, default=7)
    parser.add_argument("--demo-count", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--insertion-prefixes", nargs="+", type=int, default=[2, 6, 10])
    parser.add_argument("--prefix-stride", type=int, default=1)
    parser.add_argument("--branch-budget", type=int, default=96)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--libero-config", type=Path, default=DEFAULT_LIBERO_CONFIG)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, normalization, checkpoint_dir = load_policy(
        args.checkpoint_root, args.run_id, args.task, args.model_seed, device
    )
    env, _ = make_env(args.task, config_dir=args.libero_config)
    output_dir = args.log_root / args.run_id / "oracle_audit" / args.task
    labels_path = output_dir / "oracle_prefix_labels.jsonl"
    if labels_path.exists():
        raise FileExistsError(f"refusing to append to existing oracle labels: {labels_path}")
    records: list[dict[str, Any]] = []
    started = time.time()
    try:
        candidates = candidate_schedule(
            env,
            args.task,
            demo_count=args.demo_count,
            insertion_prefixes=args.insertion_prefixes,
            max_candidates=args.max_candidates,
        )
        if not candidates:
            raise RuntimeError(f"no phase candidates for {args.task}")
        atomic_write_json(
            output_dir / "schedule.json",
            {
                "task": args.task,
                "model_seed": args.model_seed,
                "checkpoint": str(checkpoint_dir),
                "candidates": [candidate.__dict__ for candidate in candidates],
            },
        )
        for index, candidate in enumerate(candidates):
            record = audit_candidate(
                env,
                task_name=args.task,
                candidate=candidate,
                model=model,
                normalization=normalization,
                device=device,
                branch_budget=args.branch_budget,
                execution_horizon=args.execution_horizon,
                prefix_stride=args.prefix_stride,
            )
            record["run_id"] = args.run_id
            record["model_seed"] = args.model_seed
            record["elapsed_seconds"] = time.time() - started
            append_jsonl(labels_path, record)
            records.append(record)
            if index == 0:
                atomic_write_json(
                    output_dir / "first_completed_oracle_result.json", record
                )
            print(
                f"ORACLE_CANDIDATE_COMPLETE task={args.task} index={index + 1}/{len(candidates)} "
                f"candidate={candidate.candidate_id} k_irrev={record['k_irrev_policy']} "
                f"window={record['intervention_window_policy']} replay={int(record['replay_determinism']['passed'])}",
                flush=True,
            )
    finally:
        env.close()
    summary = summarize(records)
    summary.update(
        {
            "run_id": args.run_id,
            "task": args.task,
            "model_seed": args.model_seed,
            "checkpoint": str(checkpoint_dir),
            "branch_budget": args.branch_budget,
            "prefix_stride": args.prefix_stride,
            "elapsed_seconds": time.time() - started,
        }
    )
    summary_path = output_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    print(f"ORACLE_AUDIT_COMPLETE summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
