from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from r16_p14_stage2a.envs import contact_pairs, joint_qpos
from r16_p14_stage2a.settings import TASK_SPECS

from .io_utils import atomic_write_json, load_jsonl, write_once_jsonl
from .runtime import (
    ActorBundle,
    BranchContext,
    branch_snapshot,
    is_catastrophic_object_drop,
    normalized_disagreement,
    reconstruct_to_prefix,
    severity_id,
    target_distance,
)
from .settings import (
    ACTOR_SEEDS,
    ARTIFACT_ROOT,
    ATLAS_EVALUATION_IDS,
    DETECTION_PREFIX,
    POST_DETECTION_BUDGET_CAP,
    PRIMARY_TASKS,
)


def frozen_severities() -> dict[str, tuple[float, float]]:
    payload = json.loads(
        (ARTIFACT_ROOT / "perturbation_qualification/frozen_parameters.json").read_text()
    )
    result: dict[str, tuple[float, float]] = {}
    for task in PRIMARY_TASKS:
        values = tuple(
            float(item["severity_m"])
            for item in payload["tasks"][task]["severities"]
        )
        if len(values) != 2 or values[0] == values[1]:
            raise ValueError(f"{task} requires two distinct frozen severities: {values}")
        result[task] = values
    return result


def select_evaluation_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        event
        for event in events
        if int(event["init_state_id"]) in ATLAS_EVALUATION_IDS
    ]
    selected.sort(key=lambda event: (event["task"], int(event["actor_seed"]), int(event["init_state_id"])))
    if len(selected) > 60:
        raise ValueError(f"bounded pilot has {len(selected)} events, maximum is 60")
    if any(not event.get("source_is_actor_generated_chunk") for event in selected):
        raise ValueError("formal Atlas nominal prefix is not actor generated")
    if any(event.get("source_is_demonstration_chunk") for event in selected):
        raise ValueError("demonstration chunks are forbidden in formal Atlas")
    return selected


def assign_severity(event: dict[str, Any], severities: dict[str, tuple[float, float]]) -> float:
    """Alternate severity within each task/seed's ordered evaluation split."""
    index = ATLAS_EVALUATION_IDS.index(int(event["init_state_id"]))
    return float(severities[event["task"]][index % 2])


def event_budget(event: dict[str, Any]) -> int:
    spec = TASK_SPECS[event["task"]]
    return max(
        0,
        min(
            POST_DETECTION_BUDGET_CAP,
            int(spec.horizon) - int(event["anchor_global_step"]) - DETECTION_PREFIX,
        ),
    )


def branch_contract(prefix_k: int, horizon: int, post_budget: int) -> dict[str, int]:
    if not DETECTION_PREFIX <= prefix_k <= horizon:
        raise ValueError(f"prefix {prefix_k} outside [{DETECTION_PREFIX}, {horizon}]")
    retained = prefix_k - DETECTION_PREFIX
    return {
        "prefix_k": int(prefix_k),
        "old_nominal_actions_retained_after_detection": retained,
        "nominal_actions_discarded": horizon - prefix_k,
        "post_detection_budget": int(post_budget),
        "new_action_budget": max(0, int(post_budget) - retained),
    }


def object_drop(context: BranchContext, event: dict[str, Any]) -> bool:
    spec = TASK_SPECS[event["task"]]
    assert spec.manipulated_joint
    position = joint_qpos(context.env, spec.manipulated_joint)
    initial = np.asarray(event["initial_manipulated_qpos"], dtype=np.float64)
    success = bool(context.env.check_success())
    return is_catastrophic_object_drop(
        task=event["task"],
        initial_object_z=float(initial[2]),
        current_object_z=float(position[2]),
        target_distance_value=target_distance(
            context.env,
            event["task"],
            np.asarray(event["goal_qpos"], dtype=np.float64),
        ),
        ever_stably_lifted=bool(context.tracker.ever_lifted),
        valid_release=False,
        success=success,
    )


def complete_with_h1_replan(
    context: BranchContext,
    *,
    event: dict[str, Any],
    bundle: ActorBundle,
    first_chunk: np.ndarray,
    action_budget: int,
) -> dict[str, Any]:
    actions: list[np.ndarray] = []
    chunk_hashes: list[str] = []
    contact_step_count = 0
    success = bool(context.env.check_success())
    next_chunk = np.asarray(first_chunk, dtype=np.float32)
    # A cause violation is absorbing as a label, not a simulator terminal.
    # Continue to the common budget so task success, rework, and policy calls
    # remain comparable rather than being artificially truncated at failure.
    while len(actions) < action_budget and not success:
        if actions:
            next_chunk = bundle.predict(
                context.history.state_array(), context.history.action_array(), event["task"]
            )
        from .runtime import chunk_hash

        chunk_hashes.append(chunk_hash(next_chunk))
        action = np.asarray(next_chunk[0], dtype=np.float32)
        observation, _, _, _ = context.env.step(action)
        context.history.update(observation, action)
        context.tracker.observe_action(context.env, action)
        actions.append(action.copy())
        contact_step_count += int(bool(contact_pairs(context.env)))
        success = bool(context.env.check_success())
    array = np.asarray(actions, dtype=np.float32).reshape(-1, 7)
    return {
        "task_success": success,
        "cause_violation": bool(context.tracker.violation),
        "cause_violation_type": context.tracker.violation_type,
        "new_non_nominal_actions": len(actions),
        "policy_calls": len(chunk_hashes),
        "replan_chunk_hashes": chunk_hashes,
        "first_replan_chunk_hash": chunk_hashes[0] if chunk_hashes else None,
        "last_replan_chunk_hash": chunk_hashes[-1] if chunk_hashes else None,
        "new_action_path_length": (
            float(np.linalg.norm(array[:, :3], axis=1).sum()) if len(array) else 0.0
        ),
        "contact_step_count": contact_step_count,
        "new_actions": array.tolist(),
    }


def run_branch(
    event: dict[str, Any],
    *,
    severity_m: float,
    prefix_k: int,
    bundle: ActorBundle,
) -> dict[str, Any]:
    start = time.perf_counter()
    horizon = int(event["effective_horizon"])
    budget = event_budget(event)
    contract = branch_contract(prefix_k, horizon, budget)
    context = reconstruct_to_prefix(
        event, severity_m=severity_m, prefix_k=prefix_k, bundle=bundle
    )
    try:
        snapshot = branch_snapshot(context, bundle)
        old_chunk = np.asarray(event["original_chunk"], dtype=np.float32)
        disagreement = (
            normalized_disagreement(old_chunk[prefix_k], snapshot["replanned_chunk"][0], bundle)
            if prefix_k < horizon
            else None
        )
        pre_violation = bool(context.tracker.violation)
        if pre_violation:
            completion = {
                "task_success": bool(context.env.check_success()),
                "cause_violation": True,
                "cause_violation_type": context.tracker.violation_type,
                "new_non_nominal_actions": 0,
                "policy_calls": 0,
                "replan_chunk_hashes": [],
                "first_replan_chunk_hash": None,
                "last_replan_chunk_hash": None,
                "new_action_path_length": 0.0,
                "contact_step_count": 0,
                "new_actions": [],
            }
        elif context.env.check_success():
            completion = {
                "task_success": True,
                "cause_violation": False,
                "cause_violation_type": None,
                "new_non_nominal_actions": 0,
                "policy_calls": 0,
                "replan_chunk_hashes": [],
                "first_replan_chunk_hash": snapshot["replanned_chunk_hash"],
                "last_replan_chunk_hash": snapshot["replanned_chunk_hash"],
                "new_action_path_length": 0.0,
                "contact_step_count": 0,
                "new_actions": [],
            }
        else:
            completion = complete_with_h1_replan(
                context,
                event=event,
                bundle=bundle,
                first_chunk=np.asarray(snapshot["replanned_chunk"], dtype=np.float32),
                action_budget=contract["new_action_budget"],
            )
        post_actions = contract["old_nominal_actions_retained_after_detection"] + completion["new_non_nominal_actions"]
        timeout = bool(
            not completion["task_success"]
            and post_actions >= budget
        )
        old_actions = old_chunk[DETECTION_PREFIX:prefix_k]
        old_path = (
            float(np.linalg.norm(old_actions[:, :3], axis=1).sum()) if len(old_actions) else 0.0
        )
        spec = TASK_SPECS[event["task"]]
        assert spec.manipulated_joint
        final_object = joint_qpos(context.env, spec.manipulated_joint)
        return {
            "schema_version": 1,
            "record_type": "safe_replanability_atlas_branch",
            "event_id": event["event_id"],
            "task": event["task"],
            "actor_seed": int(event["actor_seed"]),
            "init_state_id": int(event["init_state_id"]),
            "split": event["split"],
            "severity_m": float(severity_m),
            "severity_id": severity_id(event["task"], severity_m),
            "effective_horizon": horizon,
            "detection_prefix": DETECTION_PREFIX,
            **contract,
            "S_k": int(not pre_violation),
            "pre_replan_cause_violation": pre_violation,
            "cause_violation": bool(completion["cause_violation"]),
            "cause_violation_type": completion["cause_violation_type"],
            "task_success": bool(completion["task_success"]),
            "safe_success": bool(completion["task_success"] and not completion["cause_violation"]),
            "timeout": timeout,
            "new_non_nominal_actions": int(completion["new_non_nominal_actions"]),
            "policy_calls": int(completion["policy_calls"]),
            "diagnostic_policy_calls": int(pre_violation),
            "total_actions_from_anchor": int(prefix_k + completion["new_non_nominal_actions"]),
            "post_detection_actions": int(post_actions),
            "completion_steps": int(post_actions),
            "path_length": float(old_path + completion["new_action_path_length"]),
            "contact_events": int(completion["contact_step_count"]),
            "object_drop": object_drop(context, event),
            "task_phase": {
                "ever_lifted": bool(context.tracker.ever_lifted),
                "final_object_qpos": final_object.tolist(),
                "task_success": bool(completion["task_success"]),
            },
            "original_chunk_hash": event["original_chunk_hash"],
            "original_chunk_hash_verified": bool(
                context.anchor_replay["checks"]["original_chunk_hash_match"]
            ),
            "fresh_chunk_hash_at_prefix": snapshot["replanned_chunk_hash"],
            "first_replan_chunk_hash": completion["first_replan_chunk_hash"],
            "last_replan_chunk_hash": completion["last_replan_chunk_hash"],
            "cached_fresh_normalized_disagreement": disagreement,
            "source_is_actor_generated_chunk": True,
            "source_is_demonstration_chunk": False,
            "upstream_gate_override": bool(event.get("upstream_gate_override")),
            "branch_fresh_environment": True,
            "wall_time_seconds": time.perf_counter() - start,
            "error": None,
        }
    finally:
        context.close()


def event_output_path(event: dict[str, Any], *, smoke: bool) -> Path:
    directory = ARTIFACT_ROOT / "atlas_pilot" / ("smoke" if smoke else "events")
    return directory / f"{event['event_id']}.branches.jsonl"


def event_complete_path(event: dict[str, Any], *, smoke: bool) -> Path:
    return event_output_path(event, smoke=smoke).with_suffix(".complete.json")


def event_is_complete(event: dict[str, Any], *, smoke: bool = False) -> bool:
    output = event_output_path(event, smoke=smoke)
    marker = event_complete_path(event, smoke=smoke)
    if not output.is_file() or not marker.is_file():
        return False
    payload = json.loads(marker.read_text())
    rows = load_jsonl(output)
    expected = 1 if smoke else int(event["effective_horizon"]) - DETECTION_PREFIX + 1
    return payload.get("complete") is True and payload.get("branch_count") == expected and len(rows) == expected


def pending_event_ids(events: Iterable[dict[str, Any]], *, smoke: bool = False) -> list[str]:
    return [event["event_id"] for event in events if not event_is_complete(event, smoke=smoke)]


def run_event(
    event: dict[str, Any],
    *,
    severity_m: float,
    bundle: ActorBundle,
    smoke: bool = False,
) -> None:
    if event_is_complete(event, smoke=smoke):
        print(f"ATLAS_EVENT_ALREADY_COMPLETE event={event['event_id']} smoke={int(smoke)}")
        return
    horizon = int(event["effective_horizon"])
    prefixes = [DETECTION_PREFIX] if smoke else list(range(DETECTION_PREFIX, horizon + 1))
    records: list[dict[str, Any]] = []
    for prefix_k in prefixes:
        try:
            record = run_branch(
                event,
                severity_m=severity_m,
                prefix_k=prefix_k,
                bundle=bundle,
            )
        except Exception as exc:
            record = {
                "schema_version": 1,
                "record_type": "safe_replanability_atlas_branch",
                "event_id": event["event_id"],
                "task": event["task"],
                "actor_seed": int(event["actor_seed"]),
                "init_state_id": int(event["init_state_id"]),
                "severity_m": float(severity_m),
                "severity_id": severity_id(event["task"], severity_m),
                "prefix_k": prefix_k,
                "effective_horizon": horizon,
                "source_is_actor_generated_chunk": bool(event.get("source_is_actor_generated_chunk")),
                "source_is_demonstration_chunk": bool(event.get("source_is_demonstration_chunk")),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        records.append(record)
        print(
            f"ATLAS_BRANCH event={event['event_id']} k={prefix_k} "
            f"safe={record.get('S_k')} success={int(record.get('safe_success', False))} "
            f"error={record.get('error')}",
            flush=True,
        )
    output = event_output_path(event, smoke=smoke)
    write_once_jsonl(output, records)
    atomic_write_json(
        event_complete_path(event, smoke=smoke),
        {
            "complete": True,
            "event_id": event["event_id"],
            "branch_count": len(records),
            "error_count": sum(row.get("error") is not None for row in records),
            "severity_m": severity_m,
            "smoke": smoke,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=ACTOR_SEEDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    events = select_evaluation_events(load_jsonl(ARTIFACT_ROOT / "actor_events/events.jsonl"))
    severities = frozen_severities()
    if args.smoke:
        if not events:
            raise RuntimeError("no eligible evaluation event for integration smoke")
        event = events[0]
        bundle = ActorBundle.load(int(event["actor_seed"]), args.device)
        run_event(
            event,
            severity_m=assign_severity(event, severities),
            bundle=bundle,
            smoke=True,
        )
        return
    if args.seed is None:
        parser.error("provide --seed or --smoke")
    selected = [event for event in events if int(event["actor_seed"]) == args.seed]
    bundle = ActorBundle.load(args.seed, args.device)
    for event in selected:
        run_event(
            event,
            severity_m=assign_severity(event, severities),
            bundle=bundle,
            smoke=False,
        )


if __name__ == "__main__":
    main()
