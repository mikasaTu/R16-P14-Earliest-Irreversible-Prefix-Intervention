from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from r16_p14_stage2a.envs import contact_pairs, joint_qpos
from r16_p14_stage2a.settings import TASK_SPECS

from .atlas_runner import complete_with_h1_replan, event_budget, object_drop
from .io_utils import atomic_write_json, atomic_write_text, load_jsonl, write_once_jsonl
from .runtime import ActorBundle, branch_snapshot, reconstruct_to_prefix
from .settings import ACTOR_SEEDS, ARTIFACT_ROOT, DETECTION_PREFIX, EXPERIMENT_ROOT, PRIMARY_TASKS


OPERATORS = (
    "full_replan",
    "hold_one_step_replan",
    "bounded_rollback_replan",
    "cause_specific_local_repair",
)


def position_contract(event_metric: dict[str, Any]) -> list[tuple[str, int | None]]:
    return [
        ("d", DETECTION_PREFIX),
        ("k_best", event_metric.get("k_best")),
        ("k_last_safe", event_metric.get("k_last_safe")),
    ]


def neutral_hold(context) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[-1] = float(context.history.actions[-1][-1])
    return action


def operator_action(operator: str, context, event: dict[str, Any], prefix_k: int) -> np.ndarray | None:
    chunk = np.asarray(event["original_chunk"], dtype=np.float32)
    if operator == "full_replan":
        return None
    if operator == "hold_one_step_replan":
        return chunk[prefix_k].copy() if prefix_k < int(event["effective_horizon"]) else neutral_hold(context)
    if operator == "bounded_rollback_replan":
        source = chunk[max(0, prefix_k - 1)]
        action = np.zeros(7, dtype=np.float32)
        action[:6] = np.clip(-0.5 * source[:6], -1.0, 1.0)
        action[-1] = float(context.history.actions[-1][-1])
        return action
    if operator != "cause_specific_local_repair":
        raise KeyError(operator)
    spec = TASK_SPECS[event["task"]]
    assert spec.manipulated_joint
    manipulated = joint_qpos(context.env, spec.manipulated_joint)
    action = np.zeros(7, dtype=np.float32)
    if event["task"] == PRIMARY_TASKS[0]:
        assert spec.target_joint
        target = joint_qpos(context.env, spec.target_joint)
        action[:2] = np.clip((target[:2] - manipulated[:2]) * 10.0, -1.0, 1.0)
        action[2] = 0.05
    else:
        assert spec.obstacle_joint
        obstacle = joint_qpos(context.env, spec.obstacle_joint)
        away = manipulated[:2] - obstacle[:2]
        norm = float(np.linalg.norm(away))
        if norm <= 1e-9:
            away = np.asarray([1.0, 0.0], dtype=np.float64)
        else:
            away /= norm
        action[:2] = np.clip(away * 0.5, -1.0, 1.0)
        action[2] = 0.15
    action[-1] = 1.0
    return action


def execute_operator_action(context, action: np.ndarray) -> dict[str, Any]:
    observation, _, _, _ = context.env.step(action)
    context.history.update(observation, action)
    context.tracker.observe_action(context.env, action)
    return {
        "action": np.asarray(action, dtype=np.float32).tolist(),
        "path_length": float(np.linalg.norm(np.asarray(action, dtype=np.float32)[:3])),
        "contact_event": int(bool(contact_pairs(context.env))),
    }


def run_operator(
    event: dict[str, Any],
    *,
    severity_m: float,
    prefix_k: int,
    position_label: str,
    operator: str,
    bundle: ActorBundle,
) -> dict[str, Any]:
    started = time.perf_counter()
    budget = event_budget(event)
    retained = prefix_k - DETECTION_PREFIX
    available_after_prefix = max(0, budget - retained)
    context = reconstruct_to_prefix(event, severity_m=severity_m, prefix_k=prefix_k, bundle=bundle)
    try:
        pre_violation = bool(context.tracker.violation)
        op_action = operator_action(operator, context, event, prefix_k)
        op_record = {"action": None, "path_length": 0.0, "contact_event": 0}
        if (
            op_action is not None
            and available_after_prefix > 0
            and not pre_violation
            and not context.env.check_success()
        ):
            op_record = execute_operator_action(context, op_action)
        operator_action_count = int(op_record["action"] is not None)
        remaining = max(0, available_after_prefix - operator_action_count)
        if pre_violation or context.tracker.violation or context.env.check_success() or remaining == 0:
            completion = {
                "task_success": bool(context.env.check_success()),
                "cause_violation": bool(context.tracker.violation),
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
        else:
            snapshot = branch_snapshot(context, bundle)
            completion = complete_with_h1_replan(
                context,
                event=event,
                bundle=bundle,
                first_chunk=np.asarray(snapshot["replanned_chunk"], dtype=np.float32),
                action_budget=remaining,
            )
        spent = retained + operator_action_count + int(completion["new_non_nominal_actions"])
        timeout = bool(
            not completion["task_success"] and not completion["cause_violation"] and spent >= budget
        )
        return {
            "schema_version": 1,
            "record_type": "secondary_operator_branch",
            "event_id": event["event_id"],
            "task": event["task"],
            "actor_seed": int(event["actor_seed"]),
            "init_state_id": int(event["init_state_id"]),
            "severity_m": float(severity_m),
            "position_label": position_label,
            "prefix_k": int(prefix_k),
            "operator": operator,
            "post_detection_budget": budget,
            "old_nominal_actions_retained_after_detection": retained,
            "operator_action_count": operator_action_count,
            "operator_action": op_record["action"],
            "actor_actions": int(completion["new_non_nominal_actions"]),
            "total_post_detection_actions": spent,
            "policy_calls": int(completion["policy_calls"]),
            "pre_operator_cause_violation": pre_violation,
            "cause_violation": bool(completion["cause_violation"]),
            "cause_violation_type": completion["cause_violation_type"],
            "task_success": bool(completion["task_success"]),
            "safe_success": bool(completion["task_success"] and not completion["cause_violation"]),
            "timeout": timeout,
            "path_length": float(op_record["path_length"] + completion["new_action_path_length"]),
            "contact_events": int(op_record["contact_event"] + completion["contact_step_count"]),
            "object_drop": object_drop(context, event),
            "original_chunk_hash": event["original_chunk_hash"],
            "source_is_actor_generated_chunk": True,
            "identical_budget_contract": spent <= budget,
            "wall_time_seconds": time.perf_counter() - started,
            "error": None,
        }
    finally:
        context.close()


def output_path(event_id: str) -> Path:
    return ARTIFACT_ROOT / "operator_audit/events" / f"{event_id}.operators.jsonl"


def complete_path(event_id: str) -> Path:
    return output_path(event_id).with_suffix(".complete.json")


def event_complete(event_metric: dict[str, Any]) -> bool:
    path = output_path(event_metric["event_id"])
    marker = complete_path(event_metric["event_id"])
    if not path.is_file() or not marker.is_file():
        return False
    expected_positions = sum(position is not None for _, position in position_contract(event_metric))
    expected = expected_positions * len(OPERATORS)
    rows = load_jsonl(path)
    payload = json.loads(marker.read_text())
    return payload.get("complete") is True and payload.get("record_count") == expected and len(rows) == expected


def run_event(event: dict[str, Any], metric: dict[str, Any], bundle: ActorBundle) -> None:
    if event_complete(metric):
        print(f"OPERATOR_EVENT_ALREADY_COMPLETE event={event['event_id']}")
        return
    severity = float(metric["severity_m"])
    records: list[dict[str, Any]] = []
    for label, prefix in position_contract(metric):
        if prefix is None:
            continue
        for operator in OPERATORS:
            try:
                record = run_operator(
                    event,
                    severity_m=severity,
                    prefix_k=int(prefix),
                    position_label=label,
                    operator=operator,
                    bundle=bundle,
                )
            except Exception as exc:
                record = {
                    "record_type": "secondary_operator_branch",
                    "event_id": event["event_id"],
                    "task": event["task"],
                    "actor_seed": int(event["actor_seed"]),
                    "init_state_id": int(event["init_state_id"]),
                    "severity_m": severity,
                    "position_label": label,
                    "prefix_k": int(prefix),
                    "operator": operator,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            records.append(record)
            print(
                f"OPERATOR_BRANCH event={event['event_id']} pos={label} op={operator} "
                f"safe_success={int(record.get('safe_success', False))} error={record.get('error')}",
                flush=True,
            )
    write_once_jsonl(output_path(event["event_id"]), records)
    atomic_write_json(
        complete_path(event["event_id"]),
        {
            "complete": True,
            "event_id": event["event_id"],
            "record_count": len(records),
            "error_count": sum(record.get("error") is not None for record in records),
        },
    )


def aggregate() -> dict[str, Any]:
    output = ARTIFACT_ROOT / "operator_audit"
    records: list[dict[str, Any]] = []
    for path in sorted((output / "events").glob("*.operators.jsonl")):
        records.extend(load_jsonl(path))
    valid = [row for row in records if row.get("error") is None]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        groups[(row["event_id"], row["position_label"])].append(row)
    unique_wins: dict[str, set[str]] = {operator: set() for operator in OPERATORS}
    winner_tasks: dict[str, set[str]] = {operator: set() for operator in OPERATORS}
    group_results = []
    for (event_id, position), rows in sorted(groups.items()):
        successes = [row for row in rows if row["safe_success"]]
        winner = successes[0]["operator"] if len(successes) == 1 else None
        if winner:
            unique_wins[winner].add(event_id)
            winner_tasks[winner].add(successes[0]["task"])
        group_results.append(
            {
                "event_id": event_id,
                "position_label": position,
                "unique_safe_winner": winner,
                "safe_success_count": len(successes),
            }
        )
    events = {row["event_id"] for row in valid}
    rates = {
        operator: (len(unique_wins[operator]) / len(events) if events else 0.0)
        for operator in OPERATORS
    }
    qualifying = [operator for operator in OPERATORS if rates[operator] >= 0.10]
    budget_equal = all(row["identical_budget_contract"] for row in valid) and all(
        len({row["post_detection_budget"] for row in rows}) == 1 for rows in groups.values()
    )
    checks = {
        "two_operators_unique_safe_win_ge_10pct": len(qualifying) >= 2,
        "unique_winners_in_both_tasks": len(qualifying) >= 2 and all(
            set(PRIMARY_TASKS).issubset(winner_tasks[operator]) for operator in qualifying[:2]
        ),
        "identical_action_and_policy_budget": budget_equal,
        "no_errors": len(valid) == len(records),
    }
    signal = all(checks.values())
    full = [row for row in valid if row["operator"] == "full_replan"]
    repair = [row for row in valid if row["operator"] == "cause_specific_local_repair"]
    full_rate = float(np.mean([row["safe_success"] for row in full])) if full else 0.0
    repair_rate = float(np.mean([row["safe_success"] for row in repair])) if repair else 0.0
    repair_tasks = {
        task: (
            float(np.mean([row["safe_success"] for row in repair if row["task"] == task]))
            > float(np.mean([row["safe_success"] for row in full if row["task"] == task]))
        ) if any(row["task"] == task for row in repair) and any(row["task"] == task for row in full) else False
        for task in PRIMARY_TASKS
    }
    local_repair_signal = repair_rate > full_rate and all(repair_tasks.values())
    summary = {
        "schema_version": 1,
        "record_count": len(records),
        "valid_record_count": len(valid),
        "event_count": len(events),
        "unique_safe_win_event_counts": {operator: len(values) for operator, values in unique_wins.items()},
        "unique_safe_win_event_rates": rates,
        "unique_winner_tasks": {operator: sorted(values) for operator, values in winner_tasks.items()},
        "qualifying_operators": qualifying,
        "checks": checks,
        "operator_router": "PILOT_SIGNAL" if signal else "NO_OPERATOR_ROUTING_SIGNAL",
        "track_a_local_repair": "PILOT_SIGNAL" if local_repair_signal else "NO_SIGNAL",
        "full_replan_safe_success_rate": full_rate,
        "local_repair_safe_success_rate": repair_rate,
        "local_repair_task_direction": repair_tasks,
        "group_results": group_results,
    }
    write_once_jsonl(output / "raw_operator_branches.jsonl", records)
    atomic_write_json(output / "summary.json", summary)
    lines = [
        "# Phase F — secondary operator audit",
        "",
        f"Operator-router result: **{summary['operator_router']}** from {len(events)} events.",
        f"Cause-specific local-repair result: **{summary['track_a_local_repair']}**.",
        "",
        "| operator | unique-safe-win events | rate | tasks |",
        "|---|---:|---:|---|",
    ]
    for operator in OPERATORS:
        lines.append(
            f"| {operator} | {len(unique_wins[operator])} | {rates[operator]:.3f} | "
            f"{', '.join(sorted(winner_tasks[operator])) or 'none'} |"
        )
    lines.extend(["", f"Identical budget contract: `{budget_equal}`.", ""])
    report = "\n".join(lines)
    atomic_write_text(output / "report.md", report)
    atomic_write_json(EXPERIMENT_ROOT / "operator_audit/summary.json", summary)
    atomic_write_text(EXPERIMENT_ROOT / "operator_audit/report.md", report)
    print(json.dumps({key: value for key, value in summary.items() if key != "group_results"}, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=ACTOR_SEEDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
        return
    if args.seed is None:
        parser.error("provide --seed or --aggregate")
    metrics = load_jsonl(ARTIFACT_ROOT / "atlas_pilot/event_metrics.jsonl")
    events = {event["event_id"]: event for event in load_jsonl(ARTIFACT_ROOT / "actor_events/events.jsonl")}
    selected = [row for row in metrics if int(row["actor_seed"]) == args.seed]
    bundle = ActorBundle.load(args.seed, args.device)
    for metric in selected:
        run_event(events[metric["event_id"]], metric, bundle)


if __name__ == "__main__":
    main()
