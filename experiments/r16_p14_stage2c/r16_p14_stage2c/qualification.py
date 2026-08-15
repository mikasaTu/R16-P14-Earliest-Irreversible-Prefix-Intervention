from __future__ import annotations

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from r16_p14_stage2a.envs import current_observation, joint_qpos
from r16_p14_stage2a.settings import TASK_SPECS
from r16_p14_stage2b.io_utils import atomic_write_json, atomic_write_text, load_jsonl, write_once_jsonl
from r16_p14_stage2b.runtime import ActorBundle

from .contracts import monotone_observed_safety, qualify_cell, select_second_failure_family
from .runtime import CauseTracker, apply_perturbation, reconstruct_anchor
from .settings import (
    ACTOR_SEEDS,
    ALL_CANDIDATE_TASKS,
    ARTIFACT_ROOT,
    DETECTION_PREFIX,
    EXPERIMENT_ROOT,
    H_VALID,
    PATH_CLEARANCE_GRID,
    PATH_INDEX_GRID,
    PATH_TASK_CANDIDATES,
    TARGET_SHIFT_GRID,
    TARGET_SHIFT_TASK,
    MIRROR_EXPERIMENT_OUTPUTS,
)


def parameters(task: str) -> list[dict[str, Any]]:
    if task == TARGET_SHIFT_TASK:
        return [
            {"parameter_id": f"shift_{round(magnitude * 1000):03d}mm", "magnitude_m": magnitude}
            for magnitude in TARGET_SHIFT_GRID
        ]
    return [
        {
            "parameter_id": f"future_{index:02d}_lateral_{round(clearance * 1000):03d}mm",
            "future_index": index,
            "lateral_clearance_m": clearance,
        }
        for index in PATH_INDEX_GRID
        for clearance in PATH_CLEARANCE_GRID
    ]


def run_attempt(event: dict[str, Any], parameter: dict[str, Any], bundle: ActorBundle) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env, history, replay = reconstruct_anchor(event, bundle, capture_trace=False)
    try:
        if not replay["passed"]:
            raise ValueError(f"replay rejected: {replay['checks']}")
        spec = TASK_SPECS[event["task"]]
        assert spec.manipulated_joint
        initial = np.asarray(event["initial_manipulated_qpos"], dtype=np.float64)
        tracker = CauseTracker(
            task=event["task"], initial_object_z=float(initial[2]),
            previous_gripper=float(history.actions[-1][-1]),
            ever_lifted=bool(event["phase"]["ever_lifted"]),
        )
        perturbation = None
        prefix_rows = []
        for index, action in enumerate(np.asarray(event["original_chunk"], dtype=np.float32)[:H_VALID]):
            observation, _, _, _ = env.step(action)
            history.update(observation, action)
            tracker.observe_action(env, action)
            if index + 1 == DETECTION_PREFIX:
                perturbation = apply_perturbation(env, event, parameter)
                tracker.observe_injection(env, tuple(tuple(item) for item in perturbation["before_contacts"]))
                history.refresh_latest_observation(current_observation(env))
            if index + 1 >= DETECTION_PREFIX:
                prefix_rows.append({
                    "task": event["task"], "event_id": event["event_id"],
                    "actor_seed": event["actor_seed"], "init_state_id": event["init_state_id"],
                    "parameter_id": parameter["parameter_id"], "prefix_k": index + 1,
                    "S_obs": not tracker.violation, "cause_violation": tracker.violation,
                    "violation_type": tracker.violation_type,
                })
        if perturbation is None:
            raise RuntimeError("missing injection")
        monotonicity = monotone_observed_safety(prefix_rows)
        row = {
            "task": event["task"], "event_id": event["event_id"],
            "actor_seed": event["actor_seed"], "init_state_id": event["init_state_id"],
            "split": event["split"], "parameter": parameter,
            "parameter_id": parameter["parameter_id"],
            "replay_valid": True, "replay_checks": replay["checks"],
            "injection_valid": not tracker.injection_contact,
            "injection_contact": tracker.injection_contact,
            "manipulated_qpos_max_error": perturbation["manipulated_qpos_max_error"],
            "immediate_cause_violation": tracker.immediate_violation,
            "delayed_cause_violation": tracker.violation and not tracker.immediate_violation,
            "cause_violation": tracker.violation,
            "cause_violation_type": tracker.violation_type,
            "first_violation_offset": tracker.first_violation_offset,
            "monotonicity_valid": monotonicity["passed"],
            "monotonicity": monotonicity,
            "task_success_after_nominal_chunk": bool(env.check_success()),
            "intervention_method_outcome_read": False,
            "error": None if monotonicity["passed"] else "NONMONOTONIC_CAUSE_PREFIX",
        }
        return row, prefix_rows
    finally:
        env.close()


def shard_path(seed: int, task: str, event_id: str, parameter_id: str) -> Path:
    safe = f"{event_id}__{parameter_id}.json"
    return ARTIFACT_ROOT / "task_qualification/shards" / f"seed_{seed}" / task / safe


def run_shard(seed: int, task: str, device: str) -> None:
    events = [
        event for event in load_jsonl(ARTIFACT_ROOT / "actor_events/events.jsonl")
        if event["split"] == "calibration" and event["task"] == task and int(event["actor_seed"]) == seed
    ]
    bundle = ActorBundle.load(seed, device)
    for event in events:
        for parameter in parameters(task):
            path = shard_path(seed, task, event["event_id"], parameter["parameter_id"])
            if path.is_file():
                print(f"QUALIFY_RESUME {path.name}", flush=True)
                continue
            try:
                row, prefix_rows = run_attempt(event, parameter, bundle)
            except Exception as exc:
                row = {
                    "task": task, "event_id": event["event_id"], "actor_seed": seed,
                    "init_state_id": event["init_state_id"], "split": "calibration",
                    "parameter": parameter, "parameter_id": parameter["parameter_id"],
                    "replay_valid": False, "injection_valid": False, "injection_contact": False,
                    "monotonicity_valid": False,
                    "intervention_method_outcome_read": False,
                    "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                }
                prefix_rows = []
            atomic_write_json(path, {"complete": True, "attempt": row, "prefix_rows": prefix_rows})
            print(f"QUALIFY_DONE event={event['event_id']} parameter={parameter['parameter_id']} ok={int(row['error'] is None)}", flush=True)


def blocked_rank(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(item["error_count"] > 0),
        int(not item["checks"]["injection_contact_count_eq_0"]),
        abs(float(item["delayed_cause_violation_rate"]) - 0.55),
        -int(item["valid_event_count"]),
        str(item["parameter_id"]),
    )


def consolidate() -> None:
    root = ARTIFACT_ROOT / "task_qualification/shards"
    attempts, prefix_rows = [], []
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("complete"):
            attempts.append(payload["attempt"])
            prefix_rows.extend(payload["prefix_rows"])
    expected = 0
    event_rows = load_jsonl(ARTIFACT_ROOT / "actor_events/events.jsonl")
    for event in event_rows:
        if event["split"] == "calibration":
            expected += len(parameters(event["task"]))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        grouped[(row["task"], row["parameter_id"])].append(row)
    cells = []
    for task in ALL_CANDIDATE_TASKS:
        for parameter in parameters(task):
            summary = qualify_cell(grouped[(task, parameter["parameter_id"])])
            cells.append({"task": task, "parameter": parameter, "parameter_id": parameter["parameter_id"], **summary})
    selected_second = select_second_failure_family(cells, PATH_TASK_CANDIDATES)
    target_qualified = [item for item in cells if item["task"] == TARGET_SHIFT_TASK and item["qualifies"]]
    upstream_pass = len(target_qualified) >= 2 and selected_second is not None
    forced_second = selected_second is None
    if selected_second is None:
        selected_second = PATH_TASK_CANDIDATES[0]
    frozen = {}
    for task in (TARGET_SHIFT_TASK, selected_second):
        task_cells = [item for item in cells if item["task"] == task]
        qualified = sorted([item for item in task_cells if item["qualifies"]], key=lambda item: (abs(item["delayed_cause_violation_rate"] - 0.55), item["parameter_id"]))
        chosen = qualified[:2] if len(qualified) >= 2 else sorted(task_cells, key=blocked_rank)[:2]
        frozen[task] = {
            "status": "qualified" if len(qualified) >= 2 else "forced_diagnostic_after_failed_gate",
            "qualifying_parameter_count": len(qualified),
            "parameters": [{"parameter": item["parameter"], "parameter_id": item["parameter_id"], "qualifies": item["qualifies"], "metrics": item} for item in chosen],
        }
    summary = {
        "schema_version": 1,
        "status": "PASS" if upstream_pass else "BLOCKED",
        "failure_label": None if upstream_pass else ("BLOCKED_BY_SECOND_FAILURE_FAMILY" if forced_second else "BLOCKED_BY_MINIMUM_DATA"),
        "expected_attempts_from_admitted_events": expected,
        "completed_attempts": len(attempts),
        "missing_attempts": max(0, expected - len(attempts)),
        "cells": cells,
        "target_shift_two_severities": len(target_qualified) >= 2,
        "second_failure_family": selected_second,
        "second_failure_family_pass": not forced_second,
        "selection_uses_method_gain": False,
        "evaluation_outcomes_read": False,
        "continue_all_experiments_after_gate_failure": True,
        "frozen": frozen,
    }
    output = ARTIFACT_ROOT / "task_qualification"
    write_once_jsonl(output / "attempts.jsonl", attempts)
    write_once_jsonl(output / "prefix_safety.jsonl", prefix_rows)
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(output / "frozen_parameters.json", {"schema_version": 1, "selected_tasks": [TARGET_SHIFT_TASK, selected_second], "upstream_qualification_pass": upstream_pass, "tasks": frozen})
    lines = ["# Stage-2C perturbation qualification", "", f"Status: **{summary['status']}**; downstream matrices continue regardless, as requested.", "", "| task | parameter | valid | immediate | delayed | interior | replay | errors | qualifies |", "|---|---|---:|---:|---:|---:|---:|---:|:---:|"]
    for item in cells:
        lines.append(f"| {item['task']} | {item['parameter_id']} | {item['valid_event_count']} | {item['immediate_cause_violation_rate']:.3f} | {item['delayed_cause_violation_rate']:.3f} | {item['interior_violation_fraction']:.3f} | {item['replay_rate']:.3f} | {item['error_count']} | {'yes' if item['qualifies'] else 'no'} |")
    lines.append("")
    atomic_write_text(output / "report.md", "\n".join(lines))
    if MIRROR_EXPERIMENT_OUTPUTS:
        exp = EXPERIMENT_ROOT / "task_qualification"
        exp.mkdir(parents=True, exist_ok=True)
        atomic_write_json(exp / "summary.json", summary)
        atomic_write_json(exp / "frozen_parameters.json", {"schema_version": 1, "selected_tasks": [TARGET_SHIFT_TASK, selected_second], "upstream_qualification_pass": upstream_pass, "tasks": frozen})
        atomic_write_text(exp / "report.md", "\n".join(lines))
    print(json.dumps({"status": summary["status"], "attempts": len(attempts), "second_task": selected_second}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=ACTOR_SEEDS)
    parser.add_argument("--task", choices=ALL_CANDIDATE_TASKS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--consolidate", action="store_true")
    args = parser.parse_args()
    if args.consolidate:
        consolidate()
        return
    if args.seed is None or args.task is None:
        parser.error("--seed and --task are required unless --consolidate is used")
    run_shard(args.seed, args.task, args.device)


if __name__ == "__main__":
    main()
