from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, atomic_write_jsonl, atomic_write_text


ACTOR_SEEDS = (7, 17, 29)
DETECTION_PREFIX = 2
BRANCH_PREFIXES = tuple(range(DETECTION_PREFIX, 16))
SECONDARY_OPERATORS = (
    "hold_and_replan",
    "bounded_rollback_and_replan",
    "cause_specific_local_repair",
)
SECONDARY_POSITIONS = ("d", "k_best", "k_last_safe")


def severity_value(parameters: dict[str, float | int]) -> float:
    for key in (
        "shift_magnitude_m",
        "obstacle_clearance_m",
        "blocker_lateral_offset_m",
    ):
        if key in parameters:
            return float(parameters[key])
    raise ValueError(f"configuration has no registered severity field: {parameters}")


def select_two_severities(
    task: str,
    selected_config_id: str,
    grid: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    qualified = sorted(
        (
            item
            for item in grid
            if item["task"] == task and item["qualifies"]
        ),
        key=lambda item: item["selection_rank"],
    )
    primary = next(
        item for item in qualified if item["config_id"] == selected_config_id
    )
    primary_severity = severity_value(primary["parameters"])
    secondary = next(
        (
            item
            for item in qualified
            if severity_value(item["parameters"]) != primary_severity
        ),
        None,
    )
    if secondary is None:
        raise ValueError(f"{task} has no second qualified severity")
    return [
        {
            "role": role,
            "config_id": item["config_id"],
            "parameters": item["parameters"],
            "severity_value": severity_value(item["parameters"]),
            "calibration_immediate_violation_rate": item["immediate_violation_rate"],
            "calibration_delayed_nominal_violation_rate": item["nominal_delayed_violation_rate"],
            "calibration_replay_rate": item["replay_pass_rate"],
        }
        for role, item in zip(("primary", "secondary"), (primary, secondary), strict=True)
    ]


def build_schedule(
    actor: dict[str, Any],
    perturbations: dict[str, Any],
    grid: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tasks = [
        task
        for task, item in perturbations["tasks"].items()
        if item["status"] == "qualified"
        and actor["tasks"][task]["qualification_status"] == "qualified"
    ]
    events: list[dict[str, Any]] = []
    task_schedules: dict[str, Any] = {}
    for task in tasks:
        selected_id = perturbations["tasks"][task]["selected_config_id"]
        severities = select_two_severities(task, selected_id, grid)
        task_events: list[dict[str, Any]] = []
        # Evaluation has exactly 30 disjoint demonstrations, balanced 15/15.
        for offset, demo_id in enumerate(range(10, 40)):
            severity = severities[offset % len(severities)]
            task_events.append(
                {
                    "event_id": f"{task}__evaluation__demo{demo_id:02d}__{severity['config_id']}",
                    "task": task,
                    "split": "evaluation",
                    "demo_id": demo_id,
                    "config_id": severity["config_id"],
                    "severity_role": severity["role"],
                    "parameters": severity["parameters"],
                    "selection_uses_method_gain": False,
                }
            )
        # Ten held-out demonstrations crossed with two frozen severities give
        # twenty held-out events without looking outside IDs 40--49.
        for demo_id in range(40, 50):
            for severity in severities:
                task_events.append(
                    {
                        "event_id": f"{task}__heldout__demo{demo_id:02d}__{severity['config_id']}",
                        "task": task,
                        "split": "heldout",
                        "demo_id": demo_id,
                        "config_id": severity["config_id"],
                        "severity_role": severity["role"],
                        "parameters": severity["parameters"],
                        "selection_uses_method_gain": False,
                    }
                )
        events.extend(task_events)
        task_schedules[task] = {
            "severities": severities,
            "evaluation_event_count": sum(
                event["split"] == "evaluation" for event in task_events
            ),
            "heldout_event_count": sum(
                event["split"] == "heldout" for event in task_events
            ),
            "event_count": len(task_events),
            "events_per_severity": {
                severity["config_id"]: sum(
                    event["config_id"] == severity["config_id"]
                    for event in task_events
                )
                for severity in severities
            },
        }
    event_count = len(events)
    primary = event_count * len(ACTOR_SEEDS) * len(BRANCH_PREFIXES)
    secondary = (
        event_count
        * len(ACTOR_SEEDS)
        * len(SECONDARY_OPERATORS)
        * len(SECONDARY_POSITIONS)
    )
    summary = {
        "schema_version": 1,
        "status": "PREFLIGHT_ONLY_NOT_LAUNCHED",
        "full_atlas_launched": False,
        "selection_uses_method_gain": False,
        "eligible_tasks": tasks,
        "eligible_task_count": len(tasks),
        "actor_seeds": list(ACTOR_SEEDS),
        "detection_prefix": DETECTION_PREFIX,
        "branch_prefixes": list(BRANCH_PREFIXES),
        "branch_prefix_count": len(BRANCH_PREFIXES),
        "secondary_operators": list(SECONDARY_OPERATORS),
        "secondary_positions": list(SECONDARY_POSITIONS),
        "tasks": task_schedules,
        "event_count": event_count,
        "primary_R_branch_run_count": primary,
        "secondary_operator_branch_run_count_max": secondary,
        "estimated_total_branch_run_count_max": primary + secondary,
        "methods_B0_B5_A_original_N_oracle_derived_from_same_R_branches": True,
        "count_excludes_source_prefix_reconstruction_overhead": True,
    }
    return summary, events


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Safe-Replanability Atlas preflight",
        "",
        "Status: **PREFLIGHT_ONLY_NOT_LAUNCHED**.",
        "",
        "This schedule is derived only from frozen calibration qualification and the clean actor gate. It does not inspect evaluation or held-out outcomes and does not execute a branch.",
        "",
        f"Eligible task intersection: `{summary['eligible_task_count']}` tasks.",
        f"Frozen events: `{summary['event_count']}`.",
        f"Primary R(k) branches: `{summary['primary_R_branch_run_count']}`.",
        f"Maximum secondary operator branches: `{summary['secondary_operator_branch_run_count_max']}`.",
        f"Maximum total branch runs: `{summary['estimated_total_branch_run_count_max']}` (source-prefix reconstruction overhead excluded).",
        "",
        "| Task | Primary severity | Secondary severity | Evaluation | Held-out |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for task, item in summary["tasks"].items():
        lines.append(
            f"| {task} | {item['severities'][0]['config_id']} | "
            f"{item['severities'][1]['config_id']} | "
            f"{item['evaluation_event_count']} | {item['heldout_event_count']} |"
        )
    lines.extend(
        [
            "",
            "The seven primary method readouts are selected from the same exhaustive R(k) branches and therefore do not multiply the primary branch count. Secondary controls are separately budgeted at d, k_best, and k_last_safe only.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-summary", type=Path, required=True)
    parser.add_argument("--frozen-perturbations", type=Path, required=True)
    parser.add_argument("--grid-summary", type=Path, required=True)
    parser.add_argument("--replay-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    actor = json.loads(args.actor_summary.read_text())
    perturbations = json.loads(args.frozen_perturbations.read_text())
    grid = json.loads(args.grid_summary.read_text())
    replay = json.loads(args.replay_summary.read_text())
    if actor["actor_gate"] != "PASS":
        raise SystemExit("BLOCKED_BY_ACTOR")
    if replay["status"] != "PASS":
        raise SystemExit("BLOCKED_BY_REPLAY")
    summary, events = build_schedule(actor, perturbations, grid)
    if summary["eligible_task_count"] < 2:
        raise SystemExit("INSUFFICIENT_TASK_INTERSECTION_FOR_PREREGISTERED_TRACK_A")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "preflight_summary.json", summary)
    atomic_write_jsonl(args.output_dir / "frozen_event_schedule.jsonl", events)
    atomic_write_text(args.output_dir / "preflight_report.md", render(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
