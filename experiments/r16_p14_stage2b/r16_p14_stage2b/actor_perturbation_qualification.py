from __future__ import annotations

import argparse
import json
import statistics
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import atomic_write_csv, atomic_write_json, atomic_write_text, load_jsonl, write_once_jsonl
from .runtime import ActorBundle, reconstruct_to_prefix, severity_id
from .settings import (
    ACTOR_SEEDS,
    ARTIFACT_ROOT,
    CREAM_GRID_M,
    EXPERIMENT_ROOT,
    PERTURBATION_CALIBRATION_IDS,
    PRIMARY_TASKS,
    STAGE2A_FROZEN_MAGNITUDES,
    STOVE_GRID_M,
)


def run_one(event: dict[str, Any], severity: float, bundle: ActorBundle, grid_stage: str) -> dict[str, Any]:
    horizon = int(event["effective_horizon"])
    context = reconstruct_to_prefix(
        event,
        severity_m=severity,
        prefix_k=horizon,
        bundle=bundle,
    )
    try:
        return {
            "record_type": "actor_conditioned_nominal_suffix",
            "task": event["task"],
            "event_id": event["event_id"],
            "actor_seed": event["actor_seed"],
            "init_state_id": event["init_state_id"],
            "split": event["split"],
            "severity_m": float(severity),
            "severity_id": severity_id(event["task"], severity),
            "grid_stage": grid_stage,
            "effective_horizon": horizon,
            "phase_valid": all(event["phase"]["eligibility"].values()),
            "immediate_cause_violation": bool(context.tracker.immediate_violation),
            "delayed_nominal_cause_violation": bool(
                context.tracker.violation and not context.tracker.immediate_violation
            ),
            "cause_violation": bool(context.tracker.violation),
            "cause_violation_type": context.tracker.violation_type,
            "first_violation_offset_after_d": context.tracker.first_violation_offset,
            "replay_valid": bool(context.anchor_replay["passed"]),
            "anchor_replay_checks": context.anchor_replay["checks"],
            "task_success_after_nominal_suffix": bool(context.env.check_success()),
            "source_is_actor_generated_chunk": bool(event["source_is_actor_generated_chunk"]),
            "proposed_method_outcome_read": False,
            "immediate_replan_outcome_read": False,
            "last_safe_or_k_best_read": False,
            "error": None,
        }
    finally:
        context.close()


def summarize(task: str, severity: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("error") is None]
    offsets = [
        int(row["first_violation_offset_after_d"])
        for row in valid
        if row["first_violation_offset_after_d"] is not None
        and not row["immediate_cause_violation"]
    ]
    unique_events = {row["event_id"] for row in valid}
    seed_counts = Counter(str(row["actor_seed"]) for row in valid)
    immediate = float(np.mean([row["immediate_cause_violation"] for row in valid])) if valid else 1.0
    delayed = float(np.mean([row["delayed_nominal_cause_violation"] for row in valid])) if valid else 0.0
    replay = float(np.mean([row["replay_valid"] for row in valid])) if valid else 0.0
    median_offset = float(statistics.median(offsets)) if offsets else None
    checks = {
        "valid_events_ge_20": len(unique_events) >= 20,
        "two_seeds_contribute_ge_5": sum(count >= 5 for count in seed_counts.values()) >= 2,
        "immediate_violation_le_0_10": immediate <= 0.10,
        "delayed_violation_in_0_30_0_80": 0.30 <= delayed <= 0.80,
        "median_first_violation_offset_ge_2": median_offset is not None and median_offset >= 2.0,
        "replay_rate_ge_0_99": replay >= 0.99,
    }
    return {
        "task": task,
        "severity_m": float(severity),
        "severity_id": severity_id(task, severity),
        "record_count": len(rows),
        "valid_event_count": len(unique_events),
        "actor_seed_event_counts": dict(seed_counts),
        "immediate_cause_violation_rate": immediate,
        "delayed_nominal_cause_violation_rate": delayed,
        "median_first_violation_offset_after_d": median_offset,
        "replay_rate": replay,
        "error_count": len(rows) - len(valid),
        "checks": checks,
        "qualifies": all(checks.values()),
    }


def blocked_rank(item: dict[str, Any], frozen: set[float]) -> tuple[Any, ...]:
    delayed_distance = abs(float(item["delayed_nominal_cause_violation_rate"]) - 0.55)
    return (
        -float(item["replay_rate"]),
        -int(item["immediate_cause_violation_rate"] <= 0.10),
        delayed_distance,
        -int(
            item["median_first_violation_offset_after_d"] is not None
            and item["median_first_violation_offset_after_d"] >= 2.0
        ),
        -int(float(item["severity_m"]) in frozen),
        float(item["severity_m"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = ARTIFACT_ROOT / "perturbation_qualification"
    raw_path = output / "raw.jsonl"
    if raw_path.is_file() and (output / "frozen_parameters.json").is_file():
        print("PERTURBATION_QUALIFICATION_ALREADY_COMPLETE")
        return
    events = [
        event
        for event in load_jsonl(ARTIFACT_ROOT / "actor_events/events.jsonl")
        if int(event["init_state_id"]) in PERTURBATION_CALIBRATION_IDS
    ]
    bundles = {seed: ActorBundle.load(seed, args.device) for seed in ACTOR_SEEDS}
    records: list[dict[str, Any]] = []
    summaries: dict[str, list[dict[str, Any]]] = {}
    for task in PRIMARY_TASKS:
        task_events = [event for event in events if event["task"] == task]
        initial = list(STAGE2A_FROZEN_MAGNITUDES[task])
        for severity in initial:
            for event in task_events:
                try:
                    records.append(run_one(event, severity, bundles[int(event["actor_seed"])], "stage2a_frozen"))
                except Exception as exc:
                    records.append(
                        {
                            "record_type": "actor_conditioned_nominal_suffix",
                            "task": task,
                            "event_id": event["event_id"],
                            "actor_seed": event["actor_seed"],
                            "init_state_id": event["init_state_id"],
                            "split": event["split"],
                            "severity_m": float(severity),
                            "severity_id": severity_id(task, severity),
                            "grid_stage": "stage2a_frozen",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                            "proposed_method_outcome_read": False,
                            "immediate_replan_outcome_read": False,
                            "last_safe_or_k_best_read": False,
                        }
                    )
        initial_summaries = [
            summarize(task, severity, [row for row in records if row["task"] == task and row["severity_m"] == severity])
            for severity in initial
        ]
        if sum(item["qualifies"] for item in initial_summaries) < 2:
            grid = CREAM_GRID_M if task == PRIMARY_TASKS[0] else STOVE_GRID_M
            for severity in grid:
                if severity in initial:
                    continue
                for event in task_events:
                    try:
                        records.append(run_one(event, severity, bundles[int(event["actor_seed"])], "fallback_grid"))
                    except Exception as exc:
                        records.append(
                            {
                                "record_type": "actor_conditioned_nominal_suffix",
                                "task": task,
                                "event_id": event["event_id"],
                                "actor_seed": event["actor_seed"],
                                "init_state_id": event["init_state_id"],
                                "split": event["split"],
                                "severity_m": float(severity),
                                "severity_id": severity_id(task, severity),
                                "grid_stage": "fallback_grid",
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(),
                                "proposed_method_outcome_read": False,
                                "immediate_replan_outcome_read": False,
                                "last_safe_or_k_best_read": False,
                            }
                        )
        tested = sorted({float(row["severity_m"]) for row in records if row["task"] == task})
        summaries[task] = [
            summarize(task, severity, [row for row in records if row["task"] == task and row["severity_m"] == severity])
            for severity in tested
        ]
    frozen: dict[str, Any] = {}
    all_tasks_pass = True
    for task in PRIMARY_TASKS:
        items = summaries[task]
        qualified = sorted(
            [item for item in items if item["qualifies"]],
            key=lambda item: (abs(item["delayed_nominal_cause_violation_rate"] - 0.55), item["severity_m"]),
        )
        forced = len(qualified) < 2
        selected = qualified[:2]
        if forced:
            all_tasks_pass = False
            selected = sorted(items, key=lambda item: blocked_rank(item, set(STAGE2A_FROZEN_MAGNITUDES[task])))[:2]
        frozen[task] = {
            "status": "qualified" if not forced else "blocked_diagnostic_continuation",
            "qualifying_severity_count": len(qualified),
            "forced_for_continuation": forced,
            "selection_uses_replan_or_method_outcomes": False,
            "severities": [
                {
                    "severity_m": item["severity_m"],
                    "severity_id": item["severity_id"],
                    "qualifies": item["qualifies"],
                    "metrics": item,
                }
                for item in selected
            ],
        }
    status = "PASS" if all_tasks_pass else "BLOCKED"
    summary = {
        "schema_version": 1,
        "status": status,
        "failure_label": None if all_tasks_pass else "BLOCKED_BY_ACTOR_CONDITIONED_PERTURBATION",
        "task_summaries": summaries,
        "both_tasks_have_two_qualifying_severities": all_tasks_pass,
        "selection_uses_proposed_method_gain": False,
        "evaluation_or_reserved_outcomes_read": False,
        "early_stop_applied": False,
        "downstream_continuation_required_by_user": True,
    }
    write_once_jsonl(raw_path, records)
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(output / "frozen_parameters.json", {"schema_version": 1, "status": status, "tasks": frozen})
    flat = []
    for task in PRIMARY_TASKS:
        flat.extend(summaries[task])
    fields = [
        "task",
        "severity_id",
        "severity_m",
        "record_count",
        "valid_event_count",
        "immediate_cause_violation_rate",
        "delayed_nominal_cause_violation_rate",
        "median_first_violation_offset_after_d",
        "replay_rate",
        "error_count",
        "qualifies",
    ]
    atomic_write_csv(output / "grid_summary.csv", fields, [{key: row.get(key) for key in fields} for row in flat])
    negative_lines = ["# Frozen negative perturbation results", ""]
    for task in PRIMARY_TASKS:
        for item in summaries[task]:
            if not item["qualifies"]:
                failed = [name for name, passed in item["checks"].items() if not passed]
                negative_lines.append(f"- `{task}` `{item['severity_id']}`: {', '.join(failed)}")
    negative_lines.append("")
    atomic_write_text(output / "negative_results.md", "\n".join(negative_lines))
    report_lines = [
        "# Phase C — actor-conditioned perturbation qualification",
        "",
        f"Status: **{status}**. Replan, k_best, last-safe and method-gain outcomes were not read.",
        "",
        "| task | severity | immediate | delayed | median offset | replay | qualifies |",
        "|---|---|---:|---:|---:|---:|:---:|",
    ]
    for item in flat:
        report_lines.append(
            f"| {item['task']} | {item['severity_id']} | {item['immediate_cause_violation_rate']:.3f} | "
            f"{item['delayed_nominal_cause_violation_rate']:.3f} | {item['median_first_violation_offset_after_d']} | "
            f"{item['replay_rate']:.3f} | {'yes' if item['qualifies'] else 'no'} |"
        )
    report_lines.extend(["", "Blocked tasks still receive two frozen diagnostic severities solely to complete the user-required downstream matrix.", ""])
    report = "\n".join(report_lines)
    atomic_write_text(output / "report.md", report)
    atomic_write_json(EXPERIMENT_ROOT / "perturbation_qualification/summary.json", summary)
    atomic_write_json(EXPERIMENT_ROOT / "perturbation_qualification/frozen_parameters.json", {"schema_version": 1, "status": status, "tasks": frozen})
    atomic_write_text(EXPERIMENT_ROOT / "perturbation_qualification/report.md", report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
