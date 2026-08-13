from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any

from r16_p14_stage1.io_utils import atomic_write_json


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def median(values: list[float | int]) -> float | None:
    return float(statistics.median(values)) if values else None


def selected_records(
    records: list[dict[str, Any]], frozen: dict[str, Any]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for task_name, task in frozen["tasks"].items():
        config = task["selected_config_id"]
        if config is None:
            continue
        selected.extend(
            record
            for record in records
            if record.get("record_type") == "calibration_prefix_audit"
            and record.get("task") == task_name
            and record.get("config_id") == config
        )
    return selected


def paired_row(record: dict[str, Any]) -> dict[str, Any]:
    m0 = record["M0_immediate_full_replan"]
    m1 = record["M1_continue_then_full_replan"]
    m2 = record["M2_continue_then_local_repair"]
    return {
        "task": record["task"],
        "config_id": record["config_id"],
        "demo_id": record["demo_id"],
        "candidate_id": record["candidate_id"],
        "detection_prefix_d": record["insertion_prefix"],
        "k_last_safe": record["k_last_safe"],
        "intervention_window": record["intervention_window"],
        "post_detection_retention": record["post_detection_retention"],
        "M0_cause_violation": m0["cause_violation"],
        "M0_safe_success": m0["safe_success"],
        "M0_new_non_nominal_actions": m0["new_non_nominal_actions"],
        "M1_cause_violation": m1["cause_violation"],
        "M1_safe_success": m1["safe_success"],
        "M1_new_non_nominal_actions": m1["new_non_nominal_actions"],
        "M2_cause_violation": m2["cause_violation"],
        "M2_safe_success": m2["safe_success"],
        "M2_new_non_nominal_actions": m2["new_non_nominal_actions"],
        "timing_gain_cause_violation_M1_minus_M0": int(m1["cause_violation"])
        - int(m0["cause_violation"]),
        "timing_gain_safe_success_M1_minus_M0": int(m1["safe_success"])
        - int(m0["safe_success"]),
        "timing_gain_new_actions_M1_minus_M0": m1["new_non_nominal_actions"]
        - m0["new_non_nominal_actions"],
        "operator_gain_cause_violation_M2_minus_M1": int(m2["cause_violation"])
        - int(m1["cause_violation"]),
        "operator_gain_safe_success_M2_minus_M1": int(m2["safe_success"])
        - int(m1["safe_success"]),
        "operator_gain_new_actions_M2_minus_M1": m2["new_non_nominal_actions"]
        - m1["new_non_nominal_actions"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = args.source_dir.resolve()
    frozen = json.loads((source_dir / "frozen_parameters.json").read_text())
    grid = json.loads((source_dir / "grid_summary.json").read_text())
    records = [
        json.loads(line)
        for line in (source_dir / "calibration_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    selected = selected_records(records, frozen)
    rows = [paired_row(record) for record in selected]
    write_csv(source_dir / "selected_config_paired_metrics.csv", rows)

    selected_by_task: dict[str, list[dict[str, Any]]] = {}
    for record in selected:
        selected_by_task.setdefault(record["task"], []).append(record)
    task_metrics: dict[str, Any] = {}
    for task_name, task in frozen["tasks"].items():
        task_records = selected_by_task.get(task_name, [])
        if not task_records:
            task_metrics[task_name] = {
                "calibration_status": task["status"],
                "selected_config_id": None,
                "paired_candidate_count": 0,
                "median_intervention_window": None,
                "median_post_detection_retention": None,
            }
            continue
        methods = {
            "M0": [r["M0_immediate_full_replan"] for r in task_records],
            "M1": [r["M1_continue_then_full_replan"] for r in task_records],
            "M2": [r["M2_continue_then_local_repair"] for r in task_records],
        }
        task_metrics[task_name] = {
            "calibration_status": task["status"],
            "selected_config_id": task["selected_config_id"],
            "paired_candidate_count": len(task_records),
            "median_intervention_window": median(
                [r["intervention_window"] for r in task_records]
            ),
            "median_post_detection_retention": median(
                [r["post_detection_retention"] for r in task_records]
            ),
            "methods": {
                name: {
                    "cause_violation_count": sum(
                        bool(item["cause_violation"]) for item in items
                    ),
                    "safe_success_count": sum(
                        bool(item["safe_success"]) for item in items
                    ),
                    "median_new_non_nominal_actions": median(
                        [item["new_non_nominal_actions"] for item in items]
                    ),
                }
                for name, items in methods.items()
            },
            "oracle_best_safe_success_count": sum(
                any(
                    r[name]["safe_success"]
                    for name in (
                        "M0_immediate_full_replan",
                        "M1_continue_then_full_replan",
                        "M2_continue_then_local_repair",
                    )
                )
                for r in task_records
            ),
        }

    replay_pass = sum(
        record["replay_pass_count"]
        for record in records
        if record.get("record_type") == "calibration_prefix_audit"
    )
    replay_total = sum(
        record["replay_branch_point_count"]
        for record in records
        if record.get("record_type") == "calibration_prefix_audit"
    )
    tasks_window = sum(
        bool(value["median_intervention_window"] is not None)
        and value["median_intervention_window"] >= 2
        for value in task_metrics.values()
    )
    tasks_retention = sum(
        bool(value["median_post_detection_retention"] is not None)
        and value["median_post_detection_retention"] >= 0.30
        for value in task_metrics.values()
    )
    gate_checks = {
        "calibration_replay_ge_0_99": bool(
            replay_total and replay_pass / replay_total >= 0.99
        ),
        "minimum_two_qualified_tasks": len(selected_by_task) >= 2,
        "minimum_two_tasks_median_window_ge_2": tasks_window >= 2,
        "minimum_two_tasks_median_retention_ge_0_30": tasks_retention >= 2,
        "disjoint_evaluation_30_candidates_per_task": False,
        "relative_cause_violation_reduction_ge_0_30": None,
        "nonempty_timing_comparison": bool(rows),
        "nonempty_operator_comparison": bool(rows),
    }
    result = {
        "schema_version": 1,
        "phase": "C_expert_action_chunk_feasibility",
        "status": "stopped_after_calibration_gate_failure",
        "decision": "KILL_CORE_HYPOTHESIS",
        "policy_used": False,
        "learned_risk_used": False,
        "calibration_demo_ids": frozen["calibration_demo_ids"],
        "evaluation_demo_ids_inspected": [],
        "heldout_demo_ids_inspected": [],
        "qualified_task_count": len(selected_by_task),
        "qualified_tasks": sorted(selected_by_task),
        "calibration_prefix_replay": {
            "pass_count": replay_pass,
            "branch_point_count": replay_total,
            "pass_rate": replay_pass / replay_total if replay_total else None,
        },
        "task_metrics": task_metrics,
        "gate_checks": gate_checks,
        "gate_passed": False,
        "failure_is_monotonic_at_calibration": True,
        "stop_reason": (
            "Only one task has any preregistration-qualified perturbation. "
            "The expert gate requires at least two tasks, so no disjoint evaluation "
            "result can rescue the gate without violating the frozen calibration rule."
        ),
        "phase_d_executed": False,
        "phase_e_executed": False,
        "grid_config_count": len(grid),
        "selected_pair_count": len(rows),
    }
    atomic_write_json(source_dir / "expert_gate_summary.json", result)
    return result


def render_report(result: dict[str, Any]) -> str:
    selected = result["task_metrics"].get("put_the_cream_cheese_in_the_bowl", {})
    methods = selected.get("methods", {})
    lines = [
        "# Phase C — expert action-chunk feasibility gate",
        "",
        "## Decision",
        "",
        "**KILL_CORE_HYPOTHESIS**",
        "",
        "The gate became impossible at calibration: only one of three tasks produced any perturbation satisfying all frozen constraints. Evaluation demos 10–39 and held-out demos 40–49 were not inspected. Continuing would require retuning failed tasks on non-calibration data, which is forbidden.",
        "",
        "## Task result",
        "",
        "| Task | Calibration result | Selected configuration | Median window | Median post-detection retention |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for task_name, metrics in result["task_metrics"].items():
        window = metrics["median_intervention_window"]
        retention = metrics["median_post_detection_retention"]
        lines.append(
            f"| {task_name} | {metrics['calibration_status']} | "
            f"{metrics['selected_config_id'] or 'none'} | "
            f"{window if window is not None else 'n/a'} | "
            f"{f'{100 * retention:.1f}%' if retention is not None else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "Bowl never satisfied all phase/immediate-contact/rate constraints. Drawer had one rate-qualified near miss (`offset+00_clear035mm`) but its actual median recoverable window was 0. The single bounded replacement selected `lead08_shift040mm`: 0% immediate violations, 60% delayed violations, median window 7, and median retention 50% on calibration demos.",
            "",
            "## Selected replacement M0/M1/M2 (calibration only)",
            "",
            "| Method | Cause violations | Safe successes | Median new non-nominal actions |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name in ("M0", "M1", "M2"):
        item = methods.get(name, {})
        lines.append(
            f"| {name} | {item.get('cause_violation_count', 'n/a')}/6 | "
            f"{item.get('safe_success_count', 'n/a')}/6 | "
            f"{item.get('median_new_non_nominal_actions', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "The timing comparison is nonempty but has no aggregate safety advantage: M0 and M1 each have 1/6 cause violations and 5/6 safe successes. M2 executes far fewer new actions but is less safe (3/6 cause violations; 2/6 safe successes), so there is no operator win.",
            "",
            f"Independent calibration prefix replay passed {result['calibration_prefix_replay']['pass_count']}/{result['calibration_prefix_replay']['branch_point_count']} branch points ({100 * result['calibration_prefix_replay']['pass_rate']:.1f}%).",
            "",
            "No BC policy, learned risk model, RGB model, world model, policy training, Phase D, or Phase E was executed.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = aggregate(args)
    atomic_write_text(args.source_dir.resolve() / "report.md", render_report(result))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
