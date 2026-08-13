from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any, Iterable


TASKS = (
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack",
)
DEPLOYABLE_OPERATORS = (
    "trim_and_replan",
    "hold_and_replan",
    "bounded_rollback_and_replan",
    "cause_specific_local_repair",
)
RUN_ID = "r16p14-libero-stage1-pilot-v1"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _median(values: Iterable[float | int]) -> float | None:
    materialized = list(values)
    return statistics.median(materialized) if materialized else None


def branch_metrics(
    branch: dict[str, Any],
    *,
    branch_budget: int,
    retained_nominal_actions: int,
    discarded_nominal_actions: int,
) -> dict[str, Any]:
    cause_violation = bool(branch["violation"])
    task_failure = not bool(branch["task_recoverable"])
    steps = int(branch["steps"])
    timeout = bool(task_failure and not cause_violation and steps >= branch_budget)
    return {
        "cause_violation": cause_violation,
        "cause_violation_type": branch.get("violation_type"),
        "task_failure": task_failure,
        "timeout": timeout,
        "safe_success": bool(branch["safe_success"]),
        "new_non_nominal_actions_after_detection": steps,
        "retained_nominal_actions": retained_nominal_actions,
        "discarded_nominal_actions": discarded_nominal_actions,
        "policy_calls": int(branch["policy_calls"]),
        "branch_path_length": float(branch["extra_path_length"]),
        "corrected_total_path_length_after_detection": None,
        "completion_steps_after_detection": retained_nominal_actions + steps,
    }


def signed_delta(right: dict[str, Any], left: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "cause_violation",
        "task_failure",
        "timeout",
        "safe_success",
        "new_non_nominal_actions_after_detection",
        "discarded_nominal_actions",
        "policy_calls",
        "branch_path_length",
        "completion_steps_after_detection",
    )
    output: dict[str, Any] = {}
    for field in fields:
        right_value = right[field]
        left_value = left[field]
        output[field] = float(right_value) - float(left_value)
    output["new_non_nominal_action_savings"] = (
        int(left["new_non_nominal_actions_after_detection"])
        - int(right["new_non_nominal_actions_after_detection"])
    )
    return output


def reanalyze_record(record: dict[str, Any], branch_budget: int) -> dict[str, Any]:
    d = int(record["insertion_prefix"])
    chunk_length = int(record["chunk_length"])
    by_k = {int(item["prefix_k"]): item for item in record["prefixes"]}
    trigger = by_k[d]
    nominal = trigger["branches"]["nominal_continue"]
    explicit_cause = bool(nominal["violation"])
    old_mixed_unsafe = bool(nominal["violation"] or not nominal["task_recoverable"])

    safe_prefixes = [
        item
        for item in record["prefixes"]
        if int(item["prefix_k"]) >= d
        and any(
            bool(item["branches"][operator]["safe_success"])
            for operator in DEPLOYABLE_OPERATORS
        )
    ]
    chosen = max(safe_prefixes, key=lambda item: int(item["prefix_k"]), default=None)
    k_last_safe = int(chosen["prefix_k"]) if chosen is not None else None

    if explicit_cause:
        window = max(0, k_last_safe - d) if k_last_safe is not None else 0
        retention = window / (chunk_length - d)
    else:
        window = None
        retention = None

    old_retention = (
        k_last_safe / chunk_length
        if old_mixed_unsafe and k_last_safe is not None
        else (0.0 if old_mixed_unsafe else None)
    )

    m0 = branch_metrics(
        trigger["branches"]["trim_and_replan"],
        branch_budget=branch_budget,
        retained_nominal_actions=0,
        discarded_nominal_actions=chunk_length - d,
    )
    m1 = None
    m2 = None
    timing_gain = None
    operator_gain = None
    total_gain = None
    if chosen is not None:
        retained = k_last_safe - d
        discarded = chunk_length - k_last_safe
        m1 = branch_metrics(
            chosen["branches"]["trim_and_replan"],
            branch_budget=branch_budget,
            retained_nominal_actions=retained,
            discarded_nominal_actions=discarded,
        )
        m2 = branch_metrics(
            chosen["branches"]["cause_specific_local_repair"],
            branch_budget=branch_budget,
            retained_nominal_actions=retained,
            discarded_nominal_actions=discarded,
        )
        timing_gain = signed_delta(m1, m0)
        operator_gain = signed_delta(m2, m1)
        total_gain = signed_delta(m2, m0)

    nominal_metrics = branch_metrics(
        nominal,
        branch_budget=branch_budget,
        retained_nominal_actions=chunk_length - d,
        discarded_nominal_actions=0,
    )
    return {
        "schema_version": 1,
        "candidate_id": record["candidate_id"],
        "task": record["task"],
        "demo_id": int(record["demo_id"]),
        "anchor_step": int(record["anchor_step"]),
        "cause_type": record["cause_type"],
        "severity": record["severity"],
        "detection_prefix_d": d,
        "chunk_length": chunk_length,
        "old_stage1": {
            "mixed_unsafe_at_detection": old_mixed_unsafe,
            "reported_window": int(record["intervention_window_policy"]),
            "reconstructed_retention_k_over_h": old_retention,
        },
        "corrected_nominal_outcome": nominal_metrics,
        "primary_explicit_cause_cohort": explicit_cause,
        "k_last_safe": k_last_safe,
        "intervention_window_after_detection": window,
        "retention_after_detection": retention,
        "paired_methods": {
            "M0_immediate_full_replan_at_d": m0,
            "M1_continue_to_last_safe_then_full_replan": m1,
            "M2_continue_to_last_safe_then_cause_specific_repair": m2,
        },
        "signed_gains": {
            "timing_gain_M1_minus_M0": timing_gain,
            "operator_gain_M2_minus_M1": operator_gain,
            "total_gain_M2_minus_M0": total_gain,
        },
        "offline_limitations": [
            "corrected total path length is unavailable because Stage-1 did not persist nominal action values",
            "M0/M1/M2 are reconstructed from existing branch outcomes; no simulator was run",
        ],
    }


def _method_summary(rows: list[dict[str, Any]], method_key: str) -> dict[str, Any]:
    methods = [row["paired_methods"][method_key] for row in rows]
    assert all(method is not None for method in methods)
    return {
        "paired_count": len(methods),
        "cause_violation_count": sum(int(method["cause_violation"]) for method in methods),
        "cause_violation_rate": (
            sum(int(method["cause_violation"]) for method in methods) / len(methods)
            if methods
            else None
        ),
        "safe_success_count": sum(int(method["safe_success"]) for method in methods),
        "safe_success_rate": (
            sum(int(method["safe_success"]) for method in methods) / len(methods)
            if methods
            else None
        ),
        "median_new_non_nominal_actions": _median(
            method["new_non_nominal_actions_after_detection"] for method in methods
        ),
        "median_discarded_nominal_actions": _median(
            method["discarded_nominal_actions"] for method in methods
        ),
        "median_policy_calls": _median(method["policy_calls"] for method in methods),
        "median_completion_steps_after_detection": _median(
            method["completion_steps_after_detection"] for method in methods
        ),
    }


def summarize_task(
    task: str, rows: list[dict[str, Any]], old_summary: dict[str, Any]
) -> dict[str, Any]:
    cause_rows = [row for row in rows if row["primary_explicit_cause_cohort"]]
    paired = [row for row in cause_rows if row["k_last_safe"] is not None]
    nominal = [row["corrected_nominal_outcome"] for row in rows]
    result = {
        "candidate_count": len(rows),
        "old_stage1": {
            "mixed_unsafe_count": int(old_summary["no_intervention_violation_count"]),
            "median_intervention_window": old_summary["median_intervention_window"],
            "median_safe_prefix_retention_k_over_h": old_summary["median_safe_prefix_retention"],
            "oracle_mixed_unsafe_count": int(old_summary["oracle_violation_count"]),
            "relative_mixed_unsafe_reduction": old_summary["relative_violation_reduction"],
        },
        "corrected": {
            "explicit_nominal_cause_violation_count": len(cause_rows),
            "task_failure_count": sum(int(item["task_failure"]) for item in nominal),
            "timeout_count": sum(int(item["timeout"]) for item in nominal),
            "safe_success_count": sum(int(item["safe_success"]) for item in nominal),
            "task_failure_only_in_old_mixed_unsafe_count": (
                int(old_summary["no_intervention_violation_count"]) - len(cause_rows)
            ),
            "deployable_latest_safe_count": len(paired),
            "paired_comparison_count": len(paired),
            "median_intervention_window_after_detection": _median(
                row["intervention_window_after_detection"] for row in cause_rows
            ),
            "median_retention_after_detection": _median(
                row["retention_after_detection"] for row in cause_rows
            ),
            "M0": _method_summary(paired, "M0_immediate_full_replan_at_d"),
            "M1": _method_summary(
                paired, "M1_continue_to_last_safe_then_full_replan"
            ),
            "M2": _method_summary(
                paired, "M2_continue_to_last_safe_then_cause_specific_repair"
            ),
            "median_timing_new_action_savings": _median(
                row["signed_gains"]["timing_gain_M1_minus_M0"][
                    "new_non_nominal_action_savings"
                ]
                for row in paired
            ),
            "median_operator_new_action_savings": _median(
                row["signed_gains"]["operator_gain_M2_minus_M1"][
                    "new_non_nominal_action_savings"
                ]
                for row in paired
            ),
            "median_total_new_action_savings": _median(
                row["signed_gains"]["total_gain_M2_minus_M0"][
                    "new_non_nominal_action_savings"
                ]
                for row in paired
            ),
        },
    }
    corrected = result["corrected"]
    corrected["stage1_positive_signal_survives"] = bool(
        corrected["explicit_nominal_cause_violation_count"] >= 5
        and corrected["median_intervention_window_after_detection"] is not None
        and corrected["median_intervention_window_after_detection"] >= 2
        and corrected["median_retention_after_detection"] is not None
        and corrected["median_retention_after_detection"] >= 0.30
        and corrected["paired_comparison_count"] >= 5
        and corrected["M2"]["cause_violation_rate"]
        < corrected["M0"]["cause_violation_rate"]
    )
    return result


def _paired_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if not row["primary_explicit_cause_cohort"] or row["k_last_safe"] is None:
            continue
        methods = row["paired_methods"]
        gains = row["signed_gains"]
        flat: dict[str, Any] = {
            "task": row["task"],
            "candidate_id": row["candidate_id"],
            "demo_id": row["demo_id"],
            "d": row["detection_prefix_d"],
            "k_last_safe": row["k_last_safe"],
            "window": row["intervention_window_after_detection"],
            "retention_after_detection": row["retention_after_detection"],
        }
        for short, key in (
            ("M0", "M0_immediate_full_replan_at_d"),
            ("M1", "M1_continue_to_last_safe_then_full_replan"),
            ("M2", "M2_continue_to_last_safe_then_cause_specific_repair"),
        ):
            method = methods[key]
            for field in (
                "cause_violation",
                "task_failure",
                "timeout",
                "safe_success",
                "new_non_nominal_actions_after_detection",
                "discarded_nominal_actions",
                "policy_calls",
                "completion_steps_after_detection",
                "branch_path_length",
            ):
                flat[f"{short}_{field}"] = method[field]
        for short, key in (
            ("timing", "timing_gain_M1_minus_M0"),
            ("operator", "operator_gain_M2_minus_M1"),
            ("total", "total_gain_M2_minus_M0"),
        ):
            gain = gains[key]
            for field in (
                "cause_violation",
                "safe_success",
                "new_non_nominal_actions_after_detection",
                "new_non_nominal_action_savings",
                "policy_calls",
                "completion_steps_after_detection",
            ):
                flat[f"{short}_{field}"] = gain[field]
        output.append(flat)
    return output


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase A — corrected offline reanalysis",
        "",
        "No simulator or GPU was used. All 90 immutable Stage-1 oracle records were reread.",
        "",
        "## Old-to-new comparison",
        "",
        "| Task | Old mixed unsafe | Explicit cause violations | Failure-only inclusions | Old median window | Corrected median window | Old retention | Corrected post-detection retention | Paired n | Signal survives |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    labels = {
        TASKS[0]: "Drawer",
        TASKS[1]: "Bowl",
        TASKS[2]: "Wine",
    }
    for task in TASKS:
        item = summary["tasks"][task]
        old = item["old_stage1"]
        new = item["corrected"]
        old_retention = old["median_safe_prefix_retention_k_over_h"]
        new_retention = new["median_retention_after_detection"]
        lines.append(
            f"| {labels[task]} | {old['mixed_unsafe_count']} | "
            f"{new['explicit_nominal_cause_violation_count']} | "
            f"{new['task_failure_only_in_old_mixed_unsafe_count']} | "
            f"{old['median_intervention_window']} | "
            f"{new['median_intervention_window_after_detection']} | "
            f"{100 * old_retention:.1f}% | "
            f"{100 * new_retention:.1f}% | "
            f"{new['paired_comparison_count']} | "
            f"{'YES' if new['stage1_positive_signal_survives'] else 'NO'} |"
        )
    bowl = summary["tasks"][TASKS[1]]["corrected"]
    lines.extend(
        [
            "",
            "## Bowl conclusion",
            "",
            "The Stage-1 Bowl safety signal does **not** survive the corrected primary definition. "
            f"Only {bowl['explicit_nominal_cause_violation_count']} of 30 candidates had an explicit "
            "target-cause violation at no intervention, versus 17 old mixed-unsafe candidates. "
            f"The corrected median window is {bowl['median_intervention_window_after_detection']} and "
            f"median post-detection retention is {100 * bowl['median_retention_after_detection']:.1f}%. "
            f"Only {bowl['paired_comparison_count']} complete M0/M1/M2 pairs exist. On those pairs, "
            f"M0/M1/M2 cause-violation counts are {bowl['M0']['cause_violation_count']}/"
            f"{bowl['M1']['cause_violation_count']}/{bowl['M2']['cause_violation_count']}; immediate "
            "full replanning was not beaten on safety.",
            "",
            "M1 does reduce newly executed non-nominal actions on this tiny paired subset, but that "
            "observation is not a positive safety result and is too underpowered to carry the hypothesis.",
            "",
            "## Evidence limitations",
            "",
            "- Stage-1 did not persist nominal action values, so corrected total path length cannot be reconstructed offline.",
            "- Paired metrics are defined only when an explicit-cause candidate has a deployable successful intervention.",
            "- These are deterministic re-labelings of existing branches, not newly executed counterfactuals.",
            "- Phase B must independently repair and test replay before any expert calibration.",
            "",
        ]
    )
    return "\n".join(lines)


def run(stage1_artifacts: Path, output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    old_summaries: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        task_root = stage1_artifacts / "shards" / task / RUN_ID
        raw_path = task_root / "oracle_audit" / task / "oracle_prefix_labels.jsonl"
        old_summary_path = task_root / "oracle_audit" / task / "summary.json"
        old_summary = _load_json(old_summary_path)
        old_summaries[task] = old_summary
        branch_budget = int(old_summary["branch_budget"])
        task_rows = [_load for _load in _load_jsonl(raw_path)]
        assert len(task_rows) == 30, f"expected 30 records for {task}"
        rows.extend(reanalyze_record(record, branch_budget) for record in task_rows)
    assert len(rows) == 90

    task_summaries = {
        task: summarize_task(
            task,
            [row for row in rows if row["task"] == task],
            old_summaries[task],
        )
        for task in TASKS
    }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "phase": "A_offline_reanalysis",
        "simulator_used": False,
        "gpu_used": False,
        "record_count": len(rows),
        "primary_cohort": "explicit_nominal_cause_violation",
        "tasks": task_summaries,
        "bowl_positive_signal_survives": task_summaries[TASKS[1]]["corrected"][
            "stage1_positive_signal_survives"
        ],
    }

    jsonl = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _atomic_text(output_dir / "reanalysis.jsonl", jsonl)
    _atomic_text(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _atomic_text(output_dir / "report.md", render_report(summary))

    paired = _paired_csv_rows(rows)
    if not paired:
        raise AssertionError("expected at least one corrected pair")
    csv_path = output_dir / "paired_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(paired[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(paired)
    os.replace(temporary, csv_path)

    comparison_path = output_dir / "old_to_new_metrics.csv"
    comparison_rows = []
    for task in TASKS:
        item = task_summaries[task]
        old = item["old_stage1"]
        new = item["corrected"]
        comparison_rows.append(
            {
                "task": task,
                "old_mixed_unsafe_count": old["mixed_unsafe_count"],
                "corrected_explicit_cause_violation_count": new[
                    "explicit_nominal_cause_violation_count"
                ],
                "failure_only_old_inclusions": new[
                    "task_failure_only_in_old_mixed_unsafe_count"
                ],
                "old_median_window": old["median_intervention_window"],
                "corrected_median_window_after_detection": new[
                    "median_intervention_window_after_detection"
                ],
                "old_median_retention_k_over_h": old[
                    "median_safe_prefix_retention_k_over_h"
                ],
                "corrected_median_retention_after_detection": new[
                    "median_retention_after_detection"
                ],
                "corrected_paired_count": new["paired_comparison_count"],
                "stage1_positive_signal_survives": new[
                    "stage1_positive_signal_survives"
                ],
            }
        )
    temporary = comparison_path.with_name(
        f".{comparison_path.name}.tmp.{os.getpid()}"
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(comparison_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(comparison_rows)
    os.replace(temporary, comparison_path)
    return summary


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage1-artifacts",
        type=Path,
        default=repo_root / "artifacts/formal_pilot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts/stage1b/offline_reanalysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.stage1_artifacts.resolve(), args.output_dir.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
