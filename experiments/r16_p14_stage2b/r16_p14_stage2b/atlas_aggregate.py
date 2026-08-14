from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .baselines import k_best, k_last_safe, method_positions, strongest_baseline
from .bootstrap import grouped_readout, paired_event_bootstrap
from .io_utils import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    load_jsonl,
    sha256_json,
    write_once_jsonl,
)
from .settings import (
    ACTOR_SEEDS,
    ARTIFACT_ROOT,
    ATLAS_EVALUATION_IDS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DETECTION_PREFIX,
    EXPERIMENT_ROOT,
    PERTURBATION_CALIBRATION_IDS,
    PRIMARY_TASKS,
)


METHOD_NAMES = ("N0", "B0", "B1", "B2", "B3", "B4", "B5", "B6", "A_original", "N_oracle")


def clean_disagreement_threshold() -> dict[str, Any]:
    records = load_jsonl(ARTIFACT_ROOT / "chunk_executability/prefix_records.jsonl")
    values = [
        float(row["normalized_disagreement"])
        for row in records
        if row.get("record_type") == "intermediate_disagreement"
        and int(row["init_state_id"]) in PERTURBATION_CALIBRATION_IDS
        and row.get("normalized_disagreement") is not None
    ]
    if not values:
        raise ValueError("no clean calibration disagreement records")
    return {
        "threshold": float(np.quantile(np.asarray(values, dtype=np.float64), 0.95)),
        "quantile": 0.95,
        "record_count": len(values),
        "init_ids": list(PERTURBATION_CALIBRATION_IDS),
        "source": "Phase-A clean, unperturbed intermediate actor calls",
        "evaluation_outcomes_read": False,
    }


def load_formal_branches() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((ARTIFACT_ROOT / "atlas_pilot/events").glob("*.branches.jsonl")):
        records.extend(load_jsonl(path))
    return records


def valid_event_groups(branches: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in branches:
        grouped[row["event_id"]].append(row)
    valid: dict[str, list[dict[str, Any]]] = {}
    invalid: dict[str, Any] = {}
    for event_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["prefix_k"]))
        horizon = int(rows[0]["effective_horizon"])
        expected = list(range(DETECTION_PREFIX, horizon + 1))
        observed = [int(row["prefix_k"]) for row in rows]
        reasons = []
        if observed != expected:
            reasons.append(f"prefixes={observed},expected={expected}")
        if any(row.get("error") is not None for row in rows):
            reasons.append("branch_error")
        if len({int(row.get("post_detection_budget", -1)) for row in rows}) != 1:
            reasons.append("unequal_event_budget")
        if any(not row.get("source_is_actor_generated_chunk") for row in rows):
            reasons.append("non_actor_nominal")
        if any(row.get("source_is_demonstration_chunk") for row in rows):
            reasons.append("demonstration_nominal")
        if reasons:
            invalid[event_id] = reasons
        else:
            valid[event_id] = rows
    return valid, invalid


def derive_methods(
    groups: dict[str, list[dict[str, Any]]], threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    method_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for event_id, branches in sorted(groups.items()):
        first = branches[0]
        d = int(first["detection_prefix"])
        horizon = int(first["effective_horizon"])
        chunk = np.asarray(
            next(
                event["original_chunk"]
                for event in load_jsonl(ARTIFACT_ROOT / "actor_events/events.jsonl")
                if event["event_id"] == event_id
            ),
            dtype=np.float32,
        )
        positions = method_positions(
            branches,
            chunk=chunk,
            d=d,
            horizon=horizon,
            disagreement_threshold=threshold,
        )
        index = {int(branch["prefix_k"]): branch for branch in branches}
        event_record = {
            "event_id": event_id,
            "task": first["task"],
            "actor_seed": int(first["actor_seed"]),
            "init_state_id": int(first["init_state_id"]),
            "severity_m": float(first["severity_m"]),
            "severity_id": first["severity_id"],
            "post_detection_budget": int(first["post_detection_budget"]),
            "effective_horizon": horizon,
            "k_last_safe": k_last_safe(branches),
            "k_best": k_best(branches),
            "method_positions": positions,
            "same_branch_set_hash": sha256_json(
                [
                    {
                        "prefix_k": row["prefix_k"],
                        "safe_success": row["safe_success"],
                        "cause_violation": row["cause_violation"],
                        "new_non_nominal_actions": row["new_non_nominal_actions"],
                        "policy_calls": row["policy_calls"],
                    }
                    for row in branches
                ]
            ),
        }
        event_rows.append(event_record)
        for method in METHOD_NAMES:
            position = positions[method]
            if position is None:
                method_rows.append(
                    {
                        **{key: value for key, value in event_record.items() if key != "method_positions"},
                        "method": method,
                        "available": False,
                        "prefix_k": None,
                    }
                )
                continue
            branch = index[position]
            method_rows.append(
                {
                    **{key: value for key, value in event_record.items() if key != "method_positions"},
                    "method": method,
                    "available": True,
                    "prefix_k": position,
                    "safe_success": int(branch["safe_success"]),
                    "task_success": int(branch["task_success"]),
                    "cause_violation": int(branch["cause_violation"]),
                    "new_non_nominal_actions": int(branch["new_non_nominal_actions"]),
                    "policy_calls": int(branch["policy_calls"]),
                    "completion_steps": int(branch["completion_steps"]),
                    "path_length": float(branch["path_length"]),
                    "retained_nominal_actions": int(branch["old_nominal_actions_retained_after_detection"]),
                    "timeout": int(branch["timeout"]),
                    "branch_hash": sha256_json(branch),
                }
            )
    return method_rows, event_rows


def paired_rows(method_rows: list[dict[str, Any]], left: str, right: str) -> list[dict[str, Any]]:
    by_event: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in method_rows:
        if row["method"] in {left, right} and row.get("available"):
            by_event[row["event_id"]][row["method"]] = row
    result = []
    for event_id, values in by_event.items():
        if left not in values or right not in values:
            continue
        lrow, rrow = values[left], values[right]
        result.append(
            {
                "event_id": event_id,
                "task": lrow["task"],
                "actor_seed": lrow["actor_seed"],
                "severity_id": lrow["severity_id"],
                "severity_m": lrow["severity_m"],
                "left": left,
                "right": right,
                "safe_success_difference": lrow["safe_success"] - rrow["safe_success"],
                "cause_violation_difference": lrow["cause_violation"] - rrow["cause_violation"],
                "new_actions_difference": lrow["new_non_nominal_actions"] - rrow["new_non_nominal_actions"],
                "policy_calls_difference": lrow["policy_calls"] - rrow["policy_calls"],
                "completion_steps_difference": lrow["completion_steps"] - rrow["completion_steps"],
                "path_length_difference": lrow["path_length"] - rrow["path_length"],
                "left_row": lrow,
                "right_row": rrow,
            }
        )
    return result


def stratified_bootstraps(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    metrics: dict[str, Callable[[dict[str, Any]], float]] = {
        "safe_success_difference": lambda row: float(row["safe_success_difference"]),
        "cause_violation_difference": lambda row: float(row["cause_violation_difference"]),
        "new_actions_difference": lambda row: float(row["new_actions_difference"]),
        "policy_calls_difference": lambda row: float(row["policy_calls_difference"]),
    }
    output: dict[str, Any] = {"comparison": prefix, "resample_unit": "event"}
    output["task_macro"] = {
        name: paired_event_bootstrap(
            rows, function, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED + offset
        )
        for offset, (name, function) in enumerate(metrics.items())
    }
    for field in ("task", "actor_seed", "severity_id"):
        output[f"by_{field}"] = {}
        for value in sorted({str(row[field]) for row in rows}):
            subset = [row for row in rows if str(row[field]) == value]
            output[f"by_{field}"][value] = {
                name: paired_event_bootstrap(
                    subset,
                    function,
                    resamples=BOOTSTRAP_RESAMPLES,
                    seed=BOOTSTRAP_SEED + 101 * (index + 1) + offset,
                )
                for offset, (name, function) in enumerate(metrics.items())
                for index in [list(sorted({str(row[field]) for row in rows})).index(value)]
            }
    return output


def minimum_data_check(event_rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_counts = Counter(row["task"] for row in event_rows)
    task_severity = Counter((row["task"], row["severity_id"]) for row in event_rows)
    task_seed = Counter((row["task"], int(row["actor_seed"])) for row in event_rows)
    checks = {
        "events_ge_20_each_task": all(task_counts[task] >= 20 for task in PRIMARY_TASKS),
        "events_ge_8_each_task_severity": all(
            task_severity[(task, severity)] >= 8
            for task in PRIMARY_TASKS
            for severity in {row["severity_id"] for row in event_rows if row["task"] == task}
        ) and all(len({row["severity_id"] for row in event_rows if row["task"] == task}) == 2 for task in PRIMARY_TASKS),
        "events_ge_5_each_task_seed": all(
            task_seed[(task, seed)] >= 5 for task in PRIMARY_TASKS for seed in ACTOR_SEEDS
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "task_counts": dict(task_counts),
        "task_severity_counts": {f"{task}/{severity}": count for (task, severity), count in task_severity.items()},
        "task_seed_counts": {f"{task}/seed{seed}": count for (task, seed), count in task_seed.items()},
    }


def track_a_decision(event_rows: list[dict[str, Any]], pairs: list[dict[str, Any]], boot: dict[str, Any], upstream_pass: bool, data_pass: bool) -> dict[str, Any]:
    medians = {
        task: float(statistics.median([
            row["k_last_safe"] - DETECTION_PREFIX
            for row in event_rows
            if row["task"] == task and row["k_last_safe"] is not None
        ])) if any(row["task"] == task and row["k_last_safe"] is not None for row in event_rows) else None
        for task in PRIMARY_TASKS
    }
    b0_actions = float(np.mean([row["right_row"]["new_non_nominal_actions"] for row in pairs])) if pairs else 0.0
    a_actions = float(np.mean([row["left_row"]["new_non_nominal_actions"] for row in pairs])) if pairs else 0.0
    reduction = (b0_actions - a_actions) / b0_actions if b0_actions > 0 else 0.0
    seed_direction = {
        str(seed): float(np.mean([row["new_actions_difference"] for row in pairs if row["actor_seed"] == seed])) < 0
        for seed in ACTOR_SEEDS
    }
    severity_direction = {
        severity: float(np.mean([row["new_actions_difference"] for row in pairs if row["severity_id"] == severity])) < 0
        for severity in sorted({row["severity_id"] for row in pairs})
    }
    task_direction = {
        task: float(np.mean([row["new_actions_difference"] for row in pairs if row["task"] == task])) < 0
        for task in PRIMARY_TASKS
    }
    lower = boot["task_macro"]["safe_success_difference"]["ci95"][0]
    checks = {
        "both_tasks_median_delay_ge_2": all(value is not None and value >= 2 for value in medians.values()),
        "safe_success_ci_lower_ge_minus_0_05": lower is not None and lower >= -0.05,
        "new_non_nominal_actions_reduction_ge_15pct": reduction >= 0.15,
        "direction_consistent_two_of_three_seeds": sum(seed_direction.values()) >= 2,
        "same_direction_both_severities": len(severity_direction) >= 2 and all(severity_direction.values()),
        "not_one_task_or_seed": all(task_direction.values()) and sum(seed_direction.values()) >= 2,
    }
    signal = all(checks.values())
    status = "PILOT_SIGNAL" if signal else "NO_SIGNAL"
    if not upstream_pass or not data_pass:
        status = "INCONCLUSIVE"
    return {
        "status": status,
        "allowed_positive_label": "CONDITIONAL_LATE_PLACEMENT_PILOT_SIGNAL" if status == "PILOT_SIGNAL" else None,
        "median_k_last_safe_minus_d": medians,
        "mean_B0_new_actions": b0_actions,
        "mean_A_original_new_actions": a_actions,
        "new_non_nominal_actions_reduction": reduction,
        "seed_direction": seed_direction,
        "severity_direction": severity_direction,
        "task_direction": task_direction,
        "checks": checks,
        "upstream_pass": upstream_pass,
        "minimum_data_pass": data_pass,
    }


def track_b_decision(event_rows: list[dict[str, Any]], pairs: list[dict[str, Any]], boot: dict[str, Any], strongest: str | None, upstream_pass: bool, data_pass: bool) -> dict[str, Any]:
    n = len(event_rows)
    best_not_d = float(np.mean([row["k_best"] != DETECTION_PREFIX for row in event_rows])) if n else 0.0
    best_not_last = float(np.mean([row["k_best"] != row["k_last_safe"] for row in event_rows])) if n else 0.0
    intermediate = float(np.mean([
        row["k_last_safe"] is not None
        and DETECTION_PREFIX < row["k_best"] < row["k_last_safe"]
        for row in event_rows
    ])) if n else 0.0
    safe_diff = float(np.mean([row["safe_success_difference"] for row in pairs])) if pairs else 0.0
    baseline_cause = float(np.mean([row["right_row"]["cause_violation"] for row in pairs])) if pairs else 0.0
    oracle_cause = float(np.mean([row["left_row"]["cause_violation"] for row in pairs])) if pairs else 0.0
    cause_reduction = (baseline_cause - oracle_cause) / baseline_cause if baseline_cause > 0 else 0.0
    task_effect = {
        task: (
            float(np.mean([row["safe_success_difference"] for row in pairs if row["task"] == task])) > 0
            or float(np.mean([row["cause_violation_difference"] for row in pairs if row["task"] == task])) < 0
        ) if any(row["task"] == task for row in pairs) else False
        for task in PRIMARY_TASKS
    }
    budget_equal = all(
        row["left_row"]["post_detection_budget"] == row["right_row"]["post_detection_budget"]
        for row in pairs
    )
    checks = {
        "safe_success_plus_10pp_or_cause_reduction_ge_25pct": safe_diff >= 0.10 or cause_reduction >= 0.25,
        "k_best_not_d_ge_20pct": best_not_d >= 0.20,
        "k_best_not_k_last_safe_ge_20pct": best_not_last >= 0.20,
        "interior_k_best_ge_15pct": intermediate >= 0.15,
        "effect_in_both_tasks": all(task_effect.values()),
        "identical_post_detection_budget": budget_equal,
        "policy_call_difference_reported": bool(pairs),
    }
    signal = all(checks.values())
    status = "PILOT_SIGNAL" if signal else "NO_ORACLE_GAP"
    if not upstream_pass or not data_pass or strongest is None:
        status = "INCONCLUSIVE"
    return {
        "status": status,
        "allowed_positive_label": "REPLANABILITY_ORACLE_PILOT_SIGNAL" if status == "PILOT_SIGNAL" else None,
        "strongest_baseline": strongest,
        "safe_success_difference": safe_diff,
        "baseline_cause_violation_rate": baseline_cause,
        "oracle_cause_violation_rate": oracle_cause,
        "cause_violation_relative_reduction": cause_reduction,
        "k_best_not_d_rate": best_not_d,
        "k_best_not_k_last_safe_rate": best_not_last,
        "interior_k_best_rate": intermediate,
        "task_effect": task_effect,
        "mean_policy_call_difference": float(np.mean([row["policy_calls_difference"] for row in pairs])) if pairs else None,
        "checks": checks,
        "upstream_pass": upstream_pass,
        "minimum_data_pass": data_pass,
    }


def main() -> None:
    output = ARTIFACT_ROOT / "atlas_pilot"
    branches = load_formal_branches()
    groups, invalid = valid_event_groups(branches)
    threshold = clean_disagreement_threshold()
    method_rows, event_rows = derive_methods(groups, float(threshold["threshold"]))
    strongest = strongest_baseline(method_rows)
    pairs_a = paired_rows(method_rows, "A_original", "B0")
    pairs_b = paired_rows(method_rows, "N_oracle", strongest) if strongest else []
    boot_a = stratified_bootstraps(pairs_a, "A_original_minus_B0")
    boot_b = stratified_bootstraps(pairs_b, f"N_oracle_minus_{strongest}")
    minimum = minimum_data_check(event_rows)
    phase_a = json.loads((ARTIFACT_ROOT / "chunk_executability/summary.json").read_text())
    phase_c = json.loads((ARTIFACT_ROOT / "perturbation_qualification/summary.json").read_text())
    phase_d = json.loads((ARTIFACT_ROOT / "replay/summary.json").read_text())
    upstream_pass = all(item["status"] == "PASS" for item in (phase_a, phase_c, phase_d))
    track_a = track_a_decision(event_rows, pairs_a, boot_a, upstream_pass, minimum["passed"])
    track_b = track_b_decision(event_rows, pairs_b, boot_b, strongest, upstream_pass, minimum["passed"])
    metrics = (
        "safe_success",
        "cause_violation",
        "retained_nominal_actions",
        "new_non_nominal_actions",
        "policy_calls",
        "completion_steps",
        "path_length",
    )
    available = [row for row in method_rows if row.get("available")]
    readouts = {
        "per_task_severity_actor_seed_method": grouped_readout(
            available, ("task", "severity_id", "actor_seed", "method"), metrics
        ),
        "per_task_method": grouped_readout(available, ("task", "method"), metrics),
        "macro_method": grouped_readout(available, ("method",), metrics),
    }
    summary = {
        "schema_version": 1,
        "formal_branch_count": len(branches),
        "valid_event_count": len(event_rows),
        "invalid_events": invalid,
        "maximum_event_contract": 60,
        "evaluation_init_ids_only": all(row["init_state_id"] in ATLAS_EVALUATION_IDS for row in event_rows),
        "clean_disagreement_calibration": threshold,
        "minimum_valid_data": minimum,
        "strongest_baseline": strongest,
        "track_a": track_a,
        "track_b": track_b,
        "all_upstream_gates_pass": upstream_pass,
        "gate_override_descriptive_only": not upstream_pass,
    }
    write_once_jsonl(output / "raw_branches.jsonl", branches)
    write_once_jsonl(output / "method_readouts.jsonl", method_rows)
    write_once_jsonl(output / "event_metrics.jsonl", event_rows)
    atomic_write_json(output / "readouts.json", readouts)
    atomic_write_json(output / "paired_metrics.json", {"track_a": pairs_a, "track_b": pairs_b})
    atomic_write_json(output / "bootstrap_intervals.json", {"track_a": boot_a, "track_b": boot_b})
    atomic_write_json(output / "summary.json", summary)
    csv_rows = readouts["macro_method"]
    if csv_rows:
        atomic_write_csv(output / "baseline_summary.csv", list(csv_rows[0]), csv_rows)
    lines = [
        "# Phase E — bounded safe-replanability Atlas pilot",
        "",
        f"Valid events: **{len(event_rows)}**; formal branches: **{len(branches)}**; strongest frozen baseline: **{strongest}**.",
        f"Upstream A/C/D gates all pass: **{upstream_pass}**; minimum-data contract: **{minimum['passed']}**.",
        "",
        f"- Track A: **{track_a['status']}**; new-action reduction `{track_a['new_non_nominal_actions_reduction']:.3f}`.",
        f"- Track B: **{track_b['status']}**; safe-success difference `{track_b['safe_success_difference']:.3f}`, cause reduction `{track_b['cause_violation_relative_reduction']:.3f}`.",
        f"- B6 clean threshold: `{threshold['threshold']:.6f}` from `{threshold['record_count']}` event-independent calibration records (not evaluation outcomes).",
        "",
        "Prefix branches are paired within events; bootstrap resampling uses the event, never a prefix row, as the independent unit.",
        "",
    ]
    report = "\n".join(lines)
    atomic_write_text(output / "report.md", report)
    atomic_write_json(EXPERIMENT_ROOT / "atlas_pilot/summary.json", summary)
    atomic_write_text(EXPERIMENT_ROOT / "atlas_pilot/report.md", report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
