from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .io_utils import atomic_write_json, atomic_write_text, load_jsonl, write_once_jsonl
from .settings import ARTIFACT_ROOT, DETECTION_PREFIX, EXECUTION_HORIZONS, EXPERIMENT_ROOT, PRIMARY_TASKS


def pair_methods(rows: list[dict[str, Any]], left: str, right: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("available") and row["method"] in {left, right}:
            grouped[row["event_id"]][row["method"]] = row
    return [
        (values[left], values[right])
        for values in grouped.values()
        if left in values and right in values
    ]


def classify(left: dict[str, Any], right: dict[str, Any]) -> str:
    if left["safe_success"] > right["safe_success"]:
        return "safe_success_gain"
    if left["safe_success"] < right["safe_success"]:
        return "safe_success_loss"
    if left["cause_violation"] < right["cause_violation"]:
        return "cause_safety_gain"
    if left["cause_violation"] > right["cause_violation"]:
        return "cause_safety_loss"
    if left["new_non_nominal_actions"] < right["new_non_nominal_actions"]:
        return "equal_outcome_efficiency_gain"
    if left["new_non_nominal_actions"] > right["new_non_nominal_actions"]:
        return "equal_outcome_efficiency_loss"
    return "no_observed_difference"


def mechanism_case(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    category = classify(left, right)
    delay = int(left["prefix_k"]) - int(right["prefix_k"])
    if category in {"safe_success_gain", "equal_outcome_efficiency_gain"}:
        proximate = (
            "cached suffix retained task-directed actions before actor feedback, reducing ACT h=1 calls/rework"
            if delay > 0
            else "oracle branch selection chose a lower-rework/successful branch from the same realized R(k) set"
        )
    elif category in {"safe_success_loss", "cause_safety_loss"}:
        proximate = "continued cached actions crossed the absorbing task-cause boundary before replanning"
    elif category == "equal_outcome_efficiency_loss":
        proximate = "selected prefix did not offset later ACT recovery actions despite retaining cached actions"
    else:
        proximate = "both methods resolved to branches with the same measured safety and rework outcome"
    return {
        "event_id": left["event_id"],
        "task": left["task"],
        "actor_seed": left["actor_seed"],
        "severity_id": left["severity_id"],
        "comparison": f"{left['method']}_vs_{right['method']}",
        "category": category,
        "left_prefix_k": left["prefix_k"],
        "right_prefix_k": right["prefix_k"],
        "delay_difference": delay,
        "safe_success_difference": left["safe_success"] - right["safe_success"],
        "cause_violation_difference": left["cause_violation"] - right["cause_violation"],
        "new_action_difference": left["new_non_nominal_actions"] - right["new_non_nominal_actions"],
        "policy_call_difference": left["policy_calls"] - right["policy_calls"],
        "completion_step_difference": left["completion_steps"] - right["completion_steps"],
        "retained_action_difference": left["retained_nominal_actions"] - right["retained_nominal_actions"],
        "post_detection_budget_equal": left["post_detection_budget"] == right["post_detection_budget"],
        "proximate_code_mechanism": proximate,
        "causal_scope": "within-event branch counterfactual under frozen actor, simulator state reconstruction, and equal budget",
    }


def chunk_diagnostics() -> dict[str, Any]:
    episodes = load_jsonl(ARTIFACT_ROOT / "chunk_executability/raw_episodes.jsonl")
    prefixes = load_jsonl(ARTIFACT_ROOT / "chunk_executability/prefix_records.jsonl")
    output: dict[str, Any] = {}
    for task in PRIMARY_TASKS:
        task_rows = [row for row in episodes if row["task"] == task]
        h1 = [row for row in task_rows if row["execution_horizon"] == 1]
        h1_success = float(np.mean([row.get("clean_success", False) for row in h1]))
        metrics = {}
        for horizon in EXECUTION_HORIZONS:
            rows = [row for row in task_rows if row["execution_horizon"] == horizon]
            disagreements = [
                float(row["normalized_disagreement"])
                for row in prefixes
                if row.get("record_type") == "intermediate_disagreement"
                and row["task"] == task
                and int(row["execution_horizon"]) == horizon
                and row.get("normalized_disagreement") is not None
            ]
            metrics[str(horizon)] = {
                "success_rate": float(np.mean([row.get("clean_success", False) for row in rows])) if rows else None,
                "success_change_from_h1": (
                    float(np.mean([row.get("clean_success", False) for row in rows])) - h1_success if rows else None
                ),
                "object_drop_rate": float(np.mean([row.get("object_drop", False) for row in rows])) if rows else None,
                "wrong_release_rate": float(np.mean([row.get("wrong_release", False) for row in rows])) if rows else None,
                "phase_regression_rate": float(np.mean([row.get("phase_regression", False) for row in rows])) if rows else None,
                "fresh_disagreement_median": float(np.median(disagreements)) if disagreements else None,
                "fresh_disagreement_p95": float(np.quantile(disagreements, 0.95)) if disagreements else None,
                "fresh_disagreement_count": len(disagreements),
            }
        output[task] = metrics
    return output


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_comparison: dict[str, Any] = {}
    for comparison in sorted({case["comparison"] for case in cases}):
        rows = [case for case in cases if case["comparison"] == comparison]
        by_comparison[comparison] = {
            "event_count": len(rows),
            "categories": dict(Counter(row["category"] for row in rows)),
            "mean_delay_difference": float(np.mean([row["delay_difference"] for row in rows])) if rows else None,
            "mean_new_action_difference": float(np.mean([row["new_action_difference"] for row in rows])) if rows else None,
            "mean_policy_call_difference": float(np.mean([row["policy_call_difference"] for row in rows])) if rows else None,
            "all_budgets_equal": all(row["post_detection_budget_equal"] for row in rows),
        }
    return by_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="accepted for command symmetry; no new rollouts are launched")
    parser.parse_args()
    atlas = json.loads((ARTIFACT_ROOT / "atlas_pilot/summary.json").read_text())
    methods = load_jsonl(ARTIFACT_ROOT / "atlas_pilot/method_readouts.jsonl")
    strongest = atlas["strongest_baseline"]
    cases = [mechanism_case(left, right) for left, right in pair_methods(methods, "A_original", "B0")]
    if strongest:
        cases.extend(
            mechanism_case(left, right)
            for left, right in pair_methods(methods, "N_oracle", strongest)
        )
    chunk = chunk_diagnostics()
    case_summary = summarize_cases(cases)
    perturbation = json.loads((ARTIFACT_ROOT / "perturbation_qualification/summary.json").read_text())
    operators = json.loads((ARTIFACT_ROOT / "operator_audit/summary.json").read_text())
    summary = {
        "schema_version": 1,
        "method": "code-first realized-branch reverse explanation; no new idea generated",
        "chunk_executability_diagnostics": chunk,
        "atlas_counterfactual_summary": case_summary,
        "perturbation_mechanism": {
            "cream": "only the target bowl free joint is shifted; the absorbing label fires on a post-lift misaligned release",
            "stove": "the registered existing plate is placed relative to the actor chunk's nominal future bowl position; the absorbing label fires on bowl-plate contact",
            "qualification_status": perturbation["status"],
        },
        "operator_outcome": {
            "operator_router": operators["operator_router"],
            "local_repair": operators["track_a_local_repair"],
            "positive_labels_permitted": operators["positive_labels_permitted"],
            "descriptive_operator_router_criteria_met": operators[
                "descriptive_operator_router_criteria_met"
            ],
            "descriptive_local_repair_criteria_met": operators[
                "descriptive_local_repair_criteria_met"
            ],
            "unique_safe_win_rates": operators["unique_safe_win_event_rates"],
        },
        "interpretation_limits": [
            "N_oracle gains, if any, are selection gains over fully observed realized branches and are not deployable-policy evidence.",
            "A_original gains/losses are conditional on the two late-placement perturbation families and cannot reverse the immutable Stage-1b universal kill.",
            "Action-count changes are not a larger-budget artifact because all compared rows share the same event budget.",
            "No learned mechanism, learned router, new probe, VLA, or pi0.5 model was introduced.",
        ],
    }
    output = ARTIFACT_ROOT / "mechanism_audit"
    write_once_jsonl(output / "counterfactual_cases.jsonl", cases)
    atomic_write_json(output / "summary.json", summary)
    lines = [
        "# Code-first mechanism reverse audit",
        "",
        "This audit explains realized increases and decreases from the executed code paths; it does not generate a new idea.",
        "",
        "## Action-chunk substrate",
        "",
    ]
    for task, horizons in chunk.items():
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| h | success Δ vs h1 | drop | wrong release | regression | disagreement p95 |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for horizon, row in horizons.items():
            p95 = row["fresh_disagreement_p95"]
            lines.append(
                f"| {horizon} | {row['success_change_from_h1']:.3f} | {row['object_drop_rate']:.3f} | "
                f"{row['wrong_release_rate']:.3f} | {row['phase_regression_rate']:.3f} | "
                f"{p95:.3f} |" if p95 is not None else
                f"| {horizon} | {row['success_change_from_h1']:.3f} | {row['object_drop_rate']:.3f} | "
                f"{row['wrong_release_rate']:.3f} | {row['phase_regression_rate']:.3f} | n/a |"
            )
        lines.append("")
    lines.extend(["## Within-event Atlas counterfactuals", ""])
    for comparison, row in case_summary.items():
        lines.append(
            f"- `{comparison}`: {row['categories']}; mean new-action difference "
            f"`{row['mean_new_action_difference']:.3f}`, mean policy-call difference "
            f"`{row['mean_policy_call_difference']:.3f}`, equal budgets `{row['all_budgets_equal']}`."
        )
    lines.extend(
        [
            "",
            "The concrete gain path is cached task-directed suffix execution before h=1 ACT feedback, which can reduce rework and policy calls. The concrete loss path is the same continuation crossing an absorbing misaligned-release/contact boundary before feedback. Oracle gains additionally contain post-hoc branch-selection advantage and therefore are not deployable evidence.",
            "",
            f"Operator audit: `{operators['operator_router']}`; local repair: `{operators['track_a_local_repair']}`.",
            "",
        ]
    )
    report = "\n".join(lines)
    atomic_write_text(output / "report.md", report)
    atomic_write_json(EXPERIMENT_ROOT / "reports/mechanism_audit.json", summary)
    atomic_write_text(EXPERIMENT_ROOT / "reports/mechanism_audit.md", report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
