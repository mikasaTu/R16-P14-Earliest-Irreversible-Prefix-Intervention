from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from r16_p14_stage2b.io_utils import atomic_write_json, atomic_write_text, load_jsonl, write_once_jsonl

from .contracts import (
    freeze_strongest_baseline,
    hierarchical_cluster_bootstrap,
    last_recoverable_prefix,
    monotone_observed_safety,
    persistent_irreversibility_prefix,
    positive_label_allowed,
    select_crossfit_prefix,
)
from .settings import (
    ACTOR_SEEDS,
    ARTIFACT_ROOT,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DETECTION_PREFIX,
    EXPERIMENT_ROOT,
    H_VALID,
    PREFIX_INDICES,
)


def _mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else float("nan")


def _group(rows: Iterable[dict[str, Any]], *keys: str):
    result: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[tuple(row[key] for key in keys)].append(row)
    return result


def build_atlas(recovery: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    atlas = []
    for (instance, prefix), rows in sorted(_group(recovery, "event_instance_id", "prefix_k").items()):
        operators = {}
        for operator, operator_rows in _group(rows, "operator").items():
            operators[str(operator[0])] = {
                "C_family": _mean(operator_rows, "safe_success"),
                "safe_success_by_actor": {str(row["recovery_actor_seed"]): int(row["safe_success"]) for row in operator_rows},
                "new_non_nominal_actions_mean": _mean(operator_rows, "new_non_nominal_actions"),
                "actor_calls_mean": _mean(operator_rows, "actor_calls"),
            }
        generator = int(rows[0]["generator_actor_seed"])
        s_values = {bool(row["S_obs_at_k"]) for row in rows}
        atlas.append({
            "event_instance_id": instance, "event_id": rows[0]["event_id"],
            "task": rows[0]["task"], "split": rows[0]["split"],
            "init_state_id": rows[0]["init_state_id"], "generator_actor_seed": generator,
            "parameter_id": rows[0]["parameter_id"], "prefix_k": int(prefix),
            "S_obs": next(iter(s_values)) if len(s_values) == 1 else None,
            "S_obs_consistent": len(s_values) == 1,
            "C_self": max(int(row["safe_success"]) for row in rows if int(row["recovery_actor_seed"]) == generator),
            "R_U": any(item["C_family"] >= 2 / 3 for item in operators.values()),
            "operators": operators,
        })
    boundaries, invalid = [], []
    for (instance,), rows in sorted(_group(atlas, "event_instance_id").items()):
        safety_rows = [{"prefix_k": row["prefix_k"], "S_obs": row["S_obs"]} for row in rows if row["S_obs"] is not None]
        monotonicity = monotone_observed_safety(safety_rows) if len(safety_rows) == len(PREFIX_INDICES) else {"passed": False, "failure_label": "INCOMPLETE_PREFIX_GRID", "k_last_observed_safe": None}
        if not monotonicity["passed"]:
            invalid.append({"event_instance_id": instance, "reason": monotonicity["failure_label"], "monotonicity": monotonicity})
        complete = sorted(int(row["prefix_k"]) for row in rows) == list(PREFIX_INDICES)
        k_last_recoverable = last_recoverable_prefix(rows) if complete else None
        k_irrev = persistent_irreversibility_prefix(rows) if complete else None
        first = rows[0]
        boundaries.append({
            "event_instance_id": instance, "event_id": first["event_id"], "task": first["task"],
            "split": first["split"], "init_state_id": first["init_state_id"],
            "generator_actor_seed": first["generator_actor_seed"], "parameter_id": first["parameter_id"],
            "k_last_observed_safe": monotonicity["k_last_observed_safe"],
            "k_last_recoverable": k_last_recoverable, "k_irrev_U": k_irrev,
            "recoverable_window": None if k_last_recoverable is None else k_last_recoverable - DETECTION_PREFIX,
            "prefix_safety_valid": monotonicity["passed"],
            "recovery_pattern": [int(bool(row["R_U"])) for row in sorted(rows, key=lambda item: item["prefix_k"])],
            "recovery_nonmonotonic": any(not left["R_U"] and right["R_U"] for left, right in zip(sorted(rows, key=lambda item: item["prefix_k"]), sorted(rows, key=lambda item: item["prefix_k"])[1:])),
        })
    return atlas, boundaries, invalid


def track_a_rows(matched: list[dict[str, Any]], boundaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["event_instance_id"], int(row["prefix_k"]), row["branch"]): row for row in matched}
    result = []
    for boundary in boundaries:
        k = boundary["k_last_recoverable"]
        if k is None:
            continue
        cached = lookup.get((boundary["event_instance_id"], k, "CACHED_MATCHED"))
        fresh = lookup.get((boundary["event_instance_id"], k, "FRESH_MATCHED"))
        immediate = lookup.get((boundary["event_instance_id"], DETECTION_PREFIX, "FRESH_MATCHED"))
        if cached is None or fresh is None or immediate is None:
            continue
        result.append({
            **{key: boundary[key] for key in ("event_instance_id", "event_id", "task", "split", "init_state_id", "generator_actor_seed", "parameter_id", "k_last_recoverable", "recoverable_window")},
            "cached_safe_success": cached["safe_success"], "fresh_safe_success": fresh["safe_success"],
            "immediate_safe_success": immediate["safe_success"],
            "safe_success_difference": int(cached["safe_success"]) - int(fresh["safe_success"]),
            "cached_vs_immediate_safe_difference": int(cached["safe_success"]) - int(immediate["safe_success"]),
            "cached_cause_violation": cached["cause_violation"], "fresh_cause_violation": fresh["cause_violation"],
            "cached_new_actions": cached["new_non_nominal_actions"], "fresh_new_actions": fresh["new_non_nominal_actions"],
            "new_action_difference": float(fresh["new_non_nominal_actions"]) - float(cached["new_non_nominal_actions"]),
            "new_action_reduction_fraction": (float(fresh["new_non_nominal_actions"]) - float(cached["new_non_nominal_actions"])) / max(float(fresh["new_non_nominal_actions"]), 1.0),
            "cached_actor_calls": cached["actor_calls"], "fresh_actor_calls": fresh["actor_calls"],
            "cached_total_actions": cached["total_actions_executed"], "fresh_total_actions": fresh["total_actions_executed"],
            "cached_progress_at_k": cached["task_progress_retained_at_k"], "fresh_progress_at_k": fresh["task_progress_retained_at_k"],
            "action_displacement": cached["cached_fresh_action_displacement"],
            "compute_matched": cached["allocated_actor_call_budget"] == fresh["allocated_actor_call_budget"],
            "budget_matched": cached["total_post_detection_action_budget"] == fresh["total_post_detection_action_budget"],
        })
    return result


def _heuristic_positions(event: dict[str, Any], matched_rows: list[dict[str, Any]], boundary: dict[str, Any]) -> dict[str, int | None]:
    chunk = np.asarray(event["original_chunk"], dtype=np.float32)
    magnitudes = np.linalg.norm(chunk[DETECTION_PREFIX:H_VALID, :3], axis=1)
    velocity = DETECTION_PREFIX + int(np.argmin(magnitudes)) if len(magnitudes) else DETECTION_PREFIX
    disagreements = sorted(
        [(int(row["prefix_k"]), row.get("cached_fresh_action_displacement")) for row in matched_rows if row["branch"] == "CACHED_MATCHED"],
        key=lambda item: item[0],
    )
    disagreement = next((k for k, value in disagreements if value is not None and float(value) >= 1.0), H_VALID)
    return {
        "immediate_fresh_h16": DETECTION_PREFIX,
        "fixed_delay_1": DETECTION_PREFIX + 1,
        "fixed_delay_2": DETECTION_PREFIX + 2,
        "fixed_delay_4": DETECTION_PREFIX + 4,
        "fixed_delay_8": DETECTION_PREFIX + 8,
        "velocity_phase": velocity,
        "action_disagreement": disagreement,
        "k_last_observed_safe": boundary["k_last_observed_safe"],
        "k_last_recoverable": boundary["k_last_recoverable"],
    }


def baseline_and_crossfit(
    recovery: list[dict[str, Any]], matched: list[dict[str, Any]], boundaries: list[dict[str, Any]], events: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    fresh = [row for row in recovery if row["operator"] == "fresh_h16"]
    recovery_lookup = {(row["event_instance_id"], int(row["prefix_k"]), int(row["recovery_actor_seed"])): row for row in fresh}
    boundary_lookup = {row["event_instance_id"]: row for row in boundaries}
    matched_group = _group(matched, "event_instance_id")
    baseline_rows = []
    for instance, boundary in boundary_lookup.items():
        event = events[boundary["event_id"]]
        positions = _heuristic_positions(event, matched_group[(instance,)], boundary)
        for heldout in ACTOR_SEEDS:
            for method, k in positions.items():
                row = recovery_lookup.get((instance, int(k), heldout)) if k is not None else None
                if row is None:
                    continue
                baseline_rows.append({
                    **{key: boundary[key] for key in ("event_instance_id", "event_id", "task", "split", "init_state_id", "parameter_id")},
                    "heldout_actor_seed": heldout, "method": method, "prefix_k": k,
                    "safe_success": row["safe_success"], "new_non_nominal_actions": row["new_non_nominal_actions"],
                    "actor_calls": row["actor_calls"], "action_budget": row["total_post_detection_action_budget"],
                })
    calibration = [row for row in baseline_rows if row["split"] == "calibration"]
    frozen_baseline = freeze_strongest_baseline(calibration)
    crossfit = []
    for instance, boundary in boundary_lookup.items():
        for heldout in ACTOR_SEEDS:
            fit = [
                row for row in fresh
                if row["event_instance_id"] == instance and int(row["recovery_actor_seed"]) != heldout
            ]
            if not fit:
                continue
            k = select_crossfit_prefix(fit, heldout)
            selected = recovery_lookup.get((instance, k, heldout))
            baseline = next((row for row in baseline_rows if row["event_instance_id"] == instance and row["heldout_actor_seed"] == heldout and row["method"] == frozen_baseline), None)
            if selected is None or baseline is None:
                continue
            crossfit.append({
                **{key: boundary[key] for key in ("event_instance_id", "event_id", "task", "split", "init_state_id", "parameter_id", "k_last_recoverable")},
                "heldout_actor_seed": heldout, "selected_prefix": k,
                "frozen_baseline": frozen_baseline, "baseline_prefix": baseline["prefix_k"],
                "selected_safe_success": selected["safe_success"], "baseline_safe_success": baseline["safe_success"],
                "safe_success_difference": int(selected["safe_success"]) - int(baseline["safe_success"]),
                "selected_new_actions": selected["new_non_nominal_actions"], "baseline_new_actions": baseline["new_non_nominal_actions"],
                "new_action_difference": float(baseline["new_non_nominal_actions"]) - float(selected["new_non_nominal_actions"]),
                "new_action_reduction_fraction": (float(baseline["new_non_nominal_actions"]) - float(selected["new_non_nominal_actions"])) / max(float(baseline["new_non_nominal_actions"]), 1.0),
                "selected_actor_calls": selected["actor_calls"], "baseline_actor_calls": baseline["actor_calls"],
                "selected_action_budget": selected["total_post_detection_action_budget"], "baseline_action_budget": baseline["action_budget"],
                "differs_from_d": k != DETECTION_PREFIX,
                "differs_from_last_recoverable": boundary["k_last_recoverable"] is not None and k != boundary["k_last_recoverable"],
                "interior_prefix": DETECTION_PREFIX < k < H_VALID,
            })
    return baseline_rows, frozen_baseline, crossfit


def _breakdown(rows: list[dict[str, Any]], difference: str) -> dict[str, Any]:
    result = {"overall": {"n": len(rows), "mean": _mean(rows, difference)}}
    for key in ("task", "parameter_id", "generator_actor_seed", "heldout_actor_seed"):
        if not any(key in row for row in rows):
            continue
        result[key] = {
            str(value[0]): {"n": len(group), "mean": _mean(group, difference)}
            for value, group in sorted(_group([row for row in rows if key in row], key).items())
        }
    return result


def mechanism_audit(matched: list[dict[str, Any]], track_a: list[dict[str, Any]], recovery: list[dict[str, Any]]) -> dict[str, Any]:
    paired = []
    lookup = _group(matched, "event_instance_id", "prefix_k")
    for (_, _), rows in lookup.items():
        cached = next((row for row in rows if row["branch"] == "CACHED_MATCHED"), None)
        fresh = next((row for row in rows if row["branch"] == "FRESH_MATCHED"), None)
        if cached and fresh:
            paired.append({
                "safe_delta": int(cached["safe_success"]) - int(fresh["safe_success"]),
                "cause_delta": int(cached["cause_violation"]) - int(fresh["cause_violation"]),
                "new_action_delta": float(fresh["new_non_nominal_actions"]) - float(cached["new_non_nominal_actions"]),
                "progress_delta": float(cached["task_progress_retained_at_k"]) - float(fresh["task_progress_retained_at_k"]),
                "disagreement": cached.get("cached_fresh_action_displacement") or 0.0,
                "prefix_k": cached["prefix_k"], "task": cached["task"],
            })
    groups = {
        "cached_improves": [row for row in paired if row["safe_delta"] > 0],
        "cached_worsens": [row for row in paired if row["safe_delta"] < 0],
        "same_safe_outcome": [row for row in paired if row["safe_delta"] == 0],
    }
    operator_rows = {
        str(key[0]): {
            "safe_success": _mean(rows, "safe_success"),
            "cause_violation": _mean(rows, "cause_violation"),
            "new_actions": _mean(rows, "new_non_nominal_actions"),
        }
        for key, rows in sorted(_group(recovery, "operator").items())
    }
    return {
        "schema_version": 1,
        "purpose": "reverse-explain observed increases/decreases from frozen code paths; no new idea generated",
        "paired_prefix_count": len(paired),
        "outcome_groups": {
            name: {
                "count": len(rows), "mean_action_disagreement": _mean(rows, "disagreement"),
                "mean_cached_minus_fresh_progress": _mean(rows, "progress_delta"),
                "mean_fresh_minus_cached_new_actions": _mean(rows, "new_action_delta"),
                "mean_cached_minus_fresh_cause": _mean(rows, "cause_delta"),
            }
            for name, rows in groups.items()
        },
        "operator_summary": operator_rows,
        "code_path_explanation": {
            "cached": "executes the immutable actor chunk A[d:k], then replans with h=16",
            "fresh": "executes the detection-time ACT chunk for exactly k-d actions, then replans with h=16",
            "improvement_test": "whether preserved nominal progress and fewer newly generated actions occur without additional target-cause violations",
            "decrease_test": "whether stale-prefix cause violations or large cached/fresh action displacement account for lower safe success",
            "compute_control": "both primary paths make one detection-time call, one call at k, and use the same h=16 tail/action budget",
        },
        "track_a_boundary_rows": len(track_a),
        "new_idea_generated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    if (ARTIFACT_ROOT / "decision.json").is_file():
        print("STAGE2C_AGGREGATE_ALREADY_COMPLETE")
        return
    matched = load_jsonl(ARTIFACT_ROOT / "formal_matrix/matched_prefix_rows.jsonl")
    recovery = load_jsonl(ARTIFACT_ROOT / "formal_matrix/recovery_operator_rows.jsonl")
    formal_summary = json.loads((ARTIFACT_ROOT / "formal_matrix/summary.json").read_text())
    qualification = json.loads((ARTIFACT_ROOT / "task_qualification/summary.json").read_text())
    pool_summary = json.loads((ARTIFACT_ROOT / "actor_events/formal_event_pool_summary.json").read_text())
    contract = json.loads((ARTIFACT_ROOT / "contract_repair/summary.json").read_text())
    events = {event["event_id"]: event for event in load_jsonl(ARTIFACT_ROOT / "actor_events/events.jsonl")}

    atlas, boundaries, invalid = build_atlas(recovery)
    track_a = track_a_rows(matched, boundaries)
    baseline_rows, frozen_baseline, crossfit = baseline_and_crossfit(recovery, matched, boundaries, events)
    evaluation_a = [row for row in track_a if row["split"] == "evaluation"]
    evaluation_b = [row for row in crossfit if row["split"] == "evaluation"]
    stats = {
        "track_a_safe_success": hierarchical_cluster_bootstrap(evaluation_a, lambda row: row["safe_success_difference"], replicates=args.bootstrap_replicates, seed=BOOTSTRAP_SEED) if evaluation_a else None,
        "track_a_new_actions": hierarchical_cluster_bootstrap(evaluation_a, lambda row: row["new_action_difference"], replicates=args.bootstrap_replicates, seed=BOOTSTRAP_SEED + 1) if evaluation_a else None,
        "track_b_safe_success": hierarchical_cluster_bootstrap(evaluation_b, lambda row: row["safe_success_difference"], replicates=args.bootstrap_replicates, seed=BOOTSTRAP_SEED + 2) if evaluation_b else None,
        "track_b_new_actions": hierarchical_cluster_bootstrap(evaluation_b, lambda row: row["new_action_difference"], replicates=args.bootstrap_replicates, seed=BOOTSTRAP_SEED + 3) if evaluation_b else None,
        "track_a_breakdown": _breakdown(evaluation_a, "safe_success_difference"),
        "track_b_breakdown": _breakdown(evaluation_b, "safe_success_difference"),
    }
    tasks = sorted({row["task"] for row in evaluation_a})
    median_windows = {
        task: float(np.median([row["recoverable_window"] for row in evaluation_a if row["task"] == task]))
        for task in tasks
    }
    a_reduction = _mean(evaluation_a, "new_action_reduction_fraction") if evaluation_a else float("nan")
    a_seed_direction = {
        seed: _mean([row for row in evaluation_a if int(row["generator_actor_seed"]) == seed], "safe_success_difference") >= -0.03
        and _mean([row for row in evaluation_a if int(row["generator_actor_seed"]) == seed], "new_action_reduction_fraction") >= 0.15
        for seed in ACTOR_SEEDS
    }
    a_task_direction = {
        task: _mean([row for row in evaluation_a if row["task"] == task], "safe_success_difference") >= -0.03
        and _mean([row for row in evaluation_a if row["task"] == task], "new_action_reduction_fraction") >= 0.15
        for task in tasks
    }
    a_severity_direction = {
        key[0]: _mean(rows, "safe_success_difference") >= -0.03 and _mean(rows, "new_action_reduction_fraction") >= 0.15
        for key, rows in _group(evaluation_a, "parameter_id").items()
    }
    track_a_checks = {
        "both_tasks_median_window_ge_2": len(median_windows) == 2 and all(value >= 2 for value in median_windows.values()),
        "safe_ci_lower_ge_minus_0_03": bool(stats["track_a_safe_success"] and stats["track_a_safe_success"]["ci95"][0] >= -0.03),
        "new_actions_reduced_ge_15pct": bool(np.isfinite(a_reduction) and a_reduction >= 0.15),
        "direction_both_tasks": len(a_task_direction) == 2 and all(a_task_direction.values()),
        "direction_two_of_three_generator_seeds": sum(a_seed_direction.values()) >= 2,
        "direction_all_severities": bool(a_severity_direction) and all(a_severity_direction.values()),
        "all_compute_matched": bool(evaluation_a) and all(row["compute_matched"] for row in evaluation_a),
        "all_budget_matched": bool(evaluation_a) and all(row["budget_matched"] for row in evaluation_a),
    }

    b_safe = _mean(evaluation_b, "safe_success_difference") if evaluation_b else float("nan")
    b_reduction = _mean(evaluation_b, "new_action_reduction_fraction") if evaluation_b else float("nan")
    b_claim_safe = np.isfinite(b_safe) and b_safe >= 0.08
    b_claim_actions = np.isfinite(b_safe) and b_safe >= -0.03 and np.isfinite(b_reduction) and b_reduction >= 0.15
    b_primary_stats = stats["track_b_safe_success"] if b_claim_safe else stats["track_b_new_actions"]
    b_task_direction = {
        task: _mean([row for row in evaluation_b if row["task"] == task], "safe_success_difference") > 0
        or (_mean([row for row in evaluation_b if row["task"] == task], "safe_success_difference") >= -0.03 and _mean([row for row in evaluation_b if row["task"] == task], "new_action_reduction_fraction") >= 0.15)
        for task in sorted({row["task"] for row in evaluation_b})
    }
    b_seed_direction = {
        seed: _mean([row for row in evaluation_b if int(row["heldout_actor_seed"]) == seed], "safe_success_difference") > 0
        or (_mean([row for row in evaluation_b if int(row["heldout_actor_seed"]) == seed], "safe_success_difference") >= -0.03 and _mean([row for row in evaluation_b if int(row["heldout_actor_seed"]) == seed], "new_action_reduction_fraction") >= 0.15)
        for seed in ACTOR_SEEDS
    }
    track_b_checks = {
        "primary_gain_threshold": b_claim_safe or b_claim_actions,
        "claimed_gain_ci_lower_gt_0": bool(b_primary_stats and b_primary_stats["ci95"][0] > 0),
        "selected_differs_from_d_ge_20pct": _mean(evaluation_b, "differs_from_d") >= 0.20 if evaluation_b else False,
        "selected_differs_from_last_recoverable_ge_20pct": _mean(evaluation_b, "differs_from_last_recoverable") >= 0.20 if evaluation_b else False,
        "interior_selected_ge_20pct": _mean(evaluation_b, "interior_prefix") >= 0.20 if evaluation_b else False,
        "positive_both_tasks": len(b_task_direction) == 2 and all(b_task_direction.values()),
        "positive_two_of_three_heldout_seeds": sum(b_seed_direction.values()) >= 2,
        "action_and_call_budgets_equal": bool(evaluation_b) and all(row["selected_action_budget"] == row["baseline_action_budget"] and row["selected_actor_calls"] == row["baseline_actor_calls"] for row in evaluation_b),
    }

    upstream = {
        "contract_repair": contract["status"] == "PASS",
        "replay_contract": formal_summary["error_count"] == 0 and not invalid,
        "second_failure_family": qualification["second_failure_family_pass"] and qualification["target_shift_two_severities"],
        "minimum_data": pool_summary["minimum_data_pass"],
        "formal_matrix": formal_summary["all_complete"],
    }
    allow_positive = positive_label_allowed(upstream)
    a_signal_raw = all(track_a_checks.values())
    b_signal_raw = all(track_b_checks.values())
    status_a = "SIGNAL" if allow_positive and a_signal_raw else ("NO_SIGNAL" if allow_positive else "INCONCLUSIVE")
    status_b = "SIGNAL" if allow_positive and b_signal_raw else ("NO_SIGNAL" if allow_positive else "INCONCLUSIVE")
    if not upstream["replay_contract"]:
        overall = "BLOCKED_BY_REPLAY_CONTRACT"
    elif not upstream["second_failure_family"]:
        overall = "BLOCKED_BY_SECOND_FAILURE_FAMILY"
    elif not upstream["minimum_data"]:
        overall = "BLOCKED_BY_MINIMUM_DATA"
    elif not upstream["formal_matrix"] or not upstream["contract_repair"]:
        overall = "BLOCKED_BY_INFRA"
    elif status_a == "SIGNAL" and status_b == "SIGNAL":
        overall = "PROCEED_TO_LEARNED_REPLANABILITY"
    elif status_b == "SIGNAL":
        overall = "PROCEED_TO_LEARNED_REPLANABILITY"
    elif status_a == "SIGNAL":
        overall = "PROCEED_TO_OPERATOR_RELATIVE_BOUNDARY_ONLY"
    else:
        overall = "STOP_R16_P14_PREFIX_TIMING_FAMILY"
    mechanism = mechanism_audit(matched, track_a, recovery)
    decision = {
        "schema_version": 1,
        "stage1b_universal_hypothesis": "KILLED_IMMUTABLE",
        "stage2b_status": "BLOCKED_UPSTREAM",
        "contract_repair": "PASS" if upstream["contract_repair"] else "BLOCKED",
        "replay_contract": "PASS" if upstream["replay_contract"] else "BLOCKED",
        "second_failure_family": "PASS" if upstream["second_failure_family"] else "BLOCKED",
        "track_a_operator_relative_prefix_reuse": status_a,
        "track_b_crossfit_replanability": status_b,
        "local_repair": "RETIRED_NO_SIGNAL",
        "operator_router": "RETIRED_NO_SIGNAL",
        "overall": overall,
        "accepted": False,
        "novelty": "N2_ORACLE_PROTOCOL_BOUNDARY_ONLY",
        "all_experiments_continued_after_failed_gates": True,
        "upstream_gates": upstream,
        "track_a_raw_signal": a_signal_raw,
        "track_b_raw_signal": b_signal_raw,
        "track_a_checks": track_a_checks,
        "track_b_checks": track_b_checks,
        "frozen_strongest_baseline": frozen_baseline,
    }
    output = ARTIFACT_ROOT
    write_once_jsonl(output / "recoverability_atlas/rows.jsonl", atlas)
    write_once_jsonl(output / "recoverability_atlas/event_boundaries.jsonl", boundaries)
    write_once_jsonl(output / "recoverability_atlas/invalid_events.jsonl", invalid)
    write_once_jsonl(output / "matched_prefix/track_a_rows.jsonl", track_a)
    write_once_jsonl(output / "crossfit_replanability/baseline_rows.jsonl", baseline_rows)
    write_once_jsonl(output / "crossfit_replanability/crossfit_rows.jsonl", crossfit)
    atomic_write_json(output / "statistics.json", stats)
    atomic_write_json(output / "mechanism_audit.json", mechanism)
    atomic_write_json(output / "decision.json", decision)
    atomic_write_json(output / "aggregate_summary.json", {
        "atlas_rows": len(atlas), "boundary_rows": len(boundaries), "invalid_events": len(invalid),
        "track_a_rows": len(track_a), "crossfit_rows": len(crossfit),
        "median_recoverable_windows": median_windows,
        "track_a_new_action_reduction_fraction": a_reduction,
        "track_b_safe_success_difference": b_safe,
        "track_b_new_action_reduction_fraction": b_reduction,
    })
    atomic_write_text(output / "mechanism_audit.md", "\n".join([
        "# Stage-2C mechanism reverse explanation", "",
        "This audit explains the measured increase/decrease using the frozen code paths; it does not generate a new idea.", "",
        f"Paired prefix cells: {mechanism['paired_prefix_count']}. Cached-improves: {mechanism['outcome_groups']['cached_improves']['count']}; cached-worsens: {mechanism['outcome_groups']['cached_worsens']['count']}.", "",
        "Cached and fresh branches differ only in the `k-d` executed prefix. Both receive the same detection-time ACT call, call again at k, and use the same h=16 tail and action budget. Improvement is attributed only when nominal progress/fewer new actions survive without additional cause violations; degradation is tested against stale-prefix violations and cached/fresh action displacement.", "",
    ]))
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

