from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .io_utils import atomic_write_json, load_jsonl, sha256_file
from .settings import (
    ACTOR_SEEDS,
    ARTIFACT_ROOT,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPERIMENT_ROOT,
    MIRROR_EXPERIMENT_OUTPUTS,
    SAFETY_NONINFERIORITY_EPSILON,
    TASKS,
    DIAGNOSTIC_ONLY_GLOBAL,
    FORMAL_POSITIVE_EVIDENCE_ALLOWED,
)


EFFICIENCY_FIELDS = (
    "actual_post_detection_actions",
    "completion_steps",
    "actual_actor_calls",
    "actual_inference_wall_time_s",
    "eef_path_length_m",
    "manipulated_object_path_length_m",
    "total_branch_wall_time_s",
    "progress_regression_m",
)


def valid_paired_rows() -> list[dict[str, Any]]:
    rows = load_jsonl(ARTIFACT_ROOT / "confirmatory_evaluation/paired_rows.jsonl")
    return [
        row
        for row in rows
        if row["replay_admitted_3_of_3"]
        and all(not method.get("error") for method in row["methods"].values())
    ]


def paired_values(
    rows: list[dict[str, Any]], method: str, baseline: str, field: str
) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    for row in rows:
        if method not in row["methods"] or baseline not in row["methods"]:
            continue
        left, right = row["methods"][method].get(field), row["methods"][baseline].get(field)
        if left is None or right is None:
            continue
        if not (math.isfinite(float(left)) and math.isfinite(float(right))):
            continue
        pairs.append((float(left), float(right)))
    if not pairs:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    array = np.asarray(pairs, dtype=np.float64)
    return array[:, 0], array[:, 1]


def _source_rollout_value(row: dict[str, Any]) -> str:
    """Return the source-rollout cluster id for repeated actor measurements.

    Formal rows carry ``init_state_id`` as the stable source-rollout identity.
    Newer producers may provide an explicit ``source_rollout_id``; that value
    takes precedence.  ``event_id`` is deliberately not a fallback because it
    includes the actor seed in the current event writer and would make the same
    init state look like three independent bootstrap samples.
    """
    value = row.get("source_rollout_id")
    if value is None:
        value = row.get("source_event_id")
    if value is None:
        value = row.get("init_state_id")
    if value is None:
        raise ValueError("paired row has no init_state/source rollout cluster id")
    return str(value)


def _cluster_key(row: dict[str, Any], cluster_fields: tuple[str, ...]) -> tuple[str, ...]:
    values = []
    for field in cluster_fields:
        if field in {"source_rollout", "init_state_id"}:
            values.append(_source_rollout_value(row))
        else:
            value = row.get(field)
            if value is None:
                raise ValueError(f"paired row has no cluster field: {field}")
            values.append(str(value))
    return tuple(values)


def cluster_paired_values(
    rows: list[dict[str, Any]],
    method: str,
    baseline: str,
    field: str,
    cluster_fields: tuple[str, ...] = ("task", "source_rollout"),
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate source-rollout clusters before paired bootstrap resampling.

    The overall comparison uses ``(task, init_state/source_rollout)``.  A
    severity comparison adds ``parameter_id`` so different perturbations from
    one init state do not become one observation.  Actor-seed strata are
    reported separately and add ``actor_seed`` to the key.  This prevents the
    same source init state rendered by three frozen actor checkpoints from
    being counted as three independent bootstrap samples.
    """
    grouped: dict[tuple[str, ...], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if method not in row["methods"] or baseline not in row["methods"]:
            continue
        left = row["methods"][method].get(field)
        right = row["methods"][baseline].get(field)
        if left is None or right is None:
            continue
        if not (math.isfinite(float(left)) and math.isfinite(float(right))):
            continue
        key = _cluster_key(row, cluster_fields)
        grouped[key].append((float(left), float(right)))
    if not grouped:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    values = np.asarray(
        [[float(np.mean([pair[index] for pair in pairs])) for index in (0, 1)] for pairs in grouped.values()],
        dtype=np.float64,
    )
    return values[:, 0], values[:, 1]


def bootstrap_mean_difference(left: np.ndarray, right: np.ndarray, seed: int) -> dict[str, Any]:
    n = len(left)
    if n == 0:
        return {"n": 0, "estimate": None, "ci95": [None, None]}
    difference = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n))
    draws = difference[indices].mean(axis=1)
    p_two_sided = float(
        min(1.0, 2.0 * min(np.mean(draws <= 0), np.mean(draws >= 0)))
    )
    return {
        "n": n,
        "estimate": float(difference.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "p_two_sided_bootstrap": p_two_sided,
    }


def holm_bonferroni(p_values: dict[str, float | None]) -> dict[str, float | None]:
    """Return Holm-adjusted p values without adding a statistics dependency."""
    finite = [(name, float(value)) for name, value in p_values.items() if value is not None]
    ordered = sorted(finite, key=lambda item: item[1])
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return adjusted


def bootstrap_reduction(left: np.ndarray, right: np.ndarray, seed: int) -> dict[str, Any]:
    n = len(left)
    if n == 0:
        return {"n": 0, "estimate": None, "ci95": [None, None]}
    denominator = float(right.mean())
    estimate = 0.0 if denominator == 0 else (denominator - float(left.mean())) / denominator
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n))
    left_draw = left[indices].mean(axis=1)
    right_draw = right[indices].mean(axis=1)
    draws = np.divide(
        right_draw - left_draw,
        right_draw,
        out=np.zeros_like(right_draw),
        where=right_draw != 0,
    )
    return {
        "n": n,
        "estimate": float(estimate),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def exact_binary_table(rows: list[dict[str, Any]], method: str, baseline: str, field: str) -> dict[str, int]:
    table = Counter()
    for row in rows:
        if method in row["methods"] and baseline in row["methods"]:
            left = int(bool(row["methods"][method][field]))
            right = int(bool(row["methods"][baseline][field]))
            table[f"method_{left}__baseline_{right}"] += 1
    return dict(sorted(table.items()))


def comparison(
    rows: list[dict[str, Any]],
    method: str,
    baseline: str,
    seed_offset: int = 0,
    cluster_fields: tuple[str, ...] = ("task", "source_rollout"),
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": method,
        "baseline": baseline,
        "events": len(rows),
        "binary_tables": {},
        "differences": {},
        "relative_reductions": {},
        "holm_adjusted_p_values": {},
        "bootstrap_unit": "source rollout cluster; prefix rows averaged within cluster",
        "bootstrap_cluster_fields": list(cluster_fields),
    }
    for index, field in enumerate(("safe_success", "cause_violation", "task_success")):
        left, right = cluster_paired_values(
            rows, method, baseline, field, cluster_fields=cluster_fields
        )
        result["differences"][field] = bootstrap_mean_difference(
            left, right, BOOTSTRAP_SEED + seed_offset + index
        )
        result["binary_tables"][field] = exact_binary_table(rows, method, baseline, field)
    for index, field in enumerate(EFFICIENCY_FIELDS, start=10):
        left, right = cluster_paired_values(
            rows, method, baseline, field, cluster_fields=cluster_fields
        )
        result["differences"][field] = bootstrap_mean_difference(
            left, right, BOOTSTRAP_SEED + seed_offset + index
        )
        result["relative_reductions"][field] = bootstrap_reduction(
            left, right, BOOTSTRAP_SEED + seed_offset + 100 + index
        )
    result["holm_adjusted_p_values"] = holm_bonferroni(
        {
            field: metrics.get("p_two_sided_bootstrap")
            for field, metrics in result["differences"].items()
        }
    )
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    xr, yr = _rankdata(np.asarray(x, dtype=np.float64)), _rankdata(np.asarray(y, dtype=np.float64))
    return float(np.corrcoef(xr, yr)[0, 1])


def _auc(labels: list[bool], scores: list[float]) -> float | None:
    if not labels or len(set(labels)) < 2:
        return None
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    return float(
        (sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative))
        / (len(positive) * len(negative))
    )


def diagnostic_error_signal() -> dict[str, Any]:
    """Summarize observed outcome/error proxies without changing a selector.

    Stage-2D has no learned world model, ensemble uncertainty, or visual
    encoder.  We therefore report those covariates as unavailable and only
    use measured branch rows: cached/fresh action disagreement, measured
    object-path regression, and explicit cause violations.  All values are
    diagnostic-only because qualification failed upstream.
    """
    path = ARTIFACT_ROOT / "calibration_atlas/rows.jsonl"
    if not path.is_file():
        return {"status": "UNAVAILABLE", "reason": "calibration atlas absent"}
    rows = [
        row
        for row in load_jsonl(path)
        if row.get("requested_arm") == "CACHED_MATCHED" and not row.get("error")
    ]
    usable = [
        row
        for row in rows
        if row.get("cached_fresh_action_disagreement") is not None
        and row.get("progress_regression_m") is not None
    ]
    if not usable:
        return {"status": "UNAVAILABLE", "reason": "no usable cached rows"}
    x = [float(row["cached_fresh_action_disagreement"]) for row in usable]
    y = [float(row["progress_regression_m"]) for row in usable]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        grouped[_cluster_key(row, ("task", "source_rollout"))].append(row)
    centered_x, centered_y, per_event = [], [], []
    for event_id, event_rows in sorted(grouped.items()):
        ex = [float(row["cached_fresh_action_disagreement"]) for row in event_rows]
        ey = [float(row["progress_regression_m"]) for row in event_rows]
        centered_x.extend(value - float(np.mean(ex)) for value in ex)
        centered_y.extend(value - float(np.mean(ey)) for value in ey)
        corr = _spearman(ex, ey)
        if corr is not None:
            per_event.append(corr)
    threshold = float(np.quantile(y, 0.75))
    labels = [value >= threshold or bool(row.get("cause_violation")) for row, value in zip(usable, y)]
    high_error_auc = _auc(labels, x)
    # A within-event fixed-effect slope is the requested episode random-
    # intercept diagnostic in a deterministic, dependency-free form.
    if len(centered_x) >= 3 and float(np.dot(centered_x, centered_x)) > 0:
        slope = float(np.dot(centered_x, centered_y) / np.dot(centered_x, centered_x))
    else:
        slope = None
    return {
        "status": "COMPLETE",
        "diagnostic_only": True,
        "row_count": len(usable),
        "event_count": len(grouped),
        "error_proxy": "progress_regression_m_or_explicit_cause_violation",
        "score_proxy": "cached_fresh_action_disagreement",
        "global_spearman": _spearman(x, y),
        "state_centered_spearman": _spearman(centered_x, centered_y),
        "within_event_spearman_distribution": {
            "count": len(per_event),
            "median": float(np.median(per_event)) if per_event else None,
            "positive_fraction": float(np.mean(np.asarray(per_event) > 0)) if per_event else None,
            "values": per_event,
        },
        "positive_correlation_event_fraction": float(np.mean(np.asarray(per_event) > 0)) if per_event else None,
        "high_error_threshold": threshold,
        "high_error_auroc": high_error_auc,
        "mixed_effect_episode_intercept_slope": slope,
        "mixed_effect_controls": {
            "canonical_task_loss": "unavailable; no learned WM loss in Stage-2D",
            "ensemble_uncertainty": "unavailable; no ensemble in Stage-2D",
            "action_magnitude": "unavailable as a direct command norm; EEF path is retained separately",
            "state_only_variance": "unavailable; no state-only WM branch",
            "initial_dino_orbit_floor": "not applicable; no DINO branch",
            "true_future_dino_orbit_floor": "not applicable; no DINO branch",
            "episode_random_intercept": "demeaned within task+init_state/source_rollout cluster",
        },
        "holm_correction": "applied to paired arm bootstrap p-values; no positive label is inferred",
    }


def grouped_comparisons(rows: list[dict[str, Any]], method: str, baseline: str) -> dict[str, Any]:
    result = {
        "overall": comparison(
            rows,
            method,
            baseline,
            cluster_fields=("task", "source_rollout"),
        )
    }
    dimensions = {
        "task": lambda row: row["task"],
        "severity": lambda row: f"{row['task']}::{row['parameter_id']}",
        "actor_seed": lambda row: str(row["actor_seed"]),
    }
    cluster_fields_by_dimension = {
        "task": ("task", "source_rollout"),
        "severity": ("task", "parameter_id", "source_rollout"),
        "actor_seed": ("task", "parameter_id", "source_rollout", "actor_seed"),
    }
    for dimension, getter in dimensions.items():
        groups = defaultdict(list)
        for row in rows:
            groups[getter(row)].append(row)
        result[dimension] = {
            name: comparison(
                group,
                method,
                baseline,
                seed_offset=1000 + index * 100,
                cluster_fields=cluster_fields_by_dimension[dimension],
            )
            for index, (name, group) in enumerate(sorted(groups.items()))
        }
    return result


def safety_noninferior(result: dict[str, Any]) -> bool:
    safe = result["differences"]["safe_success"]
    cause = result["differences"]["cause_violation"]
    return bool(
        safe["n"] > 0
        and safe["ci95"][0] >= -SAFETY_NONINFERIORITY_EPSILON
        and cause["ci95"][1] <= SAFETY_NONINFERIORITY_EPSILON
    )


def efficient(result: dict[str, Any]) -> tuple[bool, list[str]]:
    improved = [
        field
        for field, metrics in result["relative_reductions"].items()
        if metrics["estimate"] is not None and metrics["estimate"] >= 0.15
    ]
    return bool(improved), improved


def agreement(grouped: dict[str, Any]) -> dict[str, Any]:
    task_agree = all(safety_noninferior(value) and efficient(value)[0] for value in grouped["task"].values())
    severity_agree = all(
        safety_noninferior(value) and efficient(value)[0]
        for value in grouped["severity"].values()
    )
    seed_passes = {
        seed: safety_noninferior(value) and efficient(value)[0]
        for seed, value in grouped["actor_seed"].items()
    }
    return {
        "both_tasks_agree": task_agree and len(grouped["task"]) == 2,
        "both_severities_per_task_agree": severity_agree
        and len(grouped["severity"]) >= 4,
        "actor_seed_passes": seed_passes,
        "at_least_two_of_three_actor_seeds_agree": sum(seed_passes.values()) >= 2,
    }


def calibration_gap_recovery(strongest_method: str) -> dict[str, Any]:
    rows = [row for row in load_jsonl(ARTIFACT_ROOT / "calibration_atlas/rows.jsonl") if not row.get("error")]
    pool = {
        item["event_instance_id"]: item
        for item in load_jsonl(ARTIFACT_ROOT / "actor_events/formal_event_pool.jsonl")
        if item["split"] == "calibration"
    }
    events = {
        event["event_id"]: event
        for event in load_jsonl(ARTIFACT_ROOT / "actor_events/events.jsonl")
    }
    oracle = {
        row["event_instance_id"]: row
        for row in load_jsonl(ARTIFACT_ROOT / "calibration_atlas/oracle_rows.jsonl")
    }
    delay = int(strongest_method.rsplit("_", 1)[1])
    fixed_k = 2 + delay
    by_event = defaultdict(list)
    for row in rows:
        if row["requested_arm"] == "CACHED_MATCHED":
            by_event[row["event_instance_id"]].append(row)
    fixed_actions, rule_actions, oracle_actions = [], [], []
    interior, differs = [], []
    for event_instance_id, item in pool.items():
        candidates = by_event[event_instance_id]
        if event_instance_id not in oracle:
            continue
        from .freeze_rule import rule_k

        k_rule = rule_k(events[item["event_id"]], item["parameter"])
        fixed = next((row for row in candidates if int(row["requested_prefix_k"]) == fixed_k), None)
        ruled = next((row for row in candidates if int(row["requested_prefix_k"]) == k_rule), None)
        if fixed is None or ruled is None:
            continue
        fixed_actions.append(float(fixed["actual_post_detection_actions"]))
        rule_actions.append(float(ruled["actual_post_detection_actions"]))
        oracle_actions.append(float(oracle[event_instance_id]["actual_post_detection_actions"]))
        interior.append(2 < k_rule < 16)
        differs.append(k_rule != fixed_k)
    fixed_mean = float(np.mean(fixed_actions)) if fixed_actions else 0.0
    rule_mean = float(np.mean(rule_actions)) if rule_actions else 0.0
    oracle_mean = float(np.mean(oracle_actions)) if oracle_actions else 0.0
    gap = fixed_mean - oracle_mean
    recovered = 0.0 if gap <= 0 else (fixed_mean - rule_mean) / gap
    return {
        "n": len(fixed_actions),
        "rule_interior_fraction": float(np.mean(interior)) if interior else 0.0,
        "rule_differs_from_strongest_fixed_fraction": float(np.mean(differs)) if differs else 0.0,
        "fixed_mean_post_actions": fixed_mean,
        "rule_mean_post_actions": rule_mean,
        "oracle_mean_post_actions": oracle_mean,
        "oracle_efficiency_gap_recovered": float(recovered),
        "source": "calibration_only; evaluation oracle not loaded",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", action="store_true")
    args = parser.parse_args()
    rows = valid_paired_rows()
    baselines = json.loads((ARTIFACT_ROOT / "frozen_rule/baselines.json").read_text())
    strongest = baselines["strongest_fixed_baseline"]["method"]
    definitions = {
        "h2": ("EVENT_ALIGNED_CACHED", "FRESH_MATCHED_AT_RULE_K"),
        "h3": ("EVENT_ALIGNED_CACHED", "IMMEDIATE_FRESH"),
        "h4": ("EVENT_ALIGNED_CACHED", strongest),
    }
    grouped = {
        name: grouped_comparisons(rows, method, baseline)
        for name, (method, baseline) in definitions.items()
    }
    agreements = {name: agreement(value) for name, value in grouped.items()}
    raw_pass = {}
    for name in ("h2", "h3"):
        overall = grouped[name]["overall"]
        raw_pass[name] = bool(
            safety_noninferior(overall)
            and efficient(overall)[0]
            and agreements[name]["both_tasks_agree"]
            and agreements[name]["both_severities_per_task_agree"]
            and agreements[name]["at_least_two_of_three_actor_seeds_agree"]
        )
    gap = calibration_gap_recovery(strongest)
    raw_pass["h4"] = bool(
        safety_noninferior(grouped["h4"]["overall"])
        and gap["rule_interior_fraction"] >= 0.20
        and gap["rule_differs_from_strongest_fixed_fraction"] >= 0.20
        and gap["oracle_efficiency_gap_recovered"] >= 0.40
    )
    calibration = json.loads((ARTIFACT_ROOT / "calibration_atlas/summary.json").read_text())
    isolation = json.loads((ARTIFACT_ROOT / "branch_isolation/summary.json").read_text())
    event_summary = json.loads((ARTIFACT_ROOT / "actor_events/summary.json").read_text())
    qualification = json.loads(
        (ARTIFACT_ROOT / "perturbation_qualification/frozen_parameters.json").read_text()
    )
    confirm = json.loads((ARTIFACT_ROOT / "confirmatory_evaluation/summary.json").read_text())
    event_pass = all(
        event_summary.get("availability_gate", {}).get("calibration", {}).get(task, False)
        for task in TASKS
    )
    upstream_pass = bool(
        isolation["status"] == "PASS"
        and event_pass
        and qualification["status"] == "PASS"
        and confirm["status"] == "PASS"
        and not calibration["diagnostic_only"]
    )
    h1_raw = bool(calibration["h1_raw_pass"])
    family_pass_count = sum(
        qualification["tasks"][task]["status"] == "PASS" for task in TASKS
    )
    if upstream_pass:
        h1 = "PASS" if h1_raw else "FAIL"
        h2 = "PASS" if raw_pass["h2"] else "FAIL"
        h3 = "PASS" if raw_pass["h3"] else "FAIL"
        h4 = "PASS" if raw_pass["h4"] else "FAIL"
    else:
        h1 = h2 = h3 = h4 = "INCONCLUSIVE"

    if isolation["status"] != "PASS":
        overall = "BLOCKED_BY_BRANCH_ISOLATION"
    elif not event_pass:
        overall = "BLOCKED_BY_EVENT_CONSTRUCTION"
    elif family_pass_count == 1:
        overall = "SINGLE_FAMILY_SIGNAL_ONLY"
    elif qualification["status"] != "PASS":
        overall = "BLOCKED_BY_PERTURBATION_QUALIFICATION"
    elif confirm["status"] != "PASS":
        overall = "BLOCKED_BY_MINIMUM_DATA"
    elif h1 == "FAIL":
        overall = "STOP_ALL_PREFIX_TIMING"
    elif h2 == "FAIL":
        overall = "PROCEED_REPLAN_TIMING_ONLY" if h3 == h4 == "PASS" else "STOP_ALL_PREFIX_TIMING"
    elif h1 == h2 == h3 == h4 == "PASS":
        overall = "PROCEED_TO_LEARNED_REPLANABILITY"
    else:
        overall = "LOCAL_MECHANISM_ONLY_NOT_DEPLOYABLE"
    if family_pass_count == 1:
        cached_claim = "SINGLE_FAMILY_ONLY"
    elif h2 == "PASS":
        cached_claim = "SUPPORTED_CONDITIONALLY"
    elif h2 == "FAIL":
        cached_claim = "NO_CACHED_PREFIX_CONTENT_VALUE"
    else:
        cached_claim = "RETIRED"
    statistics = {
        "schema_version": 1,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "independent_unit": "init_state_rollout_event",
        "bootstrap_cluster_contract": {
            "overall": ["task", "init_state/source_rollout"],
            "task": ["task", "init_state/source_rollout"],
            "severity": ["task", "parameter_id", "init_state/source_rollout"],
            "actor_seed": [
                "task",
                "parameter_id",
                "init_state/source_rollout",
                "actor_seed",
            ],
            "actor_seed_is_repeated_measurement": True,
            "prefix_rows_are_not_independent": True,
        },
        "valid_events": len(rows),
        "comparisons": grouped,
        "agreement": agreements,
        "efficiency_fields": list(EFFICIENCY_FIELDS),
        "raw_pass": raw_pass,
        "h1_raw_pass": h1_raw,
        "h4_calibration_oracle_gap": gap,
        "diagnostic_error_signal": diagnostic_error_signal(),
        "macro_task_average_descriptive_only": True,
        "evaluation_oracle_loaded": False,
        "diagnostic_only_global": DIAGNOSTIC_ONLY_GLOBAL,
        "formal_positive_evidence_allowed": FORMAL_POSITIVE_EVIDENCE_ALLOWED,
        "actor_seed_is_repeated_measurement": True,
    }
    decision = {
        "schema_version": 1,
        "stage1b_universal_hypothesis": "KILLED_IMMUTABLE",
        "stage2c_status": "BLOCKED_UPSTREAM_IMMUTABLE",
        "branch_isolation": "PASS" if isolation["status"] == "PASS" else "BLOCKED",
        "event_construction": "PASS" if event_pass else "BLOCKED",
        "target_shift_qualification": qualification["tasks"][TASKS[0]]["status"],
        "path_obstacle_qualification": qualification["tasks"][TASKS[1]]["status"],
        "oracle_mechanism": calibration["oracle_mechanism_decision"],
        "h1_observed_safe_window": h1,
        "h2_cached_prefix_content": h2,
        "h3_event_aligned_handoff": h3,
        "h4_nontrivial_selection": h4,
        "cached_prefix_claim": cached_claim,
        "overall": overall,
        "forced_downstream_execution": True,
        "downstream_evidence_after_failed_gate": "DIAGNOSTIC_ONLY",
        "diagnostic_only_global": DIAGNOSTIC_ONLY_GLOBAL,
        "formal_positive_evidence_allowed": FORMAL_POSITIVE_EVIDENCE_ALLOWED,
        "accepted": False,
        "novelty": "N2_ORACLE_PROTOCOL_BOUNDARY_ONLY",
    }
    output = ARTIFACT_ROOT / "statistics"
    atomic_write_json(output / "statistics.json", statistics)
    atomic_write_json(output / "primary_decision.json", decision)
    atomic_write_json(ARTIFACT_ROOT / "decision.json", decision)
    manifest = {
        "schema_version": 1,
        "primary_locked_before_evaluation_oracle": True,
        "files": {
            "confirmatory_paired_rows": sha256_file(
                ARTIFACT_ROOT / "confirmatory_evaluation/paired_rows.jsonl"
            ),
            "frozen_rule_manifest": sha256_file(ARTIFACT_ROOT / "frozen_rule/manifest.json"),
            "statistics": sha256_file(output / "statistics.json"),
            "primary_decision": sha256_file(output / "primary_decision.json"),
        },
    }
    atomic_write_json(output / "primary_manifest.json", manifest)
    if MIRROR_EXPERIMENT_OUTPUTS:
        mirror = EXPERIMENT_ROOT / "statistics"
        mirror.mkdir(parents=True, exist_ok=True)
        for name in ("statistics.json", "primary_decision.json", "primary_manifest.json"):
            (mirror / name).write_bytes((output / name).read_bytes())
        atomic_write_json(EXPERIMENT_ROOT / "decision.json", decision)
    print(json.dumps({"overall": overall, "h1": h1, "h2": h2, "h3": h3, "h4": h4}, sort_keys=True))


if __name__ == "__main__":
    main()
