"""Pure Stage-2F/S1 grid and operator-family statistics."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

OPERATORS = ("fresh_h4", "fresh_h16", "hold_1+fresh_h4", "rollback_1+fresh_h16")
FAMILY_A = ("fresh_h4", "hold_1+fresh_h4")
FAMILY_B = ("fresh_h16", "rollback_1+fresh_h16")
FAMILY_FULL = FAMILY_A + FAMILY_B
PHASE1_PREFIXES = (2, 4, 8, 12, 16)
PHASE2_PREFIXES = (2, 4, 6, 8, 10, 12, 14, 16)
EXPECTED_RECOVERY_SEEDS = ("7", "17", "29")
PHASE2_BOOTSTRAP_REPLICATES = 10000
PHASE2_BOOTSTRAP_SEED = 216214
REQUIRED_FIELDS = (
    "event_instance_id", "task", "init_state_id", "split", "generator_actor_seed",
    "recovery_actor_seed", "operator", "prefix_k", "tail_horizon", "action_budget",
    "policy_call_cap", "safe_success", "pid", "env_hash", "chunk_hash", "status",
)
DUPLICATE_KEY_FIELDS = (
    "event_instance_id", "task", "init_state_id", "split", "generator_actor_seed",
    "recovery_actor_seed", "operator", "prefix_k", "tail_horizon", "action_budget",
    "policy_call_cap",
)
ERROR_STATUSES = {"error", "failed", "failure", "exception", "timeout", "timed_out", "blocked", "invalid", "crashed"}
NONTERMINAL_STATUSES = {"running", "pending", "started", "in_progress", "incomplete"}


def _text(value: Any) -> str:
    return str(value)


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer")
    return result


def _float(value: Any, field: str = "safe_success") -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{field} must be finite and in [0,1]")
    return result


def _seed(value: Any) -> str:
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def _sort(value: Any) -> tuple[int, Any]:
    try:
        return (0, int(str(value)))
    except ValueError:
        return (1, str(value))


def _error_status(value: Any) -> bool:
    status = str(value).strip().lower()
    return status in ERROR_STATUSES or status.startswith(("blocked", "error", "failed", "failure", "exception"))


def _budget(row: Mapping[str, Any]) -> dict[str, int]:
    return {field: _int(row[field], field) for field in ("tail_horizon", "action_budget", "policy_call_cap")}


def _budget_key(budget: Mapping[str, Any]) -> str:
    return "/".join(f"{field}={int(budget[field])}" for field in ("tail_horizon", "action_budget", "policy_call_cap"))


def _event_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (_text(row["task"]), _text(row["init_state_id"]), _text(row["event_instance_id"]))


def _validate(rows: Sequence[Mapping[str, Any]], split: str | None) -> dict[str, Any]:
    selected = [row for row in rows if split is None or _text(row.get("split")) == split]
    reasons: list[str] = []
    usable: list[Mapping[str, Any]] = []
    errors: list[Mapping[str, Any]] = []
    seen: dict[tuple[str, ...], int] = {}
    for index, row in enumerate(selected):
        if not isinstance(row, Mapping):
            reasons.append(f"row {index} is not an object")
            continue
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            missing_provenance = [field for field in missing if field in ("pid", "env_hash", "chunk_hash")]
            if missing_provenance:
                reasons.append(f"row {index} missing provenance: {','.join(missing_provenance)}")
            else:
                reasons.append(f"row {index} missing fields: {','.join(missing)}")
            continue
        key = tuple(_text(row[field]) for field in DUPLICATE_KEY_FIELDS)
        if key in seen:
            raise ValueError(f"exact duplicate branch key rejected: {key!r}")
        seen[key] = index
        missing_provenance = [field for field in ("pid", "env_hash", "chunk_hash") if row.get(field) is None or not str(row.get(field)).strip()]
        if missing_provenance:
            reasons.append(f"row {index} missing provenance: {','.join(missing_provenance)}")
            continue
        status = str(row["status"]).strip().lower()
        if status in NONTERMINAL_STATUSES:
            reasons.append(f"row {index} nonterminal status: {status}")
            continue
        try:
            _float(row["safe_success"])
            _int(row["prefix_k"], "prefix_k")
            _budget(row)
        except (TypeError, ValueError) as exc:
            reasons.append(f"row {index} invalid value: {exc}")
            continue
        (errors if _error_status(status) else usable).append(row)
    return {"selected": selected, "usable": usable, "errors": errors, "reasons": reasons}


def _blocked(analysis: str, split: str | None, rows: Sequence[Any], validation: Mapping[str, Any], reasons: Iterable[str] = ()) -> dict[str, Any]:
    all_reasons = list(validation.get("reasons", ())) + list(reasons)
    return {
        "schema_version": 1, "analysis": analysis, "status": "BLOCKED", "blocked": True, "split": split,
        "row_count": len(rows), "usable_row_count": len(validation.get("usable", ())),
        "excluded_error_row_count": len(validation.get("errors", ())),
        "blocking_reasons": list(dict.fromkeys(str(reason) for reason in all_reasons))[:100] or ["no usable rows"],
        "selection_source": "calibration_only" if analysis == "grid" else None,
    }


def _events(rows: Sequence[Mapping[str, Any]], operators: Sequence[str] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    allowed = set(operators) if operators is not None else None
    values: dict[tuple[tuple[str, str, str], int, str, str], list[float]] = defaultdict(list)
    seeds: set[str] = set()
    for row in rows:
        op = _text(row["operator"])
        if allowed is not None and op not in allowed:
            continue
        event = _event_key(row)
        prefix = _int(row["prefix_k"], "prefix_k")
        actor_seed = _seed(row["recovery_actor_seed"])
        seeds.add(actor_seed)
        values[(event, prefix, op, actor_seed)].append(_float(row["safe_success"]))
    material: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (event, prefix, op, actor_seed), observations in values.items():
        record = material.setdefault(event, {"task": event[0], "init_state_id": event[1], "event_instance_id": event[2], "values": defaultdict(lambda: defaultdict(dict))})
        record["values"][prefix][op][actor_seed] = float(np.mean(observations))
    result = []
    for record in material.values():
        record["values"] = {int(prefix): {str(op): dict(actor_values) for op, actor_values in ops.items()} for prefix, ops in record["values"].items()}
        result.append(record)
    result.sort(key=lambda row: (_sort(row["task"]), _sort(row["init_state_id"]), _sort(row["event_instance_id"])))
    return result, sorted(seeds, key=_sort)


def _mean(values: Iterable[float]) -> float | None:
    material = list(values)
    return float(np.mean(material)) if material else None


def _cluster_mean(values: Mapping[tuple[str, str, str], float]) -> tuple[float | None, int, dict[tuple[str, str], float]]:
    clusters: dict[tuple[str, str], list[float]] = defaultdict(list)
    for key, value in values.items():
        clusters[(key[0], key[1])].append(float(value))
    means = {key: float(np.mean(items)) for key, items in clusters.items()}
    return _mean(means.values()), len(means), means


def _bootstrap_success_means(
    cluster_rows: Sequence[Mapping[str, Any]],
    operators: Sequence[str],
    *,
    replicates: int = PHASE2_BOOTSTRAP_REPLICATES,
    seed: int = PHASE2_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap success means over (task, init_state_id) clusters.

    Each cluster row already contains the prescribed event/prefix/actor
    aggregation.  A draw resamples whole cluster rows with replacement and
    takes the mean of the supported cluster values.  The same resampled
    cluster indices are used for the oracle and every arm so their CIs share
    the prescribed cluster unit.
    """
    metric_names = ("oracle",) + tuple(operators)
    point_values: dict[str, list[float]] = {"oracle": [], **{op: [] for op in operators}}
    for row in cluster_rows:
        oracle = row.get("oracle")
        if oracle is not None:
            point_values["oracle"].append(float(oracle))
        arms = row.get("arm_means", {})
        for op in operators:
            value = arms.get(op) if isinstance(arms, Mapping) else None
            if value is not None:
                point_values[op].append(float(value))
    point = {name: _mean(point_values[name]) for name in metric_names}
    cluster_count = len(cluster_rows)
    draws: dict[str, list[float]] = {name: [] for name in metric_names}
    if cluster_count and replicates > 0:
        rng = np.random.default_rng(seed)
        for _ in range(replicates):
            indices = rng.integers(0, cluster_count, size=cluster_count)
            for name in metric_names:
                values: list[float] = []
                for index in indices:
                    row = cluster_rows[int(index)]
                    if name == "oracle":
                        value = row.get("oracle")
                    else:
                        arms = row.get("arm_means", {})
                        value = arms.get(name) if isinstance(arms, Mapping) else None
                    if value is not None:
                        values.append(float(value))
                if values:
                    draws[name].append(float(np.mean(values)))

    def metric_result(name: str) -> dict[str, Any]:
        values = draws[name]
        return {
            "estimate": point[name],
            "ci95": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ] if values else [None, None],
            "defined_cluster_count": len(point_values[name]),
            "defined_draw_count": len(values),
            "undefined_draw_count": int(replicates) - len(values),
        }

    arm_results = {op: metric_result(op) for op in operators}
    return {
        "oracle": metric_result("oracle"),
        "arm_means": arm_results,
        "individual_arm_means": {
            op: dict(value) for op, value in arm_results.items()
        },
        "cluster_count": cluster_count,
        "cluster_unit": ["task", "init_state_id"],
        "replicates": int(replicates),
        "seed": int(seed),
        "point_estimand": (
            "event/prefix means followed by equal (task,init_state_id) "
            "cluster weighting"
        ),
    }


def _prefix_arm_means(event: Mapping[str, Any], prefixes: Sequence[int], operators: Sequence[str]) -> tuple[dict[int, dict[str, float]], dict[int, float]]:
    arms: dict[int, dict[str, float]] = {}
    oracles: dict[int, float] = {}
    for prefix in prefixes:
        per_op = {}
        for op in operators:
            actor_values = event.get("values", {}).get(int(prefix), {}).get(op, {})
            if actor_values:
                per_op[op] = float(np.mean(list(actor_values.values())))
        if per_op:
            arms[int(prefix)] = per_op
            oracles[int(prefix)] = max(per_op.values())
    return arms, oracles


def _grid_metrics(events: Sequence[Mapping[str, Any]], prefixes: Sequence[int], operators: Sequence[str], actor_count: int, *, include_bootstrap: bool = False) -> dict[str, Any]:
    event_summaries = []
    event_prefix_summaries = []
    prefix_oracles: dict[int, dict[tuple[str, str, str], float]] = defaultdict(dict)
    prefix_arms = {op: {int(prefix): {} for prefix in prefixes} for op in operators}
    complete_cells = 0
    for event in events:
        key = (_text(event["task"]), _text(event["init_state_id"]), _text(event["event_instance_id"]))
        arms, oracles = _prefix_arm_means(event, prefixes, operators)
        complete_for_event = 0
        for prefix in prefixes:
            pdata = event.get("values", {}).get(int(prefix), {})
            prefix_arm_means = dict(arms.get(int(prefix), {}))
            event_prefix_summaries.append({
                "task": event["task"],
                "init_state_id": event["init_state_id"],
                "event_instance_id": event["event_instance_id"],
                "prefix_k": int(prefix),
                "arm_means": prefix_arm_means,
                "individual_arm_means": dict(prefix_arm_means),
                "oracle": oracles.get(int(prefix)),
                "oracle_best": oracles.get(int(prefix)),
                "complete": all(op in pdata and len(pdata[op]) >= actor_count for op in operators),
            })
            complete = all(op in pdata and len(pdata[op]) >= actor_count for op in operators)
            complete_cells += int(complete)
            complete_for_event += int(complete)
            if prefix in oracles:
                prefix_oracles[int(prefix)][key] = oracles[prefix]
            for op in operators:
                if op in arms.get(prefix, {}):
                    prefix_arms[op][int(prefix)][key] = arms[prefix][op]
        event_summaries.append({
            "task": event["task"], "init_state_id": event["init_state_id"], "event_instance_id": event["event_instance_id"],
            "oracle": _mean(oracles.values()),
            "arm_means": {op: _mean(arms[prefix][op] for prefix in prefixes if op in arms.get(prefix, {})) for op in operators},
            "complete_prefix_cells": complete_for_event,
        })
    event_oracles = {(_text(row["task"]), _text(row["init_state_id"]), _text(row["event_instance_id"])): row["oracle"] for row in event_summaries if row["oracle"] is not None}
    oracle, cluster_count, clusters = _cluster_mean(event_oracles)
    arm_means: dict[str, float | None] = {}
    cluster_arms: dict[str, dict[tuple[str, str], float]] = {}
    for op in operators:
        values = {(_text(row["task"]), _text(row["init_state_id"]), _text(row["event_instance_id"])): row["arm_means"][op] for row in event_summaries if row["arm_means"].get(op) is not None}
        arm_means[op], _, cluster_arms[op] = _cluster_mean(values)
    weakest = min(arm_means.values()) if all(value is not None for value in arm_means.values()) else None
    per_prefix = []
    for prefix in prefixes:
        prefix_oracle, prefix_clusters, prefix_cluster_values = _cluster_mean(prefix_oracles[int(prefix)])
        prefix_arm = {op: _cluster_mean(prefix_arms[op][int(prefix)])[0] for op in operators}
        prefix_weakest = min(prefix_arm.values()) if all(value is not None for value in prefix_arm.values()) else None
        prefix_weakest_name = next((op for op in operators if prefix_arm.get(op) == prefix_weakest), None)
        per_prefix.append({
            "prefix_k": int(prefix), "oracle": prefix_oracle, "oracle_best": prefix_oracle,
            "arm_means": prefix_arm, "individual_arm_means": dict(prefix_arm), "weakest_arm": prefix_weakest,
            "weakest_operator": prefix_weakest_name,
            "oracle_minus_weakest": float(prefix_oracle - prefix_weakest) if prefix_oracle is not None and prefix_weakest is not None else None,
            "cluster_count": prefix_clusters,
            "cluster_values": {f"{task}/{init_state}": value for (task, init_state), value in prefix_cluster_values.items()},
        })
    total_cells = len(events) * len(prefixes)
    completeness = {
        "status": "COMPLETE" if complete_cells == total_cells else "INCOMPLETE", "complete": complete_cells == total_cells,
        "complete_event_prefix_cells": complete_cells, "expected_event_prefix_cells": total_cells,
        "fraction": float(complete_cells / total_cells) if total_cells else None,
        "expected_operator_count": len(operators), "expected_actor_count": actor_count, "prefixes": list(prefixes),
    }
    weakest_name = next((op for op in operators if arm_means.get(op) == weakest), None)
    gap = float(oracle - weakest) if oracle is not None and weakest is not None else None
    result = {
        "oracle": oracle, "oracle_best": oracle, "arm_means": arm_means, "individual_arm_means": dict(arm_means),
        "weakest_arm": weakest, "weakest_operator": weakest_name, "gap": gap, "oracle_minus_weakest": gap, "cluster_count": cluster_count,
        "event_count": len(events), "cluster_oracles": {f"{task}/{init_state}": value for (task, init_state), value in clusters.items()},
        "cluster_arm_means": {op: {f"{task}/{init_state}": value for (task, init_state), value in values.items()} for op, values in cluster_arms.items()},
        "event_summaries": event_summaries, "event_prefix_summaries": event_prefix_summaries,
        "per_event_prefix": event_prefix_summaries, "per_prefix": per_prefix,
        "completeness": completeness,
        "estimand": "cluster mean after event/prefix means; prefix rows are not independent clusters",
    }
    if include_bootstrap:
        cluster_keys = set(clusters)
        for values in cluster_arms.values():
            cluster_keys.update(values)
        cluster_rows = [
            {
                "oracle": clusters.get(key),
                "arm_means": {
                    op: cluster_arms.get(op, {}).get(key) for op in operators
                },
            }
            for key in sorted(cluster_keys)
        ]
        result["bootstrap"] = _bootstrap_success_means(cluster_rows, operators)
        result["bootstrap_replicates"] = PHASE2_BOOTSTRAP_REPLICATES
        result["bootstrap_seed"] = PHASE2_BOOTSTRAP_SEED
    return result


def _reference_summary(
    rows: Sequence[Mapping[str, Any]],
    operators: Sequence[str],
    prefixes: Sequence[int],
    *,
    include_bootstrap: bool = False,
) -> dict[str, Any]:
    if not operators:
        result = {"operators": [], "row_count": 0, "by_task": {}, "selection_eligible": False}
        if include_bootstrap:
            result.update({
                "bootstrap_replicates": PHASE2_BOOTSTRAP_REPLICATES,
                "bootstrap_seed": PHASE2_BOOTSTRAP_SEED,
            })
        return result
    events, _ = _events(rows, operators=operators)
    by_task = {}
    for task in sorted({_text(event["task"]) for event in events}, key=_sort):
        task_events = [event for event in events if _text(event["task"]) == task]
        actor_count = max([len(actor_values) for event in task_events for ops in event["values"].values() for actor_values in ops.values()] or [0])
        by_task[task] = _grid_metrics(
            task_events, prefixes, operators, actor_count,
            include_bootstrap=include_bootstrap,
        )
    result = {
        "operators": list(operators),
        "row_count": sum(_text(row["operator"]) in operators for row in rows),
        "by_task": by_task,
        "selection_eligible": False,
    }
    if include_bootstrap:
        result.update({
            "bootstrap_replicates": PHASE2_BOOTSTRAP_REPLICATES,
            "bootstrap_seed": PHASE2_BOOTSTRAP_SEED,
        })
    return result


def summarize_grid(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize Phase-1 calibration rows; evaluation input is hard rejected."""
    material = list(rows)
    bad_splits = sorted({_text(row.get("split")) for row in material if isinstance(row, Mapping) and _text(row.get("split")) != "calibration"})
    if bad_splits:
        raise ValueError("summarize_grid accepts calibration rows only; rejected split(s): " + ",".join(bad_splits))
    validation = _validate(material, "calibration")
    if validation["reasons"]:
        return _blocked("grid", "calibration", material, validation)
    usable = list(validation["usable"])
    fixed = [row for row in usable if _text(row["operator"]) in OPERATORS]
    if not fixed:
        return _blocked("grid", "calibration", material, validation, ["no fixed operator rows"])
    _, seeds = _events(fixed, operators=OPERATORS)
    if tuple(seeds) != EXPECTED_RECOVERY_SEEDS:
        return _blocked("grid", "calibration", material, validation, [f"expected recovery actor seeds {list(EXPECTED_RECOVERY_SEEDS)}; observed {seeds}"])
    grouped: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in fixed:
        budget = _budget(row)
        grouped[tuple(budget[field] for field in ("tail_horizon", "action_budget", "policy_call_cap"))].append(row)
    tasks = sorted({_text(row["task"]) for row in fixed}, key=_sort)
    budgets: dict[str, Any] = {}
    blockers = []
    for key, budget_rows in sorted(grouped.items()):
        budget = dict(zip(("tail_horizon", "action_budget", "policy_call_cap"), key))
        budget_events, _ = _events(budget_rows, operators=OPERATORS)
        by_task = {}
        for task in tasks:
            metrics = _grid_metrics([event for event in budget_events if event["task"] == task], PHASE1_PREFIXES, OPERATORS, 3)
            by_task[task] = metrics
            if not metrics["completeness"]["complete"]:
                blockers.append(f"incomplete Phase-1 matrix for {task} and {_budget_key(budget)}")
        budgets[_budget_key(budget)] = {"budget": budget, "by_task": by_task, "task_metrics": by_task, "selection_eligible": True}
    if blockers:
        return _blocked("grid", "calibration", material, validation, blockers)
    references = _reference_summary(usable, sorted({_text(row["operator"]) for row in usable if _text(row["operator"]) not in OPERATORS}, key=_sort), PHASE1_PREFIXES)
    grid = [budgets[key] for key in sorted(budgets)]
    return {
        "schema_version": 1, "analysis": "grid", "status": "COMPLETE", "blocked": False, "split": "calibration",
        "selection_source": "calibration_only", "row_count": len(material), "usable_row_count": len(usable),
        "excluded_error_row_count": len(validation["errors"]), "tasks": tasks, "operators": list(OPERATORS),
        "recovery_actor_seeds": seeds, "prefixes": list(PHASE1_PREFIXES), "budgets": budgets, "grid": grid,
        "references": references, "completeness": {"status": "COMPLETE", "cluster_unit": ["task", "init_state_id"], "actor_aggregation": "mean over three recovery actor seeds", "prefix_aggregation": "mean within event before cluster mean"},
    }


def select_budget(grid_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply K2: both tasks oracle in [.25,.85] and gap >= .15."""
    # A partially observed planned grid may still have useful descriptive
    # metrics, but it can never produce an admission receipt.  Keep this
    # guard here as well as in the consolidator so callers that pass an
    # annotated summary cannot accidentally turn a shortfall into SELECTED.
    sample = grid_summary.get("sample")
    if not isinstance(sample, Mapping):
        sample = {}
    sample_complete = grid_summary.get("sample_complete")
    if sample_complete is None:
        sample_complete = grid_summary.get("planned_sample_complete")
    if sample_complete is None:
        sample_complete = sample.get("complete")
    excluded_count = grid_summary.get(
        "structurally_excluded_event_count",
        sample.get("structurally_excluded_event_count", 0),
    )
    try:
        excluded_count = int(excluded_count or 0)
    except (TypeError, ValueError):
        excluded_count = 1
    blocking_reasons = []
    if excluded_count:
        blocking_reasons.append(
            f"structurally excluded events are not selection observations: {excluded_count}"
        )
    if sample_complete is False:
        observed = grid_summary.get("observed_events_by_task", sample.get("observed_events_by_task"))
        planned = grid_summary.get("planned_events", sample.get("planned_events", 20))
        shortfall = grid_summary.get("shortfall", sample.get("shortfall"))
        blocking_reasons.append(
            "planned calibration sample incomplete: "
            f"observed={observed!r}, planned={planned!r}, shortfall={shortfall!r}"
        )
    if blocking_reasons:
        return {
            "schema_version": 1,
            "status": "BLOCKED",
            "selected_budget": None,
            "selected": None,
            "selection_source": "calibration_only",
            "blocking_reasons": blocking_reasons,
            "qualifying_budgets": [],
            "ranked_candidates": [],
            "selection_eligible": False,
        }
    if grid_summary.get("blocked") or (grid_summary.get("status") is not None and grid_summary.get("status") != "COMPLETE"):
        return {"schema_version": 1, "status": "BLOCKED", "selected_budget": None, "selected": None, "selection_source": "calibration_only", "blocking_reasons": list(grid_summary.get("blocking_reasons", ())) or ["grid summary is not complete"], "qualifying_budgets": [], "ranked_candidates": [], "selection_eligible": False}
    records = grid_summary.get("grid")
    if records is None:
        raw = grid_summary.get("budgets", {})
        records = list(raw.values()) if isinstance(raw, Mapping) else []
    elif isinstance(records, Mapping):
        records = list(records.values())
    tasks = sorted(dict.fromkeys(grid_summary.get("tasks", ())), key=_sort)
    if len(tasks) < 2:
        tasks = sorted({task for record in records for task in (record.get("by_task") or {}).keys()}, key=_sort)
    if len(tasks) != 2:
        return {"schema_version": 1, "status": "BLOCKED", "selected_budget": None, "selected": None, "selection_source": "calibration_only", "blocking_reasons": [f"K2 requires two tasks; observed {len(tasks)}"], "qualifying_budgets": [], "ranked_candidates": []}
    candidates = []
    for record in records:
        by_task = record.get("by_task") or record.get("task_metrics") or {}
        metrics = [by_task.get(task) for task in tasks]
        if any(metric is None for metric in metrics):
            continue
        oracles = [metric.get("oracle_best", metric.get("oracle")) for metric in metrics]
        gaps = [metric.get("oracle_minus_weakest", metric.get("gap")) for metric in metrics]
        if any(value is None for value in oracles + gaps):
            continue
        if not all(0.25 <= float(value) <= 0.85 for value in oracles) or not all(float(value) >= .15 for value in gaps):
            continue
        budget = dict(record.get("budget", {}))
        if not {"tail_horizon", "action_budget"}.issubset(budget):
            continue
        candidate = {"budget": budget, "by_task": {task: by_task[task] for task in tasks}, "oracle_by_task": {task: float(metric.get("oracle_best", metric.get("oracle"))) for task, metric in zip(tasks, metrics)}, "gap_by_task": {task: float(metric.get("oracle_minus_weakest", metric.get("gap"))) for task, metric in zip(tasks, metrics)}, "min_oracle": float(min(oracles)), "min_gap": float(min(gaps)), "qualifies": True}
        candidates.append(candidate)
    candidates.sort(key=lambda item: (-item["min_oracle"], -item["min_gap"], int(item["budget"]["action_budget"]), int(item["budget"]["tail_horizon"]), int(item["budget"].get("policy_call_cap", 0))))
    selected = candidates[0]["budget"] if candidates else None
    return {"schema_version": 1, "status": "SELECTED" if selected is not None else "NO_QUALIFYING_BUDGET", "selected_budget": selected, "selected": selected, "selection_source": "calibration_only", "task_order": tasks, "qualification_rule": {"oracle_best_interval": [.25, .85], "minimum_oracle_minus_weakest": .15, "requires_both_tasks": True}, "qualifying_budgets": candidates, "ranked_candidates": candidates, "no_invented_budget": selected is None or selected in [candidate["budget"] for candidate in candidates]}


def _boundary(event: Mapping[str, Any], operators: Sequence[str], seeds: Sequence[str], threshold: float, prefixes: Sequence[int]) -> int | None:
    last = None
    for prefix in prefixes:
        recovered = False
        for op in operators:
            actor_values = event.get("values", {}).get(int(prefix), {}).get(op, {})
            if all(actor_seed in actor_values for actor_seed in seeds) and float(np.mean([actor_values[actor_seed] for actor_seed in seeds])) >= threshold:
                recovered = True
                break
        if recovered:
            last = int(prefix)
    return last


def _boundary_rows(events: Sequence[Mapping[str, Any]], seeds: Sequence[str]) -> list[dict[str, Any]]:
    result = []
    for event in events:
        a = _boundary(event, FAMILY_A, seeds, 2 / 3, PHASE2_PREFIXES)
        b = _boundary(event, FAMILY_B, seeds, 2 / 3, PHASE2_PREFIXES)
        full = _boundary(event, FAMILY_FULL, seeds, 2 / 3, PHASE2_PREFIXES)
        result.append({"task": event["task"], "init_state_id": event["init_state_id"], "event_instance_id": event["event_instance_id"], "boundary_A": a, "boundary_B": b, "boundary_full": full, "k_star_A": a, "k_star_B": b, "k_star_full": full, "defined_A": a is not None, "defined_B": b is not None, "defined_full": full is not None})
    return result


def _rank(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    result = np.empty(len(arr), dtype=float)
    ordered = arr[order]
    start = 0
    while start < len(arr):
        end = start + 1
        while end < len(arr) and ordered[end] == ordered[start]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return result


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y) or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    xr, yr = _rank(x), _rank(y)
    xc, yc = xr - xr.mean(), yr - yr.mean()
    denominator = float(np.sqrt(np.dot(xc, xc) * np.dot(yc, yc)))
    return None if denominator == 0 else float(np.dot(xc, yc) / denominator)


def _bootstrap_crossing(rows: Sequence[Mapping[str, Any]], replicates: int, seed: int, task: str | None = None, include_draws: bool = False) -> dict[str, Any]:
    selected = [row for row in rows if row.get("boundary_A") is not None and row.get("boundary_B") is not None and (task is None or _text(row["task"]) == task)]
    if not selected:
        result = {"estimate": None, "ci95": [None, None], "replicates": replicates, "seed": seed, "cluster_count": 0, "task_cluster_count": {}}
        if include_draws: result["_draws"] = np.asarray([], dtype=float)
        return result
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in selected:
        grouped[_text(row["task"])][_text(row["init_state_id"])].append(row)
    # Keep all event rows inside a sampled (task, init_state_id) cluster.  The
    # point estimand is the event mean; a cluster contributes all of its event
    # rows when sampled, so the bootstrap CI estimates the same event-level
    # quantity while respecting cluster dependence.
    clusters = {row_task: dict(by_init) for row_task, by_init in grouped.items()}
    def value(sample: Mapping[str, Sequence[Sequence[Mapping[str, Any]]]]) -> float:
        sampled_rows = [row for cluster_rows in sample.values() for cluster in cluster_rows for row in cluster]
        if not sampled_rows:
            return float("nan")
        a = float(np.mean([float(row["crossing_A"]) for row in sampled_rows]))
        b = float(np.mean([float(row["crossing_B"]) for row in sampled_rows]))
        return min(a, b)
    point = value({row_task: list(by_init.values()) for row_task, by_init in clusters.items()})
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = {}
        for row_task, by_init in clusters.items():
            values = list(by_init.values())
            sample[row_task] = [values[int(i)] for i in rng.integers(0, len(values), size=len(values))]
        draws[index] = value(sample)
    result = {"estimate": point, "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))], "replicates": replicates, "seed": seed, "cluster_count": sum(len(items) for items in clusters.values()), "task_cluster_count": {task_name: len(items) for task_name, items in sorted(clusters.items(), key=lambda item: _sort(item[0]))}}
    if include_draws: result["_draws"] = draws
    return result


def _bootstrap_spearman(rows: Sequence[Mapping[str, Any]], replicates: int, seed: int, task: str | None = None) -> dict[str, Any]:
    selected = [row for row in rows if row.get("boundary_A") is not None and row.get("boundary_B") is not None and (task is None or _text(row["task"]) == task)]
    if not selected:
        return {"estimate": None, "ci95": [None, None], "undefined_draw_count": replicates, "replicates": replicates, "seed": seed}
    estimate = _spearman([row["boundary_A"] for row in selected], [row["boundary_B"] for row in selected])
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(_text(row["task"]), _text(row["init_state_id"]))].append(row)
    clusters = list(grouped.values())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        sample = [row for index in rng.integers(0, len(clusters), size=len(clusters)) for row in clusters[int(index)]]
        value = _spearman([row["boundary_A"] for row in sample], [row["boundary_B"] for row in sample])
        if value is not None and math.isfinite(value): draws.append(float(value))
    return {"estimate": estimate, "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))] if draws else [None, None], "undefined_draw_count": replicates - len(draws), "defined_draw_count": len(draws), "replicates": replicates, "seed": seed}


def _crossing_scope(rows: Sequence[Mapping[str, Any]], task: str | None, replicates: int, seed: int, include_bootstrap_draws: bool = False) -> dict[str, Any]:
    selected = [row for row in rows if row.get("boundary_A") is not None and row.get("boundary_B") is not None and (task is None or _text(row["task"]) == task)]
    missing_both = missing_one = missing_a = missing_b = 0
    for row in rows:
        if task is not None and _text(row["task"]) != task: continue
        has_a, has_b = row.get("boundary_A") is not None, row.get("boundary_B") is not None
        if not has_a and not has_b: missing_both += 1
        elif not has_a or not has_b:
            missing_one += 1; missing_a += int(not has_a); missing_b += int(not has_b)
    a = [int(row["boundary_A"] > row["boundary_B"]) for row in selected]
    b = [int(row["boundary_B"] > row["boundary_A"]) for row in selected]
    rate_a, rate_b = _mean(a), _mean(b)
    bootstrap_rows = [{**row, "crossing_A": x, "crossing_B": y} for row, x, y in zip(selected, a, b)]
    bootstrap = _bootstrap_crossing(bootstrap_rows, replicates, seed, task, include_draws=include_bootstrap_draws)
    result = {"both_defined": len(selected), "defined_event_count": len(selected), "missing_both": missing_both, "missing_one": missing_one, "missing_A": missing_a, "missing_B": missing_b, "crossing_A_rate": rate_a, "crossing_B_rate": rate_b, "minority_crossing_rate": min(rate_a, rate_b) if rate_a is not None and rate_b is not None else None, "spearman": _spearman([row["boundary_A"] for row in selected], [row["boundary_B"] for row in selected]), "spearman_bootstrap": _bootstrap_spearman(selected, replicates, seed, task), "bootstrap": bootstrap, "bootstrap_seed": int(seed), "bootstrap_unit": ["task", "init_state_id"], "bootstrap_point_estimand": "event mean after actor aggregation; sampled init clusters retain all event rows", "actor_aggregation_before_boundary": True, "prefix_rows_are_not_bootstrap_units": True}
    if include_bootstrap_draws:
        result["_bootstrap_draws"] = bootstrap.pop("_draws", np.asarray([], dtype=float))
    return result


def _family_success(events: Sequence[Mapping[str, Any]], operators: Sequence[str]) -> dict[str, Any]:
    per_event = []
    for event in events:
        arms, oracles = _prefix_arm_means(event, PHASE2_PREFIXES, operators)
        per_event.append({"task": event["task"], "init_state_id": event["init_state_id"], "event_instance_id": event["event_instance_id"], "oracle": _mean(oracles.values()), "arm_means": {op: _mean(arms[prefix][op] for prefix in PHASE2_PREFIXES if op in arms.get(prefix, {})) for op in operators}, "per_prefix_oracle": [{"prefix_k": int(prefix), "oracle": value} for prefix, value in sorted(oracles.items())]})
    def scope(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        clusters: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in selected:
            clusters[(_text(row["task"]), _text(row["init_state_id"]))].append(row)
        cluster_rows = [{"oracle": _mean(row["oracle"] for row in cluster if row["oracle"] is not None), "arm_means": {op: _mean(row["arm_means"][op] for row in cluster if row["arm_means"].get(op) is not None) for op in operators}} for cluster in clusters.values()]
        arms = {op: _mean(row["arm_means"][op] for row in cluster_rows if row["arm_means"].get(op) is not None) for op in operators}
        oracle = _mean(row["oracle"] for row in cluster_rows if row["oracle"] is not None)
        return {
            "oracle": oracle, "oracle_best": oracle, "arm_means": arms,
            "individual_arm_means": dict(arms), "cluster_count": len(cluster_rows),
            "event_count": len(selected),
            "estimand": "family oracle is the mean over prefixes of the per-prefix max operator after actor aggregation; event means are averaged within (task,init_state_id) clusters with equal cluster weight",
            "bootstrap": _bootstrap_success_means(cluster_rows, operators),
        }
    tasks = sorted({_text(row["task"]) for row in per_event}, key=_sort)
    return {
        "operators": list(operators),
        "by_task": {task: scope([row for row in per_event if row["task"] == task]) for task in tasks},
        "overall": scope(per_event), "per_event": per_event,
        "selection_eligible": False,
        "bootstrap_replicates": PHASE2_BOOTSTRAP_REPLICATES,
        "bootstrap_seed": PHASE2_BOOTSTRAP_SEED,
    }


def _split_half(events: Sequence[Mapping[str, Any]], family: str, operators: Sequence[str], seeds: Sequence[str], replicates: int, seed: int) -> dict[str, Any]:
    partitions = []
    for singleton in seeds:
        pair = [actor_seed for actor_seed in seeds if actor_seed != singleton]
        rows = []
        for event in events:
            rows.append({"task": event["task"], "init_state_id": event["init_state_id"], "event_instance_id": event["event_instance_id"], "boundary_A": _boundary(event, operators, [singleton], 1.0, PHASE2_PREFIXES), "boundary_B": _boundary(event, operators, pair, 1.0, PHASE2_PREFIXES)})
        crossing = _crossing_scope(rows, None, replicates, seed)
        by_task = {task: _crossing_scope(rows, task, replicates, seed, include_bootstrap_draws=True) for task in sorted({_text(row["task"]) for row in rows}, key=_sort)}
        partitions.append({"singleton_seed": singleton, "pair_seeds": pair, "threshold_singleton": "1/1", "threshold_pair": "2/2", "crossing": crossing, "by_task": by_task, "direction_reversed_is_same_partition": True})
    by_task: dict[str, dict[str, Any]] = {}
    bootstrap_by_task: dict[str, list[float]] = defaultdict(list)
    for task in sorted({task for part in partitions for task in part["by_task"]}, key=_sort):
        values = [part["by_task"][task]["minority_crossing_rate"] for part in partitions if part["by_task"].get(task, {}).get("minority_crossing_rate") is not None]
        for part in partitions:
            draws = part["by_task"].get(task, {}).get("_bootstrap_draws", ())
            bootstrap_by_task[task].extend(float(value) for value in draws if math.isfinite(float(value)))
        by_task[task] = {"point_estimates": values, "point_p95": float(np.quantile(values, .95)) if values else None, "bootstrap_null_p95": (float(np.quantile(bootstrap_by_task[task], .95)) if bootstrap_by_task[task] else None), "bootstrap_null_draws": list(bootstrap_by_task[task]), "partition_count": len(values)}
        for part in partitions:
            part["by_task"].get(task, {}).pop("_bootstrap_draws", None)
    all_values = [value for item in by_task.values() for value in item["point_estimates"]]
    pooled_bootstrap = [value for values in bootstrap_by_task.values() for value in values]
    return {"family": family, "partitions": partitions, "by_task": by_task, "pooled_point_estimates": all_values, "pooled_point_p95": float(np.quantile(all_values, .95)) if all_values else None, "pooled_bootstrap_null_p95": (float(np.quantile(pooled_bootstrap, .95)) if pooled_bootstrap else None), "pooled_bootstrap_null_draws": pooled_bootstrap, "bootstrap_seed": int(seed), "method": "unique singleton-vs-complement partitions; singleton uses 1/1 and pair uses 2/2; reverse direction is retained in the same partition and is not a duplicate partition"}


def _matrix_reasons(events: Sequence[Mapping[str, Any]], seeds: Sequence[str], observed_prefixes: Sequence[int]) -> list[str]:
    reasons = []
    if set(observed_prefixes) != set(PHASE2_PREFIXES): reasons.append(f"expected Phase-2 prefixes {list(PHASE2_PREFIXES)}; observed {list(observed_prefixes)}")
    if tuple(seeds) != EXPECTED_RECOVERY_SEEDS: reasons.append(f"expected recovery actor seeds {list(EXPECTED_RECOVERY_SEEDS)}; observed {list(seeds)}")
    for event in events:
        for prefix in PHASE2_PREFIXES:
            pdata = event.get("values", {}).get(prefix, {})
            for op in OPERATORS:
                actor_values = pdata.get(op, {})
                if set(actor_values) != set(EXPECTED_RECOVERY_SEEDS):
                    reasons.append(f"incomplete Phase-2 cell {event['task']}/{event['init_state_id']}/{event['event_instance_id']} k={prefix} op={op}")
    return reasons


def analyze_crossing(rows: Iterable[Mapping[str, Any]], split: str = "evaluation", replicates: int = 10000, seed: int = 216214) -> dict[str, Any]:
    """Compute k* for U_A/U_B/U_full and preregistered crossing statistics."""
    material = list(rows)
    validation = _validate(material, split)
    if validation["reasons"]:
        return _blocked("crossing", split, material, validation)
    usable = list(validation["usable"])
    fixed = [row for row in usable if _text(row["operator"]) in OPERATORS]
    events, seeds = _events(fixed, operators=OPERATORS)
    observed_prefixes = sorted({_int(row["prefix_k"], "prefix_k") for row in fixed})
    reasons = _matrix_reasons(events, seeds, observed_prefixes)
    if not events: reasons.append("no fixed operator rows in requested split")
    tasks_seen = sorted({_text(event["task"]) for event in events})
    if len(tasks_seen) != 2: reasons.append(f"Phase-2 requires two tasks; observed {tasks_seen}")
    if reasons:
        result = _blocked("crossing", split, material, validation, reasons)
        result.update({"boundaries": [], "crossing": {}, "split_half_null": {}, "safe_success_by_family": {}, "references": {}})
        return result
    boundaries = _boundary_rows(events, seeds)
    tasks = sorted({_text(row["task"]) for row in boundaries}, key=_sort)
    by_task = {task: _crossing_scope(boundaries, task, replicates, seed) for task in tasks}
    crossing = {"overall": _crossing_scope(boundaries, None, replicates, seed), "by_task": by_task, "bootstrap_seed": int(seed), "defined_only": True, "missing_events_reported_separately": True}
    half_a = _split_half(events, "A", FAMILY_A, seeds, replicates, seed)
    half_b = _split_half(events, "B", FAMILY_B, seeds, replicates, seed)
    null_by_task = {}
    for task in sorted(set(half_a["by_task"]) | set(half_b["by_task"]), key=_sort):
        a, b = half_a["by_task"].get(task, {}), half_b["by_task"].get(task, {})
        values = list(a.get("point_estimates", [])) + list(b.get("point_estimates", []))
        point_p95 = float(np.quantile(values, .95)) if values else None
        bootstrap_draws = list(a.get("bootstrap_null_draws", ())) + list(b.get("bootstrap_null_draws", ()))
        bootstrap_p95 = float(np.quantile(bootstrap_draws, .95)) if bootstrap_draws else None
        null_by_task[task] = {"family_A_point_p95": a.get("point_p95"), "family_B_point_p95": b.get("point_p95"), "point_p95": point_p95, "bootstrap_null_p95": bootstrap_p95, "p95": bootstrap_p95, "bootstrap_null_draws": bootstrap_draws, "point_estimates": values, "partition_count": len(values), "null_draw_count": len(bootstrap_draws)}
    split_half_null = {"family_A": half_a, "family_B": half_b, "by_task": null_by_task, "bootstrap_seed": int(seed), "method": "for each task, merge the three 10000-draw cluster-bootstrap distributions from family A with the three from family B and take one 95th percentile; partition point estimates are descriptive only, and reverse directions are not duplicated"}
    family_success = {"A": _family_success(events, FAMILY_A), "B": _family_success(events, FAMILY_B), "full": _family_success(events, FAMILY_FULL)}
    reference_ops = sorted({_text(row["operator"]) for row in usable if _text(row["operator"]) not in OPERATORS}, key=_sort)
    references = _reference_summary(usable, reference_ops, PHASE2_PREFIXES, include_bootstrap=True)
    references.update({"retained_separately": True, "used_for_selection": False})
    gate_by_task = {}
    for task, metrics in by_task.items():
        lower = metrics["bootstrap"]["ci95"][0]
        floor = null_by_task.get(task, {}).get("p95")
        checks = {"both_defined_at_least_30": metrics["both_defined"] >= 30, "minority_at_least_015": metrics["minority_crossing_rate"] is not None and metrics["minority_crossing_rate"] >= .15, "ci_lower_above_split_half_p95": lower is not None and floor is not None and lower > floor, "spearman_at_most_08": metrics["spearman"] is not None and metrics["spearman"] <= .8}
        gate_by_task[task] = {"checks": checks, "pass": all(checks.values()), "minority_ci_lower": lower, "split_half_null_p95": floor, "spearman": metrics["spearman"], "both_defined": metrics["both_defined"]}
    gate_pass = bool(gate_by_task) and all(item["pass"] for item in gate_by_task.values())
    return {"schema_version": 1, "analysis": "crossing", "status": "COMPLETE", "blocked": False, "split": split, "row_count": len(material), "usable_row_count": len(usable), "excluded_error_row_count": len(validation["errors"]), "bootstrap_replicates": int(replicates), "bootstrap_seed": int(seed), "operators": list(OPERATORS), "families": {"A": list(FAMILY_A), "B": list(FAMILY_B), "full": list(FAMILY_FULL)}, "prefixes": list(PHASE2_PREFIXES), "recovery_actor_seeds": seeds, "boundaries": boundaries, "crossing": crossing, "split_half_null": split_half_null, "safe_success_by_family": family_success, "references": references, "selection_source": "reference operators are retained separately and excluded from selection", "k3": {"status": "PASS" if gate_pass else "FAIL", "decision": "NONTRIVIAL_OPERATOR_RELATIVITY" if gate_pass else "MONOTONE_RESCALING_ONLY", "by_task": gate_by_task}}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        payload = payload.get("rows", payload.get("data", [])) if isinstance(payload, Mapping) else payload
        if not isinstance(payload, list): raise ValueError("JSON input must be an array or object containing rows")
        return [dict(row) for row in payload]
    return [dict(json.loads(line)) for line in text.splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--input-path", dest="input_path", type=Path, required=True)
    parser.add_argument("--output", "--output-path", dest="output_path", type=Path, required=True)
    parser.add_argument("--mode", choices=("grid", "crossing"), default="grid")
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=216214)
    args = parser.parse_args(argv)
    rows = _load_rows(args.input_path)
    result = summarize_grid(rows) if args.mode == "grid" else analyze_crossing(rows, args.split, args.replicates, args.seed)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result.get("status"), "output": str(args.output_path)}))
    return 0 if result.get("status") != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
