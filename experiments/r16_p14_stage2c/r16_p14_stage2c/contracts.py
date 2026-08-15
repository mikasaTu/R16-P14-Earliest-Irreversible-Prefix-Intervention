from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Iterable

import numpy as np

from .settings import DETECTION_PREFIX, H_VALID


EXACT_REPLAY_CHECKS = (
    "anchor_state_exact",
    "state_history_exact",
    "action_history_exact",
    "original_chunk_exact",
    "branch_order_invariant",
)


def admit_event_replay(attempts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(attempts)
    checks = {
        "three_fresh_reconstructions": len(rows) == 3,
        "no_errors": len(rows) == 3 and all(row.get("error") is None for row in rows),
        "distinct_order_slots": len(rows) == 3 and {row.get("order_slot") for row in rows} == {0, 1, 2},
    }
    for name in EXACT_REPLAY_CHECKS:
        checks[name] = len(rows) == 3 and all(bool(row.get(name, False)) for row in rows)
    passed = all(checks.values())
    return {
        "passed": passed,
        "admitted": passed,
        "outcome_fields_read": False,
        "checks": checks,
        "failure_label": None if passed else "UNSTABLE_EVENT_EXCLUDED",
    }


def replay_cell_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    attempts = list(rows)
    successes = [
        row for row in attempts
        if row.get("error") is None
        and not row.get("missing", False)
        and bool(row.get("replay_valid", False))
    ]
    error_count = sum(row.get("error") is not None or row.get("missing", False) for row in attempts)
    rate = len(successes) / len(attempts) if attempts else 0.0
    return {
        "attempt_count": len(attempts),
        "success_count": len(successes),
        "error_count": error_count,
        "replay_rate": float(rate),
        "passes": bool(attempts) and rate >= 0.99 and error_count == 0,
    }


def monotone_observed_safety(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["prefix_k"]))
    indices = [int(row["prefix_k"]) for row in ordered]
    values = [int(bool(row["S_obs"])) for row in ordered]
    transitions = [
        {"from_k": indices[i - 1], "to_k": indices[i]}
        for i in range(1, len(values))
        if values[i - 1] == 0 and values[i] == 1
    ]
    valid_grid = indices == list(range(DETECTION_PREFIX, H_VALID + 1))
    passed = valid_grid and not transitions and all(v in (0, 1) for v in values)
    return {
        "passed": passed,
        "failure_label": None if passed else "NONMONOTONIC_CAUSE_PREFIX",
        "complete_prefix_grid": valid_grid,
        "transitions": transitions,
        "k_last_observed_safe": max((k for k, s in zip(indices, values) if s), default=None)
        if passed else None,
    }


def matched_branch_contract(prefix_k: int, post_detection_budget: int) -> dict[str, Any]:
    if prefix_k not in range(DETECTION_PREFIX, H_VALID + 1):
        raise ValueError(prefix_k)
    prefix_actions = prefix_k - DETECTION_PREFIX
    tail_budget = max(0, int(post_detection_budget) - prefix_actions)
    common = {
        "prefix_k": prefix_k,
        "prefix_actions": prefix_actions,
        "pre_k_actor_calls": 1,
        "actor_calls_through_k": 2,
        "tail_execution_horizon": 16,
        "post_detection_action_budget": int(post_detection_budget),
        "tail_action_budget": tail_budget,
    }
    return {
        "CACHED_MATCHED": {**common, "prefix_source": "cached", "d_call_executed": False},
        "FRESH_MATCHED": {**common, "prefix_source": "fresh", "d_call_executed": prefix_actions > 0},
    }


def recovery_operator_contract(post_detection_budget: int) -> dict[str, dict[str, Any]]:
    if post_detection_budget < 1:
        raise ValueError(post_detection_budget)
    call_budget = int(np.ceil(post_detection_budget / 4))
    return {
        "fresh_h16": {"action_budget": post_detection_budget, "policy_call_budget": call_budget, "execution_horizon": 16, "prelude_actions": 0},
        "fresh_h4": {"action_budget": post_detection_budget, "policy_call_budget": call_budget, "execution_horizon": 4, "prelude_actions": 0},
        "hold_one_step_then_fresh_h16": {"action_budget": post_detection_budget, "policy_call_budget": call_budget, "execution_horizon": 16, "prelude_actions": 1},
        "rollback_one_step_then_fresh_h16": {"action_budget": post_detection_budget, "policy_call_budget": call_budget, "execution_horizon": 16, "prelude_actions": 1},
    }
def persistent_irreversibility_prefix(rows: Iterable[dict[str, Any]]) -> int | None:
    by_k = {int(row["prefix_k"]): int(bool(row["R_U"])) for row in rows}
    if sorted(by_k) != list(range(DETECTION_PREFIX, H_VALID + 1)):
        raise ValueError("complete prefix grid required")
    for k in range(DETECTION_PREFIX, H_VALID + 1):
        if all(by_k[j] == 0 for j in range(k, H_VALID + 1)):
            return k
    return None


def last_recoverable_prefix(rows: Iterable[dict[str, Any]]) -> int | None:
    return max((int(row["prefix_k"]) for row in rows if bool(row["R_U"])), default=None)


def qualify_cell(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    attempts = list(rows)
    valid = [
        row for row in attempts
        if row.get("error") is None
        and row.get("replay_valid")
        and row.get("injection_valid", True)
        and row.get("monotonicity_valid", True)
    ]
    event_rows: dict[str, dict[str, Any]] = {}
    for row in valid:
        event_rows[str(row["event_id"])] = row
    events = list(event_rows.values())
    seed_counts = Counter(int(row["actor_seed"]) for row in events)
    immediate = float(np.mean([bool(row["immediate_cause_violation"]) for row in events])) if events else 1.0
    delayed = float(np.mean([bool(row["delayed_cause_violation"]) for row in events])) if events else 0.0
    offsets = [int(row["first_violation_offset"]) for row in events if row.get("first_violation_offset") is not None and not row["immediate_cause_violation"]]
    median_offset = float(np.median(offsets)) if offsets else None
    interior = [offset for offset in offsets if offset <= (H_VALID - DETECTION_PREFIX - 2)]
    interior_fraction = len(interior) / len(offsets) if offsets else 0.0
    replay = replay_cell_summary(attempts)
    checks = {
        "valid_events_ge_20": len(events) >= 20,
        "two_actor_seeds_ge_5": sum(count >= 5 for count in seed_counts.values()) >= 2,
        "immediate_le_0_10": immediate <= 0.10,
        "delayed_in_0_30_0_80": 0.30 <= delayed <= 0.80,
        "median_offset_ge_2": median_offset is not None and median_offset >= 2.0,
        "interior_fraction_ge_0_20": interior_fraction >= 0.20,
        "replay_rate_ge_0_99": replay["replay_rate"] >= 0.99,
        "error_count_eq_0": replay["error_count"] == 0,
        "injection_contact_count_eq_0": sum(bool(row.get("injection_contact", False)) for row in attempts) == 0,
        "nonmonotonic_event_count_eq_0": sum(not bool(row.get("monotonicity_valid", True)) for row in attempts) == 0,
    }
    return {
        "attempt_count": len(attempts),
        "valid_event_count": len(events),
        "actor_seed_event_counts": {str(k): v for k, v in sorted(seed_counts.items())},
        "immediate_cause_violation_rate": immediate,
        "delayed_cause_violation_rate": delayed,
        "median_first_violation_offset": median_offset,
        "interior_violation_fraction": interior_fraction,
        "replay_rate": replay["replay_rate"],
        "error_count": replay["error_count"],
        "checks": checks,
        "qualifies": all(checks.values()),
    }


def freeze_strongest_baseline(calibration_rows: Iterable[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calibration_rows:
        if row.get("split") != "calibration":
            raise ValueError("baseline selection accepts calibration rows only")
        grouped[str(row["method"])].append(row)
    if not grouped:
        raise ValueError("no calibration baselines")
    ranked = []
    for name, rows in grouped.items():
        ranked.append((-
            float(np.mean([row["safe_success"] for row in rows])),
            float(np.mean([row["new_non_nominal_actions"] for row in rows])),
            float(np.mean([row["actor_calls"] for row in rows])),
            name,
        ))
    return min(ranked)[-1]


def select_crossfit_prefix(fit_rows: Iterable[dict[str, Any]], held_out_seed: int) -> int:
    rows = list(fit_rows)
    if any(int(row["recovery_actor_seed"]) == int(held_out_seed) for row in rows):
        raise ValueError("held-out actor outcome entered prefix selection")
    by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_k[int(row["prefix_k"])].append(row)
    if not by_k:
        raise ValueError("empty fit rows")
    rank = []
    for k, group in by_k.items():
        rank.append((
            -float(np.mean([row["safe_success"] for row in group])),
            float(np.mean([row["new_non_nominal_actions"] for row in group])),
            float(np.mean([row["actor_calls"] for row in group])),
            k,
        ))
    return min(rank)[-1]


def positive_label_allowed(upstream: dict[str, bool]) -> bool:
    return all(bool(value) for value in upstream.values())


def select_second_failure_family(cells: Iterable[dict[str, Any]], candidate_order: Iterable[str]) -> str | None:
    rows = list(cells)
    forbidden = {"safe_success", "method_gain", "track_a_gain", "track_b_gain", "oracle"}
    if any(forbidden.intersection(row) for row in rows):
        raise ValueError("method outcomes are forbidden during failure-family selection")
    by_task: dict[str, int] = Counter(
        str(row["task"]) for row in rows if bool(row.get("qualifies", False))
    )
    for task in candidate_order:
        if by_task[str(task)] >= 2:
            return str(task)
    return None


def pending_shard_ids(expected_ids: Iterable[str], completed_ids: Iterable[str]) -> list[str]:
    expected = list(expected_ids)
    completed = set(completed_ids)
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate expected shard id")
    return [item for item in expected if item not in completed]


def assert_no_physical_irreversibility_label(payload: Any) -> None:
    normalized = str(payload).lower().replace("-", "_").replace(" ", "_")
    if "physical_irreversibility" in normalized:
        raise ValueError("physical irreversibility labels are forbidden")


def hierarchical_cluster_bootstrap(
    paired_rows: Iterable[dict[str, Any]],
    delta: Callable[[dict[str, Any]], float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rows = list(paired_rows)
    clusters: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[(str(row["task"]), int(row["init_state_id"]))].append(row)
    keys = sorted(clusters)
    if not keys:
        raise ValueError("no clusters")
    cluster_values = np.asarray([
        np.mean([delta(row) for row in clusters[key]]) for key in keys
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        sampled = rng.integers(0, len(keys), size=len(keys))
        draws[i] = float(np.mean(cluster_values[sampled]))
    return {
        "unit": "task_init_state_cluster",
        "cluster_count": len(keys),
        "row_count": len(rows),
        "estimate": float(np.mean(cluster_values)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "replicates": replicates,
        "seed": seed,
    }
