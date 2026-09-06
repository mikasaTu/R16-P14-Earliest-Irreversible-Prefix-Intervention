"""Phase-1 operator effect summaries over the completed calibration support set.

This module is deliberately downstream of the formal consolidator.  It reads
only phase1/grid_rows.jsonl and phase1/summary.json, requires the immutable
complete summary before opening grid_rows, and performs no selection or
evaluation read.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # package invocation
    from .consolidate import (
        BUDGETS,
        OPERATORS,
        PHASE1_PREFIXES,
        SEEDS,
        TASKS,
    )
except ImportError:  # direct script invocation
    from consolidate import (  # type: ignore
        BUDGETS,
        OPERATORS,
        PHASE1_PREFIXES,
        SEEDS,
        TASKS,
    )

# Keep the names/order frozen for the report and for deterministic bootstrap
# streams.  The three differences are paired at the same event and budget.
PAIR_DEFINITIONS = {
    "hold_1+fresh_h4-minus-fresh_h4": ("hold_1+fresh_h4", "fresh_h4"),
    "rollback_1+fresh_h16-minus-fresh_h16": (
        "rollback_1+fresh_h16",
        "fresh_h16",
    ),
    "fresh_h16-minus-fresh_h4": ("fresh_h16", "fresh_h4"),
}
ACTION_STREAM_FIELDS = (
    "action_stream_hash",
    "recovery_action_stream_hash",
    # Current runtime emits the per-new-action hashes under this name.  It is
    # compared verbatim and is never collapsed into an invented digest.
    "recovery_action_hashes",
)
EXPECTED_SEEDS = tuple(str(seed) for seed in SEEDS)
EXPECTED_PREFIXES = tuple(int(prefix) for prefix in PHASE1_PREFIXES)
CONFIGURED_BUDGETS = tuple(
    tuple(int(value) for value in budget) for budget in BUDGETS
)
BUDGET_FIELDS = ("tail_horizon", "action_budget", "policy_call_cap")
IDENTITY_FIELDS = (
    "task",
    "event_instance_id",
    "init_state_id",
    "split",
    "generator_actor_seed",
)
ROW_FIELDS = IDENTITY_FIELDS + (
    "recovery_actor_seed",
    "operator",
    "prefix_k",
    "tail_horizon",
    "action_budget",
    "policy_call_cap",
    "status",
)


class OperatorEffectsError(RuntimeError):
    """Raised for malformed input; public APIs convert it to BLOCKED."""


def _text(value: Any) -> str:
    return str(value)


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer")
    return result


def _seed(value: Any, field: str = "seed") -> str:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer seed")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer seed")
    return str(int(value)) if isinstance(value, (int, float)) else str(value)


def _float01(value: Any, field: str = "safe_success") -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be finite and in [0,1]")
    return result


def _budget_from_row(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return tuple(_int(row[field], field) for field in BUDGET_FIELDS)


def _budget_dict(budget: tuple[int, int, int]) -> dict[str, int]:
    return dict(zip(BUDGET_FIELDS, (int(value) for value in budget)))


def _budget_name(budget: tuple[int, int, int]) -> str:
    return "/".join(
        f"{field}={value}" for field, value in zip(BUDGET_FIELDS, budget)
    )


def _event_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        _text(row["task"]),
        _text(row["event_instance_id"]),
        _text(row["init_state_id"]),
        _text(row["split"]),
        _seed(row["generator_actor_seed"], "generator_actor_seed"),
    )


def _event_label(key: tuple[str, str, str, str, str]) -> dict[str, str]:
    return {
        "task": key[0],
        "event_instance_id": key[1],
        "init_state_id": key[2],
        "split": key[3],
        "generator_actor_seed": key[4],
    }


def _event_sort(key: tuple[str, str, str, str, str]) -> tuple[Any, ...]:
    try:
        init = (0, int(key[2]))
    except ValueError:
        init = (1, key[2])
    try:
        generator = (0, int(key[4]))
    except ValueError:
        generator = (1, key[4])
    return key[0], init, key[1], key[3], generator


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _dedupe_reasons(reasons: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(reason) for reason in reasons))


def _resolve_input_paths(input_root: str | Path) -> tuple[Path, Path, Path]:
    root = Path(input_root)
    if root.is_file():
        if root.name == "grid_rows.jsonl":
            phase = root.parent
            return root, phase / "summary.json", phase
        raise OperatorEffectsError(
            "input-root file must be phase1/grid_rows.jsonl"
        )
    if (root / "grid_rows.jsonl").is_file() or (root / "summary.json").is_file():
        phase = root
    else:
        phase = root / "phase1"
    return phase / "grid_rows.jsonl", phase / "summary.json", phase


def _load_summary(summary_path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, [f"cannot read Phase-1 summary: {exc}"]
    if not isinstance(payload, Mapping):
        return None, ["Phase-1 summary is not an object"]
    summary = dict(payload)
    status = summary.get("status")
    available_shortfall = status == "COMPLETE_AVAILABLE_REQUESTS_SAMPLE_SHORTFALL"
    if status not in ("COMPLETE", "COMPLETE_AVAILABLE_REQUESTS_SAMPLE_SHORTFALL") or summary.get("blocked") is True:
        reasons.append(
            f"refusing incomplete/blocked consolidated summary: "
            f"status={status!r}, blocked={summary.get('blocked')!r}"
        )
    if summary.get("split") not in (None, "calibration"):
        reasons.append(f"Phase-1 summary has non-calibration split: {summary.get('split')!r}")
    complete = summary.get("sample_complete")
    if complete is None:
        complete = summary.get("planned_sample_complete")
    if available_shortfall:
        if complete is not False:
            reasons.append(
                "available-request shortfall must declare sample_complete=false"
            )
        selection = summary.get("selection")
        if not isinstance(selection, Mapping):
            reasons.append(
                "available-request shortfall lacks formal selection receipt"
            )
        elif selection.get("status") != "BLOCKED" or selection.get("selected_budget") is not None:
            reasons.append(
                "available-request shortfall must retain BLOCKED/null selection"
            )
    elif complete is not True:
        reasons.append("formal Phase-1 sample_complete is not true")
    completeness = summary.get("completeness")
    if not isinstance(completeness, Mapping):
        reasons.append("Phase-1 summary lacks completeness metadata")
    else:
        if completeness.get("status") not in (None, "COMPLETE"):
            reasons.append(
                f"Phase-1 completeness is not COMPLETE: {completeness.get('status')!r}"
            )
    planned = summary.get("planned_events")
    if planned is not None:
        try:
            if _int(planned, "planned_events") != 20:
                reasons.append(f"planned_events must remain 20, observed {planned!r}")
        except (TypeError, ValueError) as exc:
            reasons.append(str(exc))
    observed = summary.get("observed_events_by_task")
    if observed is None and isinstance(completeness, Mapping):
        observed = completeness.get("observed_events_by_task")
    if observed is None:
        reasons.append("summary lacks observed_events_by_task")
    elif not isinstance(observed, Mapping):
        reasons.append("observed_events_by_task is not an object")
    else:
        observed_counts: dict[str, int] = {}
        for task in TASKS:
            try:
                count = _int(observed.get(task), f"observed_events_by_task[{task}]")
                observed_counts[task] = count
            except (TypeError, ValueError) as exc:
                reasons.append(str(exc))
                continue
            if available_shortfall:
                if not 0 < count <= 20:
                    reasons.append(
                        f"available Phase-1 event count for {task} must be in [1,20], "
                        f"observed {count}"
                    )
            elif count != 20:
                reasons.append(
                    f"formal Phase-1 sample is not complete for {task}: observed {count}"
                )
        if available_shortfall and observed_counts and all(
            count == 20 for count in observed_counts.values()
        ):
            reasons.append(
                "shortfall status requires at least one planned event count below 20"
            )
        if available_shortfall:
            shortfall = summary.get("shortfall")
            if shortfall is None and isinstance(completeness, Mapping):
                shortfall = completeness.get("shortfall")
            if not isinstance(shortfall, Mapping):
                reasons.append("available-request shortfall lacks shortfall metadata")
            else:
                for task, count in observed_counts.items():
                    try:
                        declared = _int(
                            shortfall.get(task),
                            f"shortfall[{task}]",
                        )
                    except (TypeError, ValueError) as exc:
                        reasons.append(str(exc))
                        continue
                    expected = 20 - count
                    if declared != expected:
                        reasons.append(
                            f"shortfall for {task} disagrees with observed/planned: "
                            f"{declared} vs {expected}"
                        )
    excluded = summary.get("structurally_excluded_events", [])
    if not isinstance(excluded, list):
        reasons.append("structurally_excluded_events is not a list")
        excluded = []
    excluded_count = summary.get("structurally_excluded_event_count")
    if excluded_count is not None:
        try:
            if _int(excluded_count, "structurally_excluded_event_count") != len(excluded):
                reasons.append(
                    "structurally_excluded_event_count disagrees with event list"
                )
        except (TypeError, ValueError) as exc:
            reasons.append(str(exc))
    excluded_keys: set[tuple[str, str, str, str, str]] = set()
    for index, item in enumerate(excluded):
        if not isinstance(item, Mapping):
            reasons.append(f"structurally excluded event {index} is not an object")
            continue
        missing = [field for field in IDENTITY_FIELDS if field not in item]
        if missing:
            reasons.append(
                f"structurally excluded event {index} missing identity: {missing}"
            )
            continue
        try:
            key = _event_key(item)
        except (KeyError, TypeError, ValueError) as exc:
            reasons.append(f"invalid structural event {index}: {exc}")
            continue
        if key in excluded_keys:
            reasons.append(f"duplicate structurally excluded event: {key!r}")
        excluded_keys.add(key)
    return summary, _dedupe_reasons(reasons)


def _row_identity(
    row: Mapping[str, Any], index: int
) -> tuple[tuple[str, str, str, str, str], tuple[int, int, int], str, int, str]:
    missing = [field for field in ROW_FIELDS if field not in row]
    if missing:
        raise ValueError(f"row {index} missing fields: {missing}")
    event = _event_key(row)
    budget = _budget_from_row(row)
    operator = _text(row["operator"])
    prefix = _int(row["prefix_k"], "prefix_k")
    if event[3] != "calibration":
        raise ValueError(f"row {index} has non-calibration split: {event[3]!r}")
    if event[0] not in TASKS:
        raise ValueError(f"row {index} has unexpected task: {event[0]!r}")
    if operator not in OPERATORS:
        raise ValueError(f"row {index} has unexpected core operator: {operator!r}")
    if prefix not in EXPECTED_PREFIXES:
        raise ValueError(f"row {index} has unexpected Phase-1 prefix: {prefix!r}")
    if budget not in CONFIGURED_BUDGETS:
        raise ValueError(f"row {index} has budget outside configured grid: {budget!r}")
    recovery_seed = _seed(row["recovery_actor_seed"], "recovery_actor_seed")
    if recovery_seed not in EXPECTED_SEEDS:
        raise ValueError(
            f"row {index} has unexpected recovery seed: {recovery_seed!r}"
        )
    return event, budget, operator, prefix, recovery_seed


def _load_rows(
    grid_path: Path,
    excluded_keys: set[tuple[str, str, str, str, str]],
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[int, int, int], dict[tuple[str, str, str, str, str], dict[tuple[str, int, str], dict[str, Any]]]],
    dict[tuple[int, int, int], dict[tuple[str, str, str, str, str], dict[tuple[str, int, str], dict[str, Any]]]],
    set[tuple[str, str, str, str, str]],
    set[tuple[str, str, str, str, str]],
    list[str],
]:
    """Read only the consolidated core grid and form raw/support indexes."""
    raw_rows: list[dict[str, Any]] = []
    support_groups: dict[
        tuple[int, int, int],
        dict[
            tuple[str, str, str, str, str],
            dict[tuple[str, int, str], dict[str, Any]],
        ],
    ] = defaultdict(lambda: defaultdict(dict))
    raw_groups: dict[
        tuple[int, int, int],
        dict[
            tuple[str, str, str, str, str],
            dict[tuple[str, int, str], dict[str, Any]],
        ],
    ] = defaultdict(lambda: defaultdict(dict))
    observed_events: set[tuple[str, str, str, str, str]] = set()
    structural_events: set[tuple[str, str, str, str, str]] = set()
    reasons: list[str] = []
    try:
        handle = grid_path.open("r", encoding="utf-8")
    except OSError as exc:
        return [], support_groups, raw_groups, observed_events, structural_events, [
            f"cannot read consolidated grid_rows.jsonl: {exc}"
        ]
    with handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                reasons.append(f"grid row {index} is invalid JSON: {exc}")
                continue
            if not isinstance(payload, Mapping):
                reasons.append(f"grid row {index} is not an object")
                continue
            row = dict(payload)
            try:
                event, budget, operator, prefix, recovery_seed = _row_identity(row, index)
            except (KeyError, TypeError, ValueError) as exc:
                reasons.append(str(exc))
                continue
            observed_events.add(event)
            branch_key = (operator, prefix, recovery_seed)
            if branch_key in raw_groups[budget][event]:
                reasons.append(
                    f"duplicate consolidated branch key: "
                    f"{event!r}, budget={budget!r}, branch={branch_key!r}"
                )
                continue
            raw_groups[budget][event][branch_key] = row
            raw_rows.append(row)
            status = _text(row.get("status")).strip().upper()
            error_type = _text(row.get("error_type")).strip()
            marker = status.startswith("BLOCKED") and error_type == "PrefixOutsideTaskHorizon"
            explicit_structural = bool(row.get("structurally_excluded"))
            if explicit_structural and not marker:
                reasons.append(
                    f"row {index} claims structural exclusion without "
                    "BLOCKED/PrefixOutsideTaskHorizon provenance"
                )
            is_structural = marker or explicit_structural
            if is_structural:
                structural_events.add(event)
                if event not in excluded_keys:
                    reasons.append(
                        f"row {index} is structurally excluded but summary does not "
                        f"list event {event!r}"
                    )
                if not marker:
                    reasons.append(
                        f"row {index} structural status/error marker is not exact"
                    )
                continue
            if status != "COMPLETE":
                reasons.append(
                    f"row {index} has non-COMPLETE status and is not a structural "
                    f"horizon row: {row.get('status')!r}"
                )
                continue
            if event in excluded_keys:
                # A structural event is removed as a whole.  Its remaining
                # COMPLETE branches are retained in the raw file but cannot
                # become observations.
                continue
            try:
                row["safe_success"] = _float01(row["safe_success"])
            except (KeyError, TypeError, ValueError) as exc:
                reasons.append(f"row {index} invalid safe_success: {exc}")
                continue
            if row.get("is_reference") is True:
                reasons.append(
                    f"core grid row {index} is marked is_reference=true"
                )
                continue
            support_groups[budget][event][branch_key] = row
    return (
        raw_rows,
        support_groups,
        raw_groups,
        observed_events,
        structural_events,
        _dedupe_reasons(reasons),
    )


def _expected_branch_keys() -> set[tuple[str, int, str]]:
    return {
        (operator, int(prefix), seed)
        for operator in OPERATORS
        for prefix in EXPECTED_PREFIXES
        for seed in EXPECTED_SEEDS
    }


def _validate_inventory(
    summary: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
    support_groups: Mapping[
        tuple[int, int, int],
        Mapping[
            tuple[str, str, str, str, str],
            Mapping[tuple[str, int, str], Mapping[str, Any]],
        ],
    ],
    raw_groups: Mapping[
        tuple[int, int, int],
        Mapping[
            tuple[str, str, str, str, str],
            Mapping[tuple[str, int, str], Mapping[str, Any]],
        ],
    ],
    observed_events: set[tuple[str, str, str, str, str]],
    structural_events: set[tuple[str, str, str, str, str]],
    excluded_keys: set[tuple[str, str, str, str, str]],
) -> list[str]:
    reasons: list[str] = []
    if structural_events != excluded_keys:
        missing = sorted(excluded_keys - structural_events, key=_event_sort)
        extra = sorted(structural_events - excluded_keys, key=_event_sort)
        if missing:
            reasons.append(
                f"summary lists structural events with no BLOCKED branch evidence: {missing!r}"
            )
        if extra:
            reasons.append(
                f"grid contains unlisted structural events: {extra!r}"
            )
    expected_tasks = set(TASKS)
    observed_tasks = {event[0] for event in observed_events}
    if observed_tasks != expected_tasks:
        reasons.append(
            f"grid event tasks mismatch: expected {sorted(expected_tasks)}, "
            f"observed {sorted(observed_tasks)}"
        )
    observed_meta = summary.get("observed_events_by_task")
    if observed_meta is None and isinstance(summary.get("completeness"), Mapping):
        observed_meta = summary["completeness"].get("observed_events_by_task")
    if isinstance(observed_meta, Mapping):
        for task in TASKS:
            actual = sum(event[0] == task for event in observed_events)
            try:
                declared = _int(observed_meta.get(task), f"observed_events_by_task[{task}]")
            except (TypeError, ValueError) as exc:
                reasons.append(str(exc))
                continue
            if actual != declared:
                reasons.append(
                    f"observed event identity count for {task} disagrees with "
                    f"summary: {actual} vs {declared}"
                )
    if set(support_groups) != set(CONFIGURED_BUDGETS):
        missing = sorted(set(CONFIGURED_BUDGETS) - set(support_groups))
        extra = sorted(set(support_groups) - set(CONFIGURED_BUDGETS))
        if missing:
            reasons.append(f"missing configured budgets in support grid: {missing!r}")
        if extra:
            reasons.append(f"unexpected budgets in support grid: {extra!r}")
    for budget in CONFIGURED_BUDGETS:
        support_events_by_task = {
            task: {
                event
                for event in support_groups.get(budget, {})
                if event[0] == task
            }
            for task in TASKS
        }
        raw_events_by_task = {
            task: {
                event for event in raw_groups.get(budget, {}) if event[0] == task
            }
            for task in TASKS
        }
        for task in TASKS:
            expected_support_events = {
                event
                for event in observed_events
                if event[0] == task and event not in excluded_keys
            }
            if support_events_by_task[task] != expected_support_events:
                reasons.append(
                    f"support event inventory mismatch for {task}, "
                    f"budget={budget!r}: observed "
                    f"{len(support_events_by_task[task])}, expected "
                    f"{len(expected_support_events)}"
                )
            expected_raw_events = {event for event in observed_events if event[0] == task}
            if raw_events_by_task[task] != expected_raw_events:
                reasons.append(
                    f"raw event inventory mismatch for {task}, budget={budget!r}: "
                    f"observed {len(raw_events_by_task[task])}, expected "
                    f"{len(expected_raw_events)}"
                )
            for event in sorted(expected_support_events, key=_event_sort):
                branches = set(support_groups.get(budget, {}).get(event, {}))
                expected_branches = _expected_branch_keys()
                if branches != expected_branches:
                    reasons.append(
                        f"incomplete support branches for {event!r}, "
                        f"budget={budget!r}: observed {len(branches)}, "
                        f"expected {len(expected_branches)}"
                    )
            for event in sorted(expected_raw_events, key=_event_sort):
                branches = set(raw_groups.get(budget, {}).get(event, {}))
                if not branches:
                    reasons.append(
                        f"missing raw branches for {event!r}, budget={budget!r}"
                    )
    declared_grid_rows = summary.get("grid_row_count")
    if declared_grid_rows is not None:
        try:
            if _int(declared_grid_rows, "grid_row_count") != len(raw_rows):
                reasons.append(
                    f"grid_row_count disagrees with consolidated file: "
                    f"{declared_grid_rows!r} vs {len(raw_rows)}"
                )
        except (TypeError, ValueError) as exc:
            reasons.append(str(exc))
    declared_support_rows = summary.get("support_grid_row_count")
    if declared_support_rows is not None:
        actual_support = sum(
            len(branches)
            for groups in support_groups.values()
            for branches in groups.values()
        )
        try:
            if _int(declared_support_rows, "support_grid_row_count") != actual_support:
                reasons.append(
                    f"support_grid_row_count disagrees with support file: "
                    f"{declared_support_rows!r} vs {actual_support}"
                )
        except (TypeError, ValueError) as exc:
            reasons.append(str(exc))
    return _dedupe_reasons(reasons)


def _cluster_metric(
    event_values: Mapping[tuple[str, str, str, str, str], float],
    *,
    branch_count: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if not event_values:
        raise OperatorEffectsError("no event support remains for metric")
    if replicates <= 0:
        raise OperatorEffectsError("bootstrap replicates must be positive")
    by_init: dict[tuple[str, str], list[float]] = defaultdict(list)
    for event, value in event_values.items():
        by_init[(event[0], event[2])].append(float(value))
    cluster_means = {
        key: float(np.mean(values)) for key, values in by_init.items()
    }
    ordered_clusters = sorted(cluster_means, key=lambda key: (key[0], key[1]))
    point = float(np.mean([cluster_means[key] for key in ordered_clusters]))
    rng = np.random.default_rng(int(seed))
    values = np.asarray([cluster_means[key] for key in ordered_clusters], dtype=float)
    indices = rng.integers(0, len(values), size=(int(replicates), len(values)))
    draws = values[indices].mean(axis=1)
    return {
        "estimate": point,
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "bootstrap_unit": ["task", "init_state_id"],
        "event_count": len(event_values),
        "init_count": len(ordered_clusters),
        "branch_count": int(branch_count),
        "cluster_values": [
            {
                "task": key[0],
                "init_state_id": key[1],
                "mean": cluster_means[key],
                "event_count": len(by_init[key]),
            }
            for key in ordered_clusters
        ],
        "event_values": [
            {
                **_event_label(key),
                "mean": float(event_values[key]),
            }
            for key in sorted(event_values, key=_event_sort)
        ],
        "estimand": (
            "mean over init_state clusters after each event mean; each event "
            "mean averages three recovery seeds at each of five fixed prefixes"
        ),
    }


def _event_arm_values(
    groups: Mapping[tuple[str, str, str, str, str], Mapping[tuple[str, int, str], Mapping[str, Any]]],
    events: Sequence[tuple[str, str, str, str, str]],
) -> tuple[dict[str, dict[tuple[str, str, str, str, str], float]], list[str]]:
    reasons: list[str] = []
    result: dict[
        str, dict[tuple[str, str, str, str, str], float]
    ] = {operator: {} for operator in OPERATORS}
    for event in events:
        for operator in OPERATORS:
            prefix_means: list[float] = []
            for prefix in EXPECTED_PREFIXES:
                values = []
                for seed in EXPECTED_SEEDS:
                    row = groups.get(event, {}).get((operator, prefix, seed))
                    if row is None:
                        reasons.append(
                            f"missing branch while aggregating {event!r}, "
                            f"operator={operator}, prefix={prefix}, seed={seed}"
                        )
                        continue
                    values.append(_float01(row["safe_success"]))
                if len(values) != len(EXPECTED_SEEDS):
                    continue
                prefix_means.append(float(np.mean(values)))
            if len(prefix_means) != len(EXPECTED_PREFIXES):
                continue
            result[operator][event] = float(np.mean(prefix_means))
    return result, _dedupe_reasons(reasons)


def _paired_values(
    values: Mapping[str, Mapping[tuple[str, str, str, str, str], float]],
    *,
    left: str,
    right: str,
) -> dict[tuple[str, str, str, str, str], float]:
    left_values = values.get(left, {})
    right_values = values.get(right, {})
    return {
        event: float(left_values[event] - right_values[event])
        for event in sorted(set(left_values) & set(right_values), key=_event_sort)
    }


def _task_budget_summary(
    budget: tuple[int, int, int],
    task: str,
    groups: Mapping[tuple[str, str, str, str, str], Mapping[tuple[str, int, str], Mapping[str, Any]]],
    all_events: Sequence[tuple[str, str, str, str, str]],
    excluded_keys: set[tuple[str, str, str, str, str]],
    raw_groups: Mapping[tuple[str, str, str, str, str], Mapping[tuple[str, int, str], Mapping[str, Any]]],
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], list[str]]:
    support_events = [
        event
        for event in all_events
        if event[0] == task and event not in excluded_keys
    ]
    if not support_events:
        return {}, [f"no complete support events for {task}, budget={budget!r}"]
    event_values, reasons = _event_arm_values(groups, support_events)
    if reasons:
        return {}, reasons
    absolute = {}
    for operator in OPERATORS:
        absolute[operator] = _cluster_metric(
            event_values[operator],
            branch_count=len(
                [
                    row
                    for event in support_events
                    for branch, row in groups.get(event, {}).items()
                    if branch[0] == operator
                ]
            ),
            replicates=replicates,
            seed=seed,
        )
    paired = {}
    for name, (left, right) in PAIR_DEFINITIONS.items():
        differences = _paired_values(event_values, left=left, right=right)
        if set(differences) != set(support_events):
            reasons.append(
                f"pair {name} does not match the same support events: "
                f"{len(differences)} vs {len(support_events)}"
            )
            continue
        paired[name] = _cluster_metric(
            differences,
            branch_count=len(differences) * len(EXPECTED_PREFIXES) * len(EXPECTED_SEEDS),
            replicates=replicates,
            seed=seed,
        )
        paired[name]["left_operator"] = left
        paired[name]["right_operator"] = right
        paired[name]["paired_event_count"] = len(differences)
        paired[name]["paired_init_count"] = len(
            {(event[0], event[2]) for event in differences}
        )
        paired[name]["left_branch_count"] = len(differences) * len(EXPECTED_PREFIXES) * len(EXPECTED_SEEDS)
        paired[name]["right_branch_count"] = paired[name]["left_branch_count"]
    support_branch_count = sum(
        len(groups.get(event, {})) for event in support_events
    )
    raw_branch_count = sum(len(raw_groups.get(event, {})) for event in all_events if event[0] == task)
    task_summary = {
        "task": task,
        "event_count": len(support_events),
        "init_count": len({(event[0], event[2]) for event in support_events}),
        "branch_count": int(support_branch_count),
        "raw_event_count": sum(event[0] == task for event in all_events),
        "raw_branch_count": int(raw_branch_count),
        "excluded_event_count": sum(
            event[0] == task for event in excluded_keys
        ),
        "excluded_events": [
            _event_label(event)
            for event in sorted(excluded_keys, key=_event_sort)
            if event[0] == task
        ],
        "absolute": absolute,
        "paired_differences": paired,
        "prefixes": list(EXPECTED_PREFIXES),
        "recovery_actor_seeds": list(EXPECTED_SEEDS),
        "estimand": (
            "for each event and operator, mean over three recovery seeds "
            "within each fixed prefix then mean over five prefixes; mean over "
            "(task, init_state_id) clusters"
        ),
    }
    return task_summary, _dedupe_reasons(reasons)


def _action_stream_value(row: Mapping[str, Any]) -> tuple[str | None, Any]:
    for field in ACTION_STREAM_FIELDS:
        value = row.get(field)
        if value is not None and value != "":
            return field, _jsonable(value)
    return None, None


def _action_stream_check(
    support_groups: Mapping[
        tuple[int, int, int],
        Mapping[
            tuple[str, str, str, str, str],
            Mapping[tuple[str, int, str], Mapping[str, Any]],
        ],
    ],
    excluded_keys: set[tuple[str, str, str, str, str]],
    raw_groups: Mapping[
        tuple[int, int, int],
        Mapping[
            tuple[str, str, str, str, str],
            Mapping[tuple[str, int, str], Mapping[str, Any]],
        ],
    ] | None = None,
    *,
    max_evidence: int = 1000,
) -> dict[str, Any]:
    by_budget_task: dict[
        tuple[tuple[int, int, int], str],
        dict[tuple[tuple[str, str, str, str, str], int, int], dict[str, Any]],
    ] = defaultdict(lambda: defaultdict(dict))
    for budget, event_groups in support_groups.items():
        if budget[0] != 4:
            continue
        for event, branches in event_groups.items():
            if event in excluded_keys:
                continue
            for (operator, prefix, seed), row in branches.items():
                if operator in ("fresh_h4", "fresh_h16"):
                    by_budget_task[(budget, event[0])][
                        (event, prefix, int(seed))
                    ][operator] = row
    checked = 0
    safe_mismatches = 0
    hash_mismatches = 0
    missing_pairs: list[dict[str, Any]] = []
    unverifiable: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for (budget, task), pairs in sorted(
        by_budget_task.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        for (event, prefix, seed), operators in sorted(
            pairs.items(),
            key=lambda item: (_event_sort(item[0][0]), item[0][1], item[0][2]),
        ):
            left = operators.get("fresh_h4")
            right = operators.get("fresh_h16")
            identity = {
                **_event_label(event),
                "prefix_k": int(prefix),
                "recovery_actor_seed": str(seed),
                "budget": _budget_dict(budget),
            }
            if left is None or right is None:
                missing_pairs.append(identity)
                continue
            checked += 1
            safe_left = _float01(left["safe_success"])
            safe_right = _float01(right["safe_success"])
            safe_equal = safe_left == safe_right
            left_field, left_hash = _action_stream_value(left)
            right_field, right_hash = _action_stream_value(right)
            hash_available = left_field is not None and right_field is not None
            hash_equal = (
                hash_available
                and left_field == right_field
                and left_hash == right_hash
            )
            reasons: list[str] = []
            if not safe_equal:
                safe_mismatches += 1
                reasons.append("safe_success differs")
            if hash_available and not hash_equal:
                hash_mismatches += 1
                reasons.append("action stream hash differs")
            if not hash_available:
                unverifiable.append(
                    {
                        **identity,
                        "left_field": left_field,
                        "right_field": right_field,
                        "left_value": left_hash,
                        "right_value": right_hash,
                    }
                )
                reasons.append("action stream hash field missing on one or both rows")
            if reasons:
                mismatches.append(
                    {
                        **identity,
                        "safe_success": {
                            "fresh_h4": safe_left,
                            "fresh_h16": safe_right,
                            "equal": safe_equal,
                        },
                        "action_stream": {
                            "fresh_h4": {
                                "field": left_field,
                                "value": left_hash,
                            },
                            "fresh_h16": {
                                "field": right_field,
                                "value": right_hash,
                            },
                            "equal": hash_equal if hash_available else None,
                        },
                        "reasons": reasons,
                    }
                )
    excluded_pairs = 0
    if raw_groups is not None:
        for budget, event_groups in raw_groups.items():
            if budget[0] != 4:
                continue
            for event, branches in event_groups.items():
                if event in excluded_keys:
                    excluded_pairs += sum(
                        1
                        for branch in branches
                        if branch[0] in ("fresh_h4", "fresh_h16")
                    )
    if safe_mismatches or hash_mismatches:
        status = "MISMATCH"
    elif missing_pairs or unverifiable:
        status = "NOT_VERIFIABLE"
    else:
        status = "EQUIVALENT_OBSERVED"
    return {
        "status": status,
        "checked_pairs": int(checked),
        "safe_success_mismatch_count": int(safe_mismatches),
        "action_stream_hash_mismatch_count": int(hash_mismatches),
        "missing_pair_count": len(missing_pairs),
        "unverifiable_hash_count": len(unverifiable),
        "mismatches": mismatches[:max_evidence],
        "unverifiable": unverifiable[:max_evidence],
        "missing_pairs": missing_pairs[:max_evidence],
        "excluded_branch_pairs_skipped": int(excluded_pairs),
        "budgets_checked": [
            _budget_dict(budget)
            for budget in CONFIGURED_BUDGETS
            if budget[0] == 4
        ],
        "tasks_checked": list(TASKS),
        "compared_fields": list(ACTION_STREAM_FIELDS),
        "interpretation": (
            "EQUIVALENT_OBSERVED means every matched tail_horizon=4 "
            "fresh_h4/fresh_h16 row pair had equal safe_success and equal "
            "runtime action-stream field; it does not assume equivalence when "
            "a field or pair is missing."
        ),
    }


def _blocked(
    reasons: Iterable[str],
    *,
    input_root: Path | None = None,
    summary_status: Any = None,
    summary_read: bool = False,
    grid_read: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis": "phase1_operator_effects",
        "status": "BLOCKED",
        "blocked": True,
        "blocking_reasons": _dedupe_reasons(reasons) or ["unspecified blocker"],
        "evaluation_read": False,
        "references_read": False,
        "summary_read": bool(summary_read),
        "grid_rows_read": bool(grid_read),
        "summary_status": summary_status,
        "input_root": None if input_root is None else str(input_root),
        "selection_performed": False,
    }


def summarize_operator_effects(
    input_root: str | Path,
    output_path: str | Path | None = None,
    *,
    replicates: int = 10000,
    seed: int = 216214,
) -> dict[str, Any]:
    """Summarize all nine Phase-1 budgets for both tasks.

    The complete summary is opened first.  If it is incomplete, grid_rows is
    never opened, which prevents a running/partial matrix from being treated
    as science data.
    """
    try:
        grid_path, summary_path, phase_path = _resolve_input_paths(input_root)
    except (OSError, TypeError, ValueError, OperatorEffectsError) as exc:
        result = _blocked([str(exc)], input_root=Path(input_root))
        _write_result(result, output_path)
        return result
    summary, summary_reasons = _load_summary(summary_path)
    if summary is None or summary_reasons:
        result = _blocked(
            summary_reasons,
            input_root=phase_path,
            summary_status=None if summary is None else summary.get("status"),
            summary_read=True,
            grid_read=False,
        )
        _write_result(result, output_path)
        return result
    try:
        replicates = _int(replicates, "replicates")
        seed = _int(seed, "seed")
        if replicates <= 0:
            raise ValueError("replicates must be positive")
    except (TypeError, ValueError) as exc:
        result = _blocked(
            [str(exc)],
            input_root=phase_path,
            summary_status=summary.get("status"),
            summary_read=True,
            grid_read=False,
        )
        _write_result(result, output_path)
        return result
    excluded = summary.get("structurally_excluded_events", [])
    excluded_keys = set()
    for item in excluded:
        if isinstance(item, Mapping):
            try:
                excluded_keys.add(_event_key(item))
            except (KeyError, TypeError, ValueError):
                # _load_summary already reported this; keep the fail-closed
                # path explicit in case a caller mutates the mapping.
                pass
    (
        raw_rows,
        support_groups,
        raw_groups,
        observed_events,
        structural_events,
        row_reasons,
    ) = _load_rows(grid_path, excluded_keys)
    reasons = list(row_reasons)
    reasons.extend(
        _validate_inventory(
            summary,
            raw_rows,
            support_groups,
            raw_groups,
            observed_events,
            structural_events,
            excluded_keys,
        )
    )
    if reasons:
        result = _blocked(
            reasons,
            input_root=phase_path,
            summary_status=summary.get("status"),
            summary_read=True,
            grid_read=True,
        )
        _write_result(result, output_path)
        return result
    budgets_out: dict[str, Any] = {}
    for budget in CONFIGURED_BUDGETS:
        groups_for_budget = support_groups[budget]
        raw_for_budget = raw_groups[budget]
        all_events = sorted(
            {event for event in observed_events if event[0] in TASKS},
            key=_event_sort,
        )
        task_out: dict[str, Any] = {}
        for task in TASKS:
            task_summary, task_reasons = _task_budget_summary(
                budget,
                task,
                groups_for_budget,
                all_events,
                excluded_keys,
                raw_for_budget,
                replicates=replicates,
                seed=seed,
            )
            reasons.extend(task_reasons)
            if task_summary:
                task_out[task] = task_summary
        budgets_out[_budget_name(budget)] = {
            "budget": _budget_dict(budget),
            "by_task": task_out,
        }
    if reasons:
        result = _blocked(
            reasons,
            input_root=phase_path,
            summary_status=summary.get("status"),
            summary_read=True,
            grid_read=True,
        )
        _write_result(result, output_path)
        return result
    result = {
        "schema_version": 1,
        "analysis": "phase1_operator_effects",
        "status": "COMPLETE",
        "blocked": False,
        "split": "calibration",
        "input_root": str(phase_path),
        "grid_rows_path": str(grid_path),
        "summary_path": str(summary_path),
        "summary_status": summary.get("status"),
        "sample_complete": bool(summary.get("sample_complete", summary.get("planned_sample_complete"))),
        "planned_sample_complete": bool(summary.get("planned_sample_complete", summary.get("sample_complete"))),
        "selection_performed": False,
        "evaluation_read": False,
        "references_read": False,
        "configured_budgets": [
            _budget_dict(budget) for budget in CONFIGURED_BUDGETS
        ],
        "operators": list(OPERATORS),
        "prefixes": list(EXPECTED_PREFIXES),
        "recovery_actor_seeds": list(EXPECTED_SEEDS),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "budgets": budgets_out,
        "action_stream_check": _action_stream_check(
            support_groups, excluded_keys, raw_groups
        ),
        "structurally_excluded_events": [
            _event_label(event)
            for event in sorted(excluded_keys, key=_event_sort)
        ],
        "support_event_count_by_task": {
            task: sum(
                event[0] == task and event not in excluded_keys
                for event in observed_events
            )
            for task in TASKS
        },
        "raw_event_count_by_task": {
            task: sum(event[0] == task for event in observed_events)
            for task in TASKS
        },
        "estimand": (
            "Phase-1 calibration only; each event/arm mean averages three "
            "recovery seeds at five fixed prefixes, then init_state_id cluster "
            "means are equally weighted. BLOCKED structural horizon rows are "
            "excluded by whole event and never interpreted as safe_success=false."
        ),
    }
    _write_result(result, output_path)
    return result


def _write_result(result: Mapping[str, Any], output_path: str | Path | None) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(dict(result)), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        "--input",
        dest="input_root",
        type=Path,
        required=True,
        help="consolidated artifact root (contains phase1/)",
    )
    parser.add_argument(
        "--output",
        "--output-path",
        dest="output_path",
        type=Path,
        required=True,
    )
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=216214)
    args = parser.parse_args(argv)
    result = summarize_operator_effects(
        args.input_root,
        args.output_path,
        replicates=args.replicates,
        seed=args.seed,
    )
    print(
        json.dumps(
            {"status": result.get("status"), "output": str(args.output_path)},
            sort_keys=True,
        )
    )
    return 0 if result.get("status") != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
