from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics as py_statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .fresh_process import run_spawned_branch
from .io_utils import atomic_write_json, atomic_write_jsonl, load_jsonl
from .settings import (
    ACTOR_SEEDS,
    ARTIFACT_ROOT,
    EXPERIMENT_ROOT,
    FORCED_DIAGNOSTIC_CONTINUATION,
    H_VALID,
    MIRROR_EXPERIMENT_OUTPUTS,
    PATH_CLEARANCE_DELTAS,
    PATH_FUTURE_INDICES,
    TARGET_SHIFT_GRID,
    TARGET_SHIFT_TASK,
    TASKS,
)


def parameters(task: str) -> list[dict[str, Any]]:
    if task == TARGET_SHIFT_TASK:
        return [
            {"parameter_id": f"shift_{round(magnitude * 1000):03d}mm", "magnitude_m": magnitude}
            for magnitude in TARGET_SHIFT_GRID
        ]
    return [
        {
            "parameter_id": (
                f"future_{future:02d}__clearance_"
                f"{'m' if clearance < 0 else 'p'}{round(abs(clearance) * 1000):03d}mm"
            ),
            "future_index": future,
            "clearance_delta_m": clearance,
        }
        for future in PATH_FUTURE_INDICES
        for clearance in PATH_CLEARANCE_DELTAS
    ]


def formal_events() -> list[dict[str, Any]]:
    return [event for event in load_jsonl(ARTIFACT_ROOT / "actor_events/events.jsonl") if event["split"] == "calibration"]


def assignment_key(event: dict[str, Any], parameter: dict[str, Any]) -> str:
    return f"{event['event_id']}__{parameter['parameter_id']}"


def shard_path(event: dict[str, Any], parameter: dict[str, Any]) -> Path:
    return (
        ARTIFACT_ROOT
        / "perturbation_qualification/shards"
        / event["task"]
        / f"{assignment_key(event, parameter)}.json"
    )


def run_worker(worker_index: int, worker_count: int, device: str) -> None:
    events = formal_events()
    for event in events:
        for parameter in parameters(event["task"]):
            key = assignment_key(event, parameter)
            if int(hashlib.sha256(key.encode()).hexdigest(), 16) % worker_count != worker_index:
                continue
            path = shard_path(event, parameter)
            if path.is_file():
                print(f"QUAL_RESUME {key}", flush=True)
                continue
            row = run_spawned_branch(
                event=event,
                parameter=parameter,
                prefix_k=H_VALID,
                arm="FULL_OLD_CHUNK",
                repeat=0,
                device=device,
            )
            row["qualification_only"] = True
            row["method_outcomes_inspected_for_parameter_selection"] = False
            atomic_write_json(path, row)
            print(f"QUAL_DONE {key} error={int(bool(row.get('error')))}", flush=True)


def gate_distance(metrics: dict[str, Any]) -> float:
    distance = 0.0
    distance += max(0.0, 20 - metrics["valid_events"]) / 20
    distance += metrics["injection_contact_rate"] * 4
    distance += max(0.0, metrics["immediate_violation_rate"] - 0.05) * 4
    delayed = metrics["delayed_violation_rate"]
    distance += max(0.0, 0.30 - delayed) + max(0.0, delayed - 0.80)
    distance += max(0.0, 3 - metrics["median_first_violation_offset"]) / 3
    distance += max(0.0, 0.30 - metrics["interior_violation_fraction"])
    distance += 1.0 - metrics["replay_rate"]
    distance += min(1.0, metrics["error_count"])
    distance += min(1.0, metrics["nonmonotonic_event_count"])
    return float(distance)


def cell_metrics(rows: list[dict[str, Any]], parameter: dict[str, Any]) -> dict[str, Any]:
    errors = [row for row in rows if row.get("error")]
    valid = [row for row in rows if not row.get("error")]
    delayed = [row for row in valid if row["cause_violation"] and not row["immediate_violation"]]
    offsets = [int(row["first_violation_offset"]) for row in delayed if row["first_violation_offset"] is not None]
    interior = [offset for offset in offsets if 3 <= offset <= H_VALID - 4]
    seed_counts = Counter(int(row["actor_seed"]) for row in valid)
    replay = [
        row
        for row in valid
        if all(
            bool(value)
            for key, value in row["reconstruction"].items()
            if key != "max_anchor_state_error"
        )
        and float(row["reconstruction"]["max_anchor_state_error"]) <= 1e-9
    ]
    monotonic_count = 0
    for row in valid:
        offset = row["first_violation_offset"]
        values = [offset is None or offset > k - 2 for k in range(2, H_VALID + 1)]
        monotonic_count += int(any(not left and right for left, right in zip(values, values[1:])))
    metrics = {
        **parameter,
        "attempted_events": len(rows),
        "valid_events": len(valid),
        "actor_seed_counts": {str(seed): seed_counts[seed] for seed in ACTOR_SEEDS},
        "contributing_seeds_ge5": sum(seed_counts[seed] >= 5 for seed in ACTOR_SEEDS),
        "injection_contact_rate": sum(bool(row["injection_contact"]) for row in valid) / len(valid) if valid else 1.0,
        "immediate_violation_rate": sum(bool(row["immediate_violation"]) for row in valid) / len(valid) if valid else 1.0,
        "delayed_violation_rate": len(delayed) / len(valid) if valid else 0.0,
        "median_first_violation_offset": float(py_statistics.median(offsets)) if offsets else 0.0,
        "interior_violation_fraction": len(interior) / len(offsets) if offsets else 0.0,
        "replay_rate": len(replay) / len(rows) if rows else 0.0,
        "error_count": len(errors),
        "nonmonotonic_event_count": monotonic_count,
    }
    checks = {
        "valid_events_ge20": metrics["valid_events"] >= 20,
        "two_actor_seeds_ge5": metrics["contributing_seeds_ge5"] >= 2,
        "injection_contact_rate_zero": metrics["injection_contact_rate"] == 0.0,
        "immediate_violation_le005": metrics["immediate_violation_rate"] <= 0.05,
        "delayed_violation_in_030_080": 0.30 <= metrics["delayed_violation_rate"] <= 0.80,
        "median_first_violation_offset_ge3": metrics["median_first_violation_offset"] >= 3,
        "interior_violation_fraction_ge030": metrics["interior_violation_fraction"] >= 0.30,
        "replay_100_percent": metrics["replay_rate"] == 1.0,
        "error_count_zero": metrics["error_count"] == 0,
        "nonmonotonic_count_zero": metrics["nonmonotonic_event_count"] == 0,
    }
    metrics["checks"] = checks
    metrics["qualified"] = all(checks.values())
    metrics["gate_distance"] = gate_distance(metrics)
    return metrics


def consolidate() -> dict[str, Any]:
    events = formal_events()
    expected: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (event, parameter) for event in events for parameter in parameters(event["task"])
    ]
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for event, parameter in expected:
        path = shard_path(event, parameter)
        if path.is_file():
            rows.append(json.loads(path.read_text()))
        else:
            missing.append(str(path.relative_to(ARTIFACT_ROOT)))
    output = ARTIFACT_ROOT / "perturbation_qualification"
    atomic_write_jsonl(output / "raw_attempts.jsonl", rows)
    prefix_rows = []
    for row in rows:
        if row.get("error"):
            continue
        offset = row["first_violation_offset"]
        for k in range(2, H_VALID + 1):
            prefix_rows.append(
                {
                    "event_id": row["event_id"],
                    "task": row["task"],
                    "actor_seed": row["actor_seed"],
                    "parameter_id": row["parameter_id"],
                    "prefix_k": k,
                    "S_obs": bool(offset is None or offset > k - 2),
                    "source": "absorbing_full_old_trajectory",
                }
            )
    atomic_write_jsonl(output / "prefix_safety.jsonl", prefix_rows)
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[(row["task"], row["parameter"]["parameter_id"])].append(row)
    table: list[dict[str, Any]] = []
    selected: dict[str, list[dict[str, Any]]] = {}
    family_status: dict[str, str] = {}
    for task in TASKS:
        task_metrics = []
        for parameter in parameters(task):
            metrics = cell_metrics(by_cell[(task, parameter["parameter_id"])], parameter)
            metrics["task"] = task
            table.append(metrics)
            task_metrics.append(metrics)
        qualified = sorted(
            [item for item in task_metrics if item["qualified"]], key=lambda item: item["parameter_id"]
        )
        if len(qualified) >= 2:
            chosen = qualified[:2]
            family_status[task] = "PASS"
        else:
            chosen = sorted(task_metrics, key=lambda item: (item["gate_distance"], item["parameter_id"]))[:2]
            family_status[task] = "BLOCKED"
        selected[task] = [
            {key: value for key, value in item.items() if key in parameter_keys(task)}
            | {
                "qualified": bool(item["qualified"]),
                "diagnostic_fallback": not bool(item["qualified"]),
                "gate_distance": item["gate_distance"],
            }
            for item in chosen
        ]

    csv_path = output / "grid_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    flattened = []
    for item in table:
        flattened.append(
            {
                key: (json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value)
                for key, value in item.items()
            }
        )
    if flattened:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in flattened for key in row}))
            writer.writeheader()
            writer.writerows(flattened)
    overall_pass = not missing and all(value == "PASS" for value in family_status.values())
    frozen = {
        "schema_version": 1,
        "status": "PASS" if overall_pass else "BLOCKED_BY_PERTURBATION_QUALIFICATION",
        "tasks": {task: {"status": family_status[task], "parameters": selected[task]} for task in TASKS},
        "selection_rule": "first two qualified cells in preregistered lexical order",
        "diagnostic_fallback_rule": "two minimum qualification-gate-distance cells; no positive label permitted",
        "method_outcomes_read": False,
        "forced_diagnostic_continuation": FORCED_DIAGNOSTIC_CONTINUATION,
    }
    atomic_write_json(output / "frozen_parameters.json", frozen)
    summary = {
        "schema_version": 1,
        "status": frozen["status"],
        "expected_branches": len(expected),
        "completed_branches": len(rows),
        "missing_shards": missing,
        "family_status": family_status,
        "selected": selected,
        "grid": table,
        "diagnostic_continuation": not overall_pass and FORCED_DIAGNOSTIC_CONTINUATION,
    }
    atomic_write_json(output / "summary.json", summary)
    negative = [item for item in table if not item["qualified"]]
    (output / "negative_results.md").write_text(
        "# Qualification negative cells\n\n"
        + "\n".join(
            f"- `{item['task']} / {item['parameter_id']}`: failed "
            f"{[key for key, value in item['checks'].items() if not value]}"
            for item in negative
        )
        + "\n"
    )
    (output / "report.md").write_text(
        "# Event-aligned perturbation qualification\n\n"
        f"Status: **{frozen['status']}**. Qualification read only opportunity, injection, "
        "old-chunk cause-onset, replay, and monotonicity fields; no method outcome was loaded.\n\n"
        + "\n".join(f"- {task}: {family_status[task]}" for task in TASKS)
        + ("\n\nDownstream work is diagnostic-only under the explicit user override.\n" if not overall_pass else "\n")
    )
    if MIRROR_EXPERIMENT_OUTPUTS:
        mirror = EXPERIMENT_ROOT / "perturbation_qualification"
        mirror.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "frozen_parameters.json", "negative_results.md", "report.md"):
            (mirror / name).write_bytes((output / name).read_bytes())
    print(json.dumps({"status": summary["status"], "family_status": family_status}, sort_keys=True))
    return summary


def parameter_keys(task: str) -> set[str]:
    return (
        {"parameter_id", "magnitude_m"}
        if task == TARGET_SHIFT_TASK
        else {"parameter_id", "future_index", "clearance_delta_m"}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--consolidate", action="store_true")
    args = parser.parse_args()
    if args.consolidate:
        consolidate()
    else:
        run_worker(args.worker_index, args.worker_count, args.device)


if __name__ == "__main__":
    main()
