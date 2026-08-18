from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as py_statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .fresh_process import run_spawned_branch
from .io_utils import atomic_write_json, atomic_write_jsonl, load_jsonl
from .settings import (
    ARTIFACT_ROOT,
    CALIBRATION_ARMS,
    DETECTION_PREFIX,
    EXPERIMENT_ROOT,
    H_VALID,
    DIAGNOSTIC_ONLY_GLOBAL,
    FORMAL_POSITIVE_EVIDENCE_ALLOWED,
    MIRROR_EXPERIMENT_OUTPUTS,
    PREFIX_INDICES,
    TASKS,
)


def pool_and_events() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    pool = [
        item
        for item in load_jsonl(ARTIFACT_ROOT / "actor_events/formal_event_pool.jsonl")
        if item["split"] == "calibration"
    ]
    events = {
        event["event_id"]: event
        for event in load_jsonl(ARTIFACT_ROOT / "actor_events/events.jsonl")
    }
    return pool, events


def key(item: dict[str, Any], prefix_k: int, arm: str) -> str:
    return f"{item['event_instance_id']}__k{prefix_k:02d}__{arm}"


def shard_path(item: dict[str, Any], prefix_k: int, arm: str) -> Path:
    return ARTIFACT_ROOT / "calibration_atlas/shards" / f"{key(item, prefix_k, arm)}.json"


def upstream_pass() -> bool:
    isolation = json.loads((ARTIFACT_ROOT / "branch_isolation/summary.json").read_text())
    qualification = json.loads(
        (ARTIFACT_ROOT / "perturbation_qualification/frozen_parameters.json").read_text()
    )
    event_summary = json.loads((ARTIFACT_ROOT / "actor_events/summary.json").read_text())
    availability = event_summary.get("availability_gate", {}).get("calibration", {})
    return (
        isolation["status"] == "PASS"
        and qualification["status"] == "PASS"
        and all(availability.get(task, False) for task in TASKS)
    )


def run_worker(
    worker_index: int,
    worker_count: int,
    device: str,
    max_new_shards: int | None = None,
) -> None:
    pool, events = pool_and_events()
    diagnostic = not upstream_pass()
    new_shards = 0
    for item in pool:
        event = events[item["event_id"]]
        for prefix_k in PREFIX_INDICES:
            for arm in CALIBRATION_ARMS:
                branch_key = key(item, prefix_k, arm)
                if int(hashlib.sha256(branch_key.encode()).hexdigest(), 16) % worker_count != worker_index:
                    continue
                path = shard_path(item, prefix_k, arm)
                if path.is_file():
                    print(f"ATLAS_RESUME {branch_key}", flush=True)
                    continue
                row = run_spawned_branch(
                    event=event,
                    parameter=item["parameter"],
                    prefix_k=prefix_k,
                    arm=arm,
                    repeat=0,
                    device=device,
                )
                row["event_instance_id"] = item["event_instance_id"]
                row["diagnostic_only"] = diagnostic
                row["formal_positive_evidence_allowed"] = not diagnostic
                atomic_write_json(path, row)
                new_shards += 1
                print(f"ATLAS_DONE {branch_key} error={int(bool(row.get('error')))}", flush=True)
                if max_new_shards is not None and new_shards >= max_new_shards:
                    return


def objective(row: dict[str, Any]) -> tuple[Any, ...]:
    completion = row["completion_steps"]
    return (
        -int(bool(row["safe_success"])),
        int(bool(row["cause_violation"])),
        int(row["actual_post_detection_actions"]),
        math.inf if completion is None else int(completion),
        int(row["actual_actor_calls"]),
        int(row["prefix_k"]),
    )


def ratio_reduction(new: list[float], old: list[float]) -> float:
    old_mean = float(np.mean(old)) if old else 0.0
    return 0.0 if old_mean <= 0 else float((old_mean - float(np.mean(new))) / old_mean)


def consolidate() -> dict[str, Any]:
    pool, _ = pool_and_events()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for item in pool:
        for prefix_k in PREFIX_INDICES:
            for arm in CALIBRATION_ARMS:
                path = shard_path(item, prefix_k, arm)
                if path.is_file():
                    rows.append(json.loads(path.read_text()))
                else:
                    missing.append(str(path.relative_to(ARTIFACT_ROOT)))
    output = ARTIFACT_ROOT / "calibration_atlas"
    expected_rows = len(pool) * len(PREFIX_INDICES) * len(CALIBRATION_ARMS)
    complete_matrix = not missing and len(rows) == expected_rows
    terminal_path = output / "terminal_receipt.json"
    terminal_receipt = json.loads(terminal_path.read_text()) if terminal_path.is_file() else {
        "status": "MISSING",
        "complete_matrix": False,
        "source": "no terminal PAI receipt was imported before consolidation",
    }
    atomic_write_jsonl(output / "rows.jsonl", rows)
    valid = [row for row in rows if not row.get("error")]
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_event[row["event_instance_id"]].append(row)
    oracle_rows: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []
    for item in pool:
        event_rows = by_event[item["event_instance_id"]]
        expected_event_rows = len(PREFIX_INDICES) * len(CALIBRATION_ARMS)
        if len(event_rows) != expected_event_rows:
            # Never compute an oracle from a partial event.  The raw shards
            # remain available for a diagnostic missing-shard audit.
            continue
        cached = [row for row in event_rows if row["requested_arm"] == "CACHED_MATCHED"]
        if not cached:
            continue
        oracle = min(cached, key=objective)
        s_rows = sorted(cached, key=lambda row: int(row["requested_prefix_k"]))
        safe_prefixes = [
            int(row["requested_prefix_k"]) for row in s_rows if bool(row["S_obs_at_k"])
        ]
        last_safe = max(safe_prefixes) if safe_prefixes else DETECTION_PREFIX - 1
        safe_window = max(0, last_safe - DETECTION_PREFIX)
        oracle_rows.append(
            {
                "event_instance_id": item["event_instance_id"],
                "event_id": item["event_id"],
                "task": item["task"],
                "actor_seed": item["actor_seed"],
                "parameter_id": item["parameter_id"],
                "oracle_k": int(oracle["requested_prefix_k"]),
                "oracle_objective": list(objective(oracle)),
                "safe_success": oracle["safe_success"],
                "cause_violation": oracle["cause_violation"],
                "actual_post_detection_actions": oracle["actual_post_detection_actions"],
                "actual_actor_calls": oracle["actual_actor_calls"],
                "observed_safe_window": safe_window,
                "last_observed_safe_prefix": last_safe,
                "interior_safe_prefix": any(
                    DETECTION_PREFIX + 2 <= value <= H_VALID - 1 for value in safe_prefixes
                ),
            }
        )
        event_summaries.append(
            {
                "event_instance_id": item["event_instance_id"],
                "row_count": len(event_rows),
                "all_arms_complete": len(event_rows) == len(PREFIX_INDICES) * len(CALIBRATION_ARMS),
            }
        )
    atomic_write_jsonl(output / "oracle_rows.jsonl", oracle_rows)

    h1_by_task = {}
    oracle_by_task = {}
    for task in TASKS:
        task_oracle = [row for row in oracle_rows if row["task"] == task]
        windows = [row["observed_safe_window"] for row in task_oracle]
        h1_by_task[task] = {
            "events": len(task_oracle),
            "median_observed_safe_window": float(py_statistics.median(windows)) if windows else 0.0,
            "interior_safe_fraction": (
                sum(row["interior_safe_prefix"] for row in task_oracle) / len(task_oracle)
                if task_oracle
                else 0.0
            ),
        }
        oracle_by_task[task] = {
            "oracle_not_d_fraction": (
                sum(row["oracle_k"] != DETECTION_PREFIX for row in task_oracle) / len(task_oracle)
                if task_oracle
                else 0.0
            ),
            "oracle_not_H_fraction": (
                sum(row["oracle_k"] != H_VALID for row in task_oracle) / len(task_oracle)
                if task_oracle
                else 0.0
            ),
        }
    h1_pass = all(
        h1_by_task[task]["median_observed_safe_window"] >= 2
        and h1_by_task[task]["interior_safe_fraction"] >= 0.30
        for task in TASKS
    )
    interior_oracle = all(
        oracle_by_task[task]["oracle_not_d_fraction"] >= 0.30
        and oracle_by_task[task]["oracle_not_H_fraction"] >= 0.30
        for task in TASKS
    )

    # Compare the cached-prefix oracle to the strongest fixed cached delay.
    fixed_ks = (DETECTION_PREFIX, DETECTION_PREFIX + 2, DETECTION_PREFIX + 4, DETECTION_PREFIX + 8, H_VALID)
    fixed_metrics = []
    for fixed_k in fixed_ks:
        fixed_rows = [
            row
            for row in valid
            if row["requested_arm"] == "CACHED_MATCHED" and int(row["requested_prefix_k"]) == fixed_k
        ]
        fixed_metrics.append(
            {
                "k": fixed_k,
                "safe_success": float(np.mean([row["safe_success"] for row in fixed_rows])) if fixed_rows else 0.0,
                "cause_violation": float(np.mean([row["cause_violation"] for row in fixed_rows])) if fixed_rows else 1.0,
                "post_actions": float(np.mean([row["actual_post_detection_actions"] for row in fixed_rows])) if fixed_rows else math.inf,
            }
        )
    strongest = min(fixed_metrics, key=lambda row: (-row["safe_success"], row["cause_violation"], row["post_actions"], row["k"]))
    oracle_safe = float(np.mean([row["safe_success"] for row in oracle_rows])) if oracle_rows else 0.0
    oracle_actions = float(np.mean([row["actual_post_detection_actions"] for row in oracle_rows])) if oracle_rows else math.inf
    safe_gain = oracle_safe - strongest["safe_success"]
    action_reduction = (
        (strongest["post_actions"] - oracle_actions) / strongest["post_actions"]
        if strongest["post_actions"] not in (0.0, math.inf)
        else 0.0
    )
    severity_signal = {}
    for task in TASKS:
        severity_signal[task] = {}
        ids = sorted({row["parameter_id"] for row in oracle_rows if row["task"] == task})
        for parameter_id in ids:
            subset = [row for row in oracle_rows if row["task"] == task and row["parameter_id"] == parameter_id]
            severity_signal[task][parameter_id] = {
                "events": len(subset),
                "nontrivial_oracle_fraction": sum(row["oracle_k"] not in (DETECTION_PREFIX, H_VALID) for row in subset) / len(subset) if subset else 0.0,
            }
    both_tasks_severities = all(
        len(severity_signal[task]) >= 2
        and all(value["events"] > 0 and value["nontrivial_oracle_fraction"] > 0 for value in severity_signal[task].values())
        for task in TASKS
    )
    oracle_gate = bool(
        h1_pass
        and interior_oracle
        and (safe_gain >= 0.10 or (safe_gain >= -0.03 and action_reduction >= 0.15))
        and both_tasks_severities
    )
    summary = {
        "schema_version": 1,
        "status": "PASS" if complete_matrix and not any(row.get("error") for row in rows) else "INCOMPLETE_OR_ERRORS",
        "diagnostic_only": DIAGNOSTIC_ONLY_GLOBAL or not upstream_pass(),
        "formal_positive_evidence_allowed": FORMAL_POSITIVE_EVIDENCE_ALLOWED,
        "expected_rows": expected_rows,
        "completed_rows": len(rows),
        "valid_rows": len(valid),
        "error_count": sum(bool(row.get("error")) for row in rows),
        "missing_shards": missing,
        "complete_matrix": complete_matrix,
        "statistics_eligible": bool(complete_matrix and not any(row.get("error") for row in rows) and terminal_receipt.get("status") == "SUCCEEDED"),
        "terminal_receipt": terminal_receipt,
        "event_summaries": event_summaries,
        "h1_by_task": h1_by_task,
        "h1_raw_pass": h1_pass,
        "oracle_interior_by_task": oracle_by_task,
        "strongest_fixed_cached_baseline": strongest,
        "oracle_safe_success": oracle_safe,
        "oracle_safe_success_gain": safe_gain,
        "oracle_post_action_reduction": action_reduction,
        "severity_signal": severity_signal,
        "oracle_mechanism_raw_pass": oracle_gate,
        # Keep the preregistered decision vocabulary exact. A raw signal behind
        # a failed upstream gate is retained above for diagnosis, but the formal
        # oracle mechanism was not adjudicated and therefore remains NOT_RUN.
        "oracle_mechanism_decision": (
            "PASS"
            if oracle_gate and upstream_pass() and not DIAGNOSTIC_ONLY_GLOBAL
            else "NOT_RUN"
            if oracle_gate
            else "NO_ORACLE_GAP"
        ),
    }
    atomic_write_json(output / "summary.json", summary)
    (output / "report.md").write_text(
        "# Calibration oracle atlas\n\n"
        f"Rows: {len(rows)} / {summary['expected_rows']}; errors: {summary['error_count']}; "
        f"diagnostic-only: {summary['diagnostic_only']}.\n\n"
        f"Raw H1 pass: {h1_pass}; raw oracle mechanism gate: {oracle_gate}. "
        "The oracle is an upper bound and is not a deployable selector.\n"
    )
    if MIRROR_EXPERIMENT_OUTPUTS:
        mirror = EXPERIMENT_ROOT / "calibration_atlas"
        mirror.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "report.md"):
            (mirror / name).write_bytes((output / name).read_bytes())
    print(json.dumps({"status": summary["status"], "oracle_gate": oracle_gate, "diagnostic_only": summary["diagnostic_only"]}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-shards", type=int)
    parser.add_argument("--consolidate", action="store_true")
    args = parser.parse_args()
    if args.consolidate:
        consolidate()
    else:
        run_worker(
            args.worker_index,
            args.worker_count,
            args.device,
            args.max_new_shards,
        )


if __name__ == "__main__":
    main()
