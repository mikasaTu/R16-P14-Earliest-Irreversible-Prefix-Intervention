from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .fresh_process import run_spawned_branch
from .io_utils import atomic_write_json, atomic_write_jsonl
from .settings import ARTIFACT_ROOT, EXPERIMENT_ROOT, MIRROR_EXPERIMENT_OUTPUTS, TASKS, TARGET_SHIFT_TASK


PREFIXES = (2, 6, 16)
PERMUTATIONS = (
    ("CACHED_MATCHED", "CACHED_NOQUERY", "FRESH_MATCHED"),
    ("FRESH_MATCHED", "CACHED_MATCHED", "CACHED_NOQUERY"),
    ("CACHED_NOQUERY", "FRESH_MATCHED", "CACHED_MATCHED"),
)


def available_events() -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted((ARTIFACT_ROOT / "actor_events/shards").rglob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("event") is not None:
            events.append(payload["event"])
    return events


def select_smoke_events() -> dict[str, dict[str, Any]]:
    events = available_events()
    selected = {}
    for task in TASKS:
        choices = sorted(
            [event for event in events if event["task"] == task],
            key=lambda item: (
                0 if item["split"] == "infrastructure" else 1,
                int(item["actor_seed"]),
                int(item["init_state_id"]),
            ),
        )
        if not choices:
            raise RuntimeError(f"no strict actor event available for isolation smoke: {task}")
        selected[task] = choices[0]
    return selected


def smoke_parameter(event: dict[str, Any]) -> dict[str, Any]:
    if event["task"] == TARGET_SHIFT_TASK:
        return {"parameter_id": "shift_040mm", "magnitude_m": 0.04}
    future = int(event["predicted_path_indices"][0])
    return {
        "parameter_id": f"future_{future:02d}__clearance_p010",
        "future_index": future,
        "clearance_delta_m": 0.010,
    }


def branch_path(task: str, prefix_k: int, arm: str, repeat: int) -> Path:
    return (
        ARTIFACT_ROOT
        / "branch_isolation/shards"
        / f"{task}__k{prefix_k:02d}__{arm}__repeat{repeat}.json"
    )


def run(device: str) -> list[dict[str, Any]]:
    selected = select_smoke_events()
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        event = selected[task]
        parameter = smoke_parameter(event)
        for repeat, order in enumerate(PERMUTATIONS):
            for prefix_k in PREFIXES:
                for order_slot, arm in enumerate(order):
                    path = branch_path(task, prefix_k, arm, repeat)
                    if path.is_file():
                        row = json.loads(path.read_text())
                    else:
                        row = run_spawned_branch(
                            event=event,
                            parameter=parameter,
                            prefix_k=prefix_k,
                            arm=arm,
                            repeat=repeat,
                            device=device,
                        )
                        row["arm_order_slot"] = order_slot
                        row["arm_order"] = list(order)
                        atomic_write_json(path, row)
                    rows.append(row)
                    print(
                        f"ISOLATION task={task} k={prefix_k} arm={arm} repeat={repeat} "
                        f"error={int(bool(row.get('error')))}",
                        flush=True,
                    )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    error_rows = [row for row in rows if row.get("error")]
    process_ids = [row.get("pid") for row in rows if row.get("pid") is not None]
    reconstruction = [row.get("reconstruction", {}) for row in rows if not row.get("error")]
    same_action_checks: list[dict[str, Any]] = []
    for task in TASKS:
        for prefix_k in PREFIXES:
            for repeat in range(3):
                pair = {
                    row["effective_arm"]: row
                    for row in rows
                    if row.get("task") == task
                    and row.get("prefix_k") == prefix_k
                    and row.get("repeat") == repeat
                    and row.get("effective_arm") in {"CACHED_MATCHED", "CACHED_NOQUERY"}
                    and not row.get("error")
                }
                exact = len(pair) == 2 and (
                    pair["CACHED_MATCHED"]["pre_tail_signature"]["complete_signature_hash"]
                    == pair["CACHED_NOQUERY"]["pre_tail_signature"]["complete_signature_hash"]
                )
                cause_agreement = len(pair) == 2 and (
                    pair["CACHED_MATCHED"]["cause_violation"]
                    == pair["CACHED_NOQUERY"]["cause_violation"]
                    and pair["CACHED_MATCHED"]["S_obs_at_k"]
                    == pair["CACHED_NOQUERY"]["S_obs_at_k"]
                    and pair["CACHED_MATCHED"]["pre_tail_signature"]["contact_pairs_hash"]
                    == pair["CACHED_NOQUERY"]["pre_tail_signature"]["contact_pairs_hash"]
                )
                same_action_checks.append(
                    {
                        "task": task,
                        "prefix_k": prefix_k,
                        "repeat": repeat,
                        "signature_exact": exact,
                        "contact_cause_agreement": cause_agreement,
                    }
                )

    order_invariance: list[dict[str, Any]] = []
    order_metric_fields = (
        "final_signature_hash",
        "actual_actor_calls",
        "actual_post_detection_actions",
        "actual_inference_wall_time_s",
        "total_branch_wall_time_s",
        "eef_path_length_m",
        "manipulated_object_path_length_m",
        "progress_retained_end_m",
        "progress_regression_m",
        "contact_count",
        "S_obs_at_k",
        "cause_violation",
        "task_success",
        "safe_success",
    )
    def metric_invariant(field: str, values: list[Any]) -> bool:
        if len(values) != 3 or any(value is None for value in values):
            return False
        if field in {"actual_inference_wall_time_s", "total_branch_wall_time_s"}:
            numeric = [float(value) for value in values]
            # Wall time is measured evidence, not a deterministic simulator
            # state. Verify that every permutation produced a finite measured
            # value, and retain the raw values/range below; bit equality would
            # incorrectly turn ordinary OS scheduling jitter into a state
            # reconstruction failure.
            return all(math.isfinite(value) and value >= 0.0 for value in numeric)
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            return all(math.isclose(float(value), float(values[0]), rel_tol=1e-9, abs_tol=1e-9) for value in values[1:])
        return len(set(map(str, values))) == 1
    for task in TASKS:
        for prefix_k in PREFIXES:
            for arm in ("CACHED_MATCHED", "CACHED_NOQUERY", "FRESH_MATCHED"):
                matches = [
                    row
                    for row in rows
                    if row.get("task") == task
                    and row.get("prefix_k") == prefix_k
                    and row.get("effective_arm") == arm
                    and not row.get("error")
                ]
                signatures = {
                    row["pre_tail_signature"]["complete_signature_hash"] for row in matches
                }
                outcomes = {
                    (row["S_obs_at_k"], row["cause_violation"], row["task_success"])
                    for row in matches
                }
                metric_rows = [
                    {field: row.get(field) if field != "final_signature_hash" else row.get("final_signature", {}).get("complete_signature_hash") for field in order_metric_fields}
                    for row in matches
                ]
                metric_invariance = {
                    field: metric_invariant(field, [item[field] for item in metric_rows])
                    for field in order_metric_fields
                }
                order_invariance.append(
                    {
                        "task": task,
                        "prefix_k": prefix_k,
                        "arm": arm,
                        "repeat_count": len(matches),
                        "signature_invariant": len(matches) == 3 and len(signatures) == 1,
                        "outcome_invariant": len(matches) == 3 and len(outcomes) == 1,
                        "metric_fields": list(order_metric_fields),
                        "execution_metrics_invariant": all(metric_invariance.values()),
                        "metric_invariance": metric_invariance,
                        "metric_values": metric_rows,
                    }
                )

    monotonicity: list[dict[str, Any]] = []
    for task in TASKS:
        for repeat in range(3):
            for arm in ("CACHED_MATCHED", "CACHED_NOQUERY", "FRESH_MATCHED"):
                sequence = sorted(
                    [
                        (int(row["prefix_k"]), bool(row["S_obs_at_k"]))
                        for row in rows
                        if row.get("task") == task
                        and row.get("repeat") == repeat
                        and row.get("effective_arm") == arm
                        and not row.get("error")
                    ]
                )
                zero_to_one = any(not left[1] and right[1] for left, right in zip(sequence, sequence[1:]))
                monotonicity.append(
                    {
                        "task": task,
                        "repeat": repeat,
                        "arm": arm,
                        "sequence": sequence,
                        "zero_to_one": zero_to_one,
                    }
                )

    checks = {
        "expected_rows_54": len(rows) == 54,
        "no_missing_or_error_records": len(error_rows) == 0,
        "unique_spawned_process_per_branch": len(process_ids) == len(rows)
        and len(set(process_ids)) == len(process_ids)
        and all(row.get("unique_process_contract") for row in rows),
        "reconstruction_100_percent": len(reconstruction) == len(rows)
        and all(
            all(value for key, value in item.items() if key != "max_anchor_state_error")
            for item in reconstruction
        ),
        "maximum_state_error_le_1e_9": bool(reconstruction)
        and max(float(item["max_anchor_state_error"]) for item in reconstruction) <= 1e-9,
        "same_action_signature_100_percent": all(
            item["signature_exact"] for item in same_action_checks
        ),
        "contact_cause_agreement_100_percent": all(
            item["contact_cause_agreement"] for item in same_action_checks
        ),
        "no_S_obs_zero_to_one": not any(item["zero_to_one"] for item in monotonicity),
        "branch_order_invariant": all(
            item["signature_invariant"]
            and item["outcome_invariant"]
            and item["execution_metrics_invariant"]
            for item in order_invariance
        ),
        "actor_inference_side_effect_free": all(
            row.get("actor_inference_side_effect_free") for row in rows if not row.get("error")
        ),
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "BLOCKED_BY_BRANCH_ISOLATION",
        "checks": checks,
        "row_count": len(rows),
        "error_count": len(error_rows),
        "maximum_state_error": max(
            [float(item["max_anchor_state_error"]) for item in reconstruction] or [float("inf")]
        ),
        "same_action_checks": same_action_checks,
        "order_invariance": order_invariance,
        "monotonicity": monotonicity,
        "diagnostic_continuation_if_failed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    output = ARTIFACT_ROOT / "branch_isolation"
    if args.summarize_only:
        rows = [json.loads(path.read_text()) for path in sorted((output / "shards").glob("*.json"))]
    else:
        rows = run(args.device)
    summary = summarize(rows)
    atomic_write_jsonl(output / "raw.jsonl", rows)
    atomic_write_jsonl(
        output / "signatures.jsonl",
        [
            {
                "event_id": row.get("event_id"),
                "task": row.get("task"),
                "prefix_k": row.get("prefix_k"),
                "arm": row.get("effective_arm"),
                "repeat": row.get("repeat"),
                "pid": row.get("pid"),
                "detection_signature": row.get("detection_signature"),
                "pre_tail_signature": row.get("pre_tail_signature"),
                "error": row.get("error"),
            }
            for row in rows
        ],
    )
    atomic_write_json(output / "summary.json", summary)
    report = (
        "# Fresh-process branch-isolation gate\n\n"
        f"Status: **{summary['status']}**. Each of {len(rows)} smoke branches ran in a unique "
        "`spawn` process with a newly created LIBERO environment and newly loaded frozen ACT.\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in summary["checks"].items())
        + "\n"
    )
    (output / "report.md").write_text(report)
    if MIRROR_EXPERIMENT_OUTPUTS:
        mirror = EXPERIMENT_ROOT / "branch_isolation"
        mirror.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "report.md"):
            (mirror / name).write_bytes((output / name).read_bytes())
    print(json.dumps({"status": summary["status"], "checks": summary["checks"]}, sort_keys=True))


if __name__ == "__main__":
    main()
