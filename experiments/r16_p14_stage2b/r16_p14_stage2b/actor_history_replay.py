from __future__ import annotations

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from r16_p14_stage2a.envs import joint_qpos
from r16_p14_stage2a.settings import TASK_SPECS

from .io_utils import atomic_write_json, atomic_write_text, load_jsonl, write_once_jsonl
from .runtime import ActorBundle, branch_snapshot, reconstruct_to_prefix
from .settings import (
    ACTOR_SEEDS,
    ARTIFACT_ROOT,
    DETECTION_PREFIX,
    EXPERIMENT_ROOT,
    PERTURBATION_CALIBRATION_IDS,
    PRIMARY_TASKS,
)


def frozen_severities() -> dict[str, list[float]]:
    payload = json.loads((ARTIFACT_ROOT / "perturbation_qualification/frozen_parameters.json").read_text())
    return {
        task: [float(item["severity_m"]) for item in payload["tasks"][task]["severities"]]
        for task in PRIMARY_TASKS
    }


def prefix_positions(horizon: int) -> tuple[int, ...]:
    values = (
        DETECTION_PREFIX,
        min(DETECTION_PREFIX + 2, horizon),
        (DETECTION_PREFIX + horizon) // 2,
        horizon,
    )
    if len(set(values)) != 4:
        raise ValueError(f"replay contract requires four distinct prefixes, got {values}")
    return values


def select_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    shortfall: dict[str, Any] = {}
    for task in PRIMARY_TASKS:
        for seed in ACTOR_SEEDS:
            candidates = sorted(
                [
                    event
                    for event in events
                    if event["task"] == task
                    and int(event["actor_seed"]) == seed
                    and int(event["init_state_id"]) in PERTURBATION_CALIBRATION_IDS
                ],
                key=lambda event: int(event["init_state_id"]),
            )
            selected.extend(candidates[:3])
            if len(candidates) < 3:
                shortfall[f"{task}/seed{seed}"] = {"available": len(candidates), "required": 3}
    return selected, shortfall


def run_reconstruction(
    event: dict[str, Any],
    severity: float,
    prefix_k: int,
    repeat: int,
    order_index: int,
    bundle: ActorBundle,
) -> dict[str, Any]:
    context = reconstruct_to_prefix(
        event,
        severity_m=severity,
        prefix_k=prefix_k,
        bundle=bundle,
    )
    try:
        snapshot = branch_snapshot(context, bundle)
        spec = TASK_SPECS[event["task"]]
        assert spec.manipulated_joint
        return {
            "record_type": "fresh_actor_history_reconstruction",
            "task": event["task"],
            "event_id": event["event_id"],
            "actor_seed": event["actor_seed"],
            "init_state_id": event["init_state_id"],
            "severity_m": severity,
            "prefix_k": prefix_k,
            "repeat": repeat,
            "order_index": order_index,
            "fresh_environment": True,
            "anchor_replay_passed": context.anchor_replay["passed"],
            "anchor_replay_checks": context.anchor_replay["checks"],
            "anchor_state_error": context.anchor_replay["max_anchor_state_error"],
            "simulator_state": snapshot["state"].tolist(),
            "simulator_state_hash": snapshot["state_hash"],
            "state_history": snapshot["state_history"].tolist(),
            "state_history_hash": snapshot["state_history_hash"],
            "action_history": snapshot["action_history"].tolist(),
            "action_history_hash": snapshot["action_history_hash"],
            "contact_pairs": snapshot["contacts"],
            "contact_hash": snapshot["contacts_hash"],
            "object_qpos": joint_qpos(context.env, spec.manipulated_joint).tolist(),
            "gripper_action": float(snapshot["action_history"][-1, -1]),
            "task_phase": {
                "ever_lifted": context.tracker.ever_lifted,
                "cause_violation": context.tracker.violation,
                "task_success": snapshot["task_success"],
            },
            "original_chunk_hash": event["original_chunk_hash"],
            "original_chunk_hash_verified": context.anchor_replay["checks"]["original_chunk_hash_match"],
            "replanned_chunk": snapshot["replanned_chunk"].tolist(),
            "replanned_chunk_hash": snapshot["replanned_chunk_hash"],
            "cause_tracker": snapshot["cause_tracker"],
            "cause_tracker_hash": snapshot["cause_tracker_hash"],
            "outcome": {
                "cause_violation": snapshot["cause_tracker"]["violation"],
                "violation_type": snapshot["cause_tracker"]["violation_type"],
                "task_success": snapshot["task_success"],
            },
            "error": None,
        }
    finally:
        context.close()


def compare_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("error") is None]
    if len(valid) != 3:
        return {"passed": False, "valid_repeats": len(valid), "max_state_error": None}
    reference = valid[0]
    state_ref = np.asarray(reference["simulator_state"], dtype=np.float64)
    errors = [
        float(np.max(np.abs(np.asarray(row["simulator_state"], dtype=np.float64) - state_ref)))
        for row in valid
    ]
    checks = {
        "anchor_replay_match": all(row["anchor_replay_passed"] for row in valid),
        "state_hash_match": len({row["simulator_state_hash"] for row in valid}) == 1,
        "state_history_match": len({row["state_history_hash"] for row in valid}) == 1,
        "action_history_match": len({row["action_history_hash"] for row in valid}) == 1,
        "contact_match": len({row["contact_hash"] for row in valid}) == 1,
        "outcome_match": len({json.dumps(row["outcome"], sort_keys=True) for row in valid}) == 1,
        "original_chunk_match": all(row["original_chunk_hash_verified"] for row in valid),
        "replanned_chunk_match": len({row["replanned_chunk_hash"] for row in valid}) == 1,
        "cause_tracker_match": len({row["cause_tracker_hash"] for row in valid}) == 1,
        "max_state_error_le_1e_9": max(errors) <= 1e-9,
        "distinct_order_positions": len({row["order_index"] for row in valid}) == 3,
    }
    return {
        "task": reference["task"],
        "event_id": reference["event_id"],
        "actor_seed": reference["actor_seed"],
        "severity_m": reference["severity_m"],
        "prefix_k": reference["prefix_k"],
        "valid_repeats": len(valid),
        "max_state_error": max(errors),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = ARTIFACT_ROOT / "replay"
    raw_path = output / "raw_reconstructions.jsonl"
    if raw_path.is_file() and (output / "summary.json").is_file():
        print("ACTOR_HISTORY_REPLAY_ALREADY_COMPLETE")
        return
    events = load_jsonl(ARTIFACT_ROOT / "actor_events/events.jsonl")
    selected, shortfall = select_events(events)
    severities = frozen_severities()
    bundles = {seed: ActorBundle.load(seed, args.device) for seed in ACTOR_SEEDS}
    records: list[dict[str, Any]] = []
    for event in selected:
        horizon = int(event["effective_horizon"])
        positions = prefix_positions(horizon)
        orders = (
            positions,
            positions[1:] + positions[:1],
            positions[2:] + positions[:2],
        )
        for severity in severities[event["task"]]:
            for repeat, order in enumerate(orders):
                for order_index, prefix_k in enumerate(order):
                    try:
                        record = run_reconstruction(
                            event,
                            severity,
                            prefix_k,
                            repeat,
                            order_index,
                            bundles[int(event["actor_seed"])],
                        )
                    except Exception as exc:
                        record = {
                            "record_type": "fresh_actor_history_reconstruction",
                            "task": event["task"],
                            "event_id": event["event_id"],
                            "actor_seed": event["actor_seed"],
                            "init_state_id": event["init_state_id"],
                            "severity_m": severity,
                            "prefix_k": prefix_k,
                            "repeat": repeat,
                            "order_index": order_index,
                            "fresh_environment": True,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    records.append(record)
                    print(
                        f"HISTORY_REPLAY event={event['event_id']} severity={severity:.3f} "
                        f"k={prefix_k} repeat={repeat} ok={int(record.get('error') is None)}",
                        flush=True,
                    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["event_id"], record["severity_m"], record["prefix_k"])].append(record)
    groups = [compare_group(rows) for rows in grouped.values()]
    pass_rate = float(np.mean([group["passed"] for group in groups])) if groups else 0.0
    contact_outcome = float(
        np.mean(
            [
                group.get("checks", {}).get("contact_match", False)
                and group.get("checks", {}).get("outcome_match", False)
                for group in groups
            ]
        )
    ) if groups else 0.0
    max_error = max(
        [group["max_state_error"] for group in groups if group["max_state_error"] is not None],
        default=None,
    )
    no_order_dependence = all(
        group.get("checks", {}).get("distinct_order_positions", False) and group["passed"]
        for group in groups
    )
    checks = {
        "selection_shortfall_empty": not shortfall,
        "state_history_chunk_reconstruction_ge_0_99": pass_rate >= 0.99,
        "contact_outcome_agreement_eq_1": contact_outcome == 1.0,
        "no_branch_order_dependence": no_order_dependence,
        "max_numerical_state_error_le_1e_9": max_error is not None and max_error <= 1e-9,
    }
    status = "PASS" if all(checks.values()) else "BLOCKED"
    summary = {
        "schema_version": 1,
        "status": status,
        "failure_label": None if status == "PASS" else "BLOCKED_BY_ACTOR_HISTORY_REPLAY",
        "selected_event_count": len(selected),
        "selection_shortfall": shortfall,
        "reconstruction_record_count": len(records),
        "comparison_group_count": len(groups),
        "state_history_chunk_reconstruction_rate": pass_rate,
        "contact_outcome_agreement": contact_outcome,
        "max_numerical_state_error": max_error,
        "no_branch_order_dependence": no_order_dependence,
        "checks": checks,
        "early_stop_applied": False,
        "downstream_continuation_required_by_user": True,
        "groups": groups,
    }
    write_once_jsonl(raw_path, records)
    atomic_write_json(output / "summary.json", summary)
    lines = [
        "# Phase D — full actor-history replay gate",
        "",
        f"Status: **{status}** from {len(groups)} event/severity/prefix groups and {len(records)} fresh reconstructions.",
        "",
        f"- State/history/chunk reconstruction rate: `{pass_rate:.6f}`",
        f"- Contact/outcome agreement: `{contact_outcome:.6f}`",
        f"- Maximum numerical state error: `{max_error}`",
        f"- Branch-order dependence absent: `{no_order_dependence}`",
        f"- Selection shortfall: `{json.dumps(shortfall, sort_keys=True)}`",
        "",
    ]
    report = "\n".join(lines)
    atomic_write_text(output / "report.md", report)
    atomic_write_json(EXPERIMENT_ROOT / "replay/summary.json", summary)
    atomic_write_text(EXPERIMENT_ROOT / "replay/report.md", report)
    print(json.dumps({key: value for key, value in summary.items() if key != "groups"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
