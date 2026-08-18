from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import atomic_write_json, load_jsonl, sha256_file
from .settings import (
    ARTIFACT_ROOT,
    BOOTSTRAP_SEED,
    DETECTION_PREFIX,
    EXPERIMENT_ROOT,
    H_VALID,
    MIRROR_EXPERIMENT_OUTPUTS,
    SAFETY_NONINFERIORITY_EPSILON,
    TAIL_HORIZON,
    TARGET_SHIFT_TASK,
    DIAGNOSTIC_ONLY_GLOBAL,
    FORMAL_POSITIVE_EVIDENCE_ALLOWED,
)


def rule_k(event: dict[str, Any], parameter: dict[str, Any]) -> int:
    if event["task"] == TARGET_SHIFT_TASK:
        source_index = int(event["release_index"])
    else:
        source_index = int(parameter["future_index"])
    return int(np.clip(source_index - 2, DETECTION_PREFIX, H_VALID))


def main() -> None:
    evaluation_shards = ARTIFACT_ROOT / "confirmatory_evaluation/shards"
    if evaluation_shards.exists() and any(evaluation_shards.rglob("*.json")):
        raise RuntimeError("evaluation outcomes already exist; rule freeze is outcome contaminated")
    calibration = json.loads((ARTIFACT_ROOT / "calibration_atlas/summary.json").read_text())
    if not calibration.get("complete_matrix") or not calibration.get("terminal_receipt", {}).get("status") == "SUCCEEDED":
        raise RuntimeError("calibration atlas must have a terminal receipt and complete shards before rule freeze")
    rows = [row for row in load_jsonl(ARTIFACT_ROOT / "calibration_atlas/rows.jsonl") if not row.get("error")]
    pool = [
        item
        for item in load_jsonl(ARTIFACT_ROOT / "actor_events/formal_event_pool.jsonl")
        if item["split"] == "calibration"
    ]
    events = {
        event["event_id"]: event
        for event in load_jsonl(ARTIFACT_ROOT / "actor_events/events.jsonl")
    }
    selection_rows = [
        {
            "event_instance_id": item["event_instance_id"],
            "task": item["task"],
            "actor_seed": item["actor_seed"],
            "init_state_id": item["init_state_id"],
            "parameter_id": item["parameter_id"],
            "k_rule": rule_k(events[item["event_id"]], item["parameter"]),
            "rule_source": "release_index_minus_2" if item["task"] == TARGET_SHIFT_TASK else "predicted_path_intersection_index_minus_2",
            "outcome_fields_read": False,
        }
        for item in pool
    ]
    candidate_delays = (2, 4, 8)
    fixed = []
    for delay in candidate_delays:
        k = DETECTION_PREFIX + delay
        subset = [
            row
            for row in rows
            if row["requested_arm"] == "CACHED_MATCHED"
            and int(row["requested_prefix_k"]) == k
        ]
        fixed.append(
            {
                "method": f"FIXED_DELAY_{delay}",
                "delay": delay,
                "prefix_k": k,
                "events": len(subset),
                "safe_success": float(np.mean([row["safe_success"] for row in subset])) if subset else 0.0,
                "cause_violation": float(np.mean([row["cause_violation"] for row in subset])) if subset else 1.0,
                "actual_post_detection_actions": float(np.mean([row["actual_post_detection_actions"] for row in subset])) if subset else float("inf"),
            }
        )
    strongest = min(
        fixed,
        key=lambda item: (
            -item["safe_success"],
            item["cause_violation"],
            item["actual_post_detection_actions"],
            item["delay"],
        ),
    )
    disagreements = [
        float(row["cached_fresh_action_disagreement"])
        for row in rows
        if row["requested_arm"] == "CACHED_MATCHED"
        and row.get("cached_fresh_action_disagreement") is not None
    ]
    threshold = float(np.quantile(disagreements, 0.75)) if disagreements else None
    output = ARTIFACT_ROOT / "frozen_rule"
    rule = {
        "schema_version": 1,
        "frozen_from_split": "calibration",
        "target_shift": "clip(release_index - 2, d, H)",
        "path_obstacle": "clip(predicted_path_intersection_index - 2, d, H)",
        "d": DETECTION_PREFIX,
        "H": H_VALID,
        "tail_horizon": TAIL_HORIZON,
        "maximum_action_budget": H_VALID - DETECTION_PREFIX + TAIL_HORIZON,
        "maximum_policy_call_budget": 2,
        "safety_tolerance_epsilon": SAFETY_NONINFERIORITY_EPSILON,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "action_disagreement_threshold": threshold,
        "action_disagreement_threshold_rule": "calibration 75th percentile; analysis only",
        "evaluation_outcomes_read": False,
        "diagnostic_only": DIAGNOSTIC_ONLY_GLOBAL or bool(calibration["diagnostic_only"]),
        "formal_positive_evidence_allowed": FORMAL_POSITIVE_EVIDENCE_ALLOWED,
    }
    baselines = {
        "schema_version": 1,
        "candidate_fixed_delays": fixed,
        "strongest_fixed_baseline": strongest,
        "selection_order": [
            "safe_success_descending",
            "cause_violation_ascending",
            "actual_post_detection_actions_ascending",
            "delay_ascending",
        ],
        "evaluation_outcomes_read": False,
    }
    atomic_write_json(output / "rule.json", rule)
    atomic_write_json(output / "calibration_selection.json", selection_rows)
    atomic_write_json(output / "baselines.json", baselines)
    (output / "report.md").write_text(
        "# Frozen outcome-blind handoff rule\n\n"
        "The target-shift rule is `release_index - 2`; the path-obstacle rule is "
        "`future intersection index - 2`, clipped to `[d,H]`. These formulas were fixed "
        "before evaluation. Calibration outcomes select only the strongest fixed comparator "
        f"(`{strongest['method']}`) and the analysis-only disagreement threshold.\n"
    )
    manifest = {
        "schema_version": 1,
        "files": {
            name: sha256_file(output / name)
            for name in ("rule.json", "calibration_selection.json", "baselines.json", "report.md")
        },
        "evaluation_outcomes_absent_at_freeze": True,
        "diagnostic_only_global": DIAGNOSTIC_ONLY_GLOBAL,
        "formal_positive_evidence_allowed": FORMAL_POSITIVE_EVIDENCE_ALLOWED,
    }
    atomic_write_json(output / "manifest.json", manifest)
    if MIRROR_EXPERIMENT_OUTPUTS:
        mirror = EXPERIMENT_ROOT / "frozen_rule"
        mirror.mkdir(parents=True, exist_ok=True)
        for name in ("rule.json", "calibration_selection.json", "baselines.json", "report.md", "manifest.json"):
            (mirror / name).write_bytes((output / name).read_bytes())
    print(json.dumps({"status": "FROZEN", "strongest_fixed": strongest["method"], "diagnostic_only": rule["diagnostic_only"]}, sort_keys=True))


if __name__ == "__main__":
    main()
