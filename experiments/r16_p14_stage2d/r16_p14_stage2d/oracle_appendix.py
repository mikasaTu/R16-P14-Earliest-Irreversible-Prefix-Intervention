from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .fresh_process import run_spawned_branch
from .io_utils import atomic_write_json, atomic_write_jsonl, load_jsonl, sha256_file
from .settings import (
    ARTIFACT_ROOT,
    DIAGNOSTIC_ONLY_GLOBAL,
    EXPERIMENT_ROOT,
    FORMAL_POSITIVE_EVIDENCE_ALLOWED,
    MIRROR_EXPERIMENT_OUTPUTS,
    PREFIX_INDICES,
    PROJECT_ROOT,
)


def verify_primary_lock() -> dict[str, Any]:
    path = ARTIFACT_ROOT / "statistics/primary_manifest.json"
    if not path.is_file():
        raise RuntimeError("primary decision manifest is missing")
    manifest = json.loads(path.read_text())
    for label, digest in manifest["files"].items():
        target = {
            "confirmatory_paired_rows": ARTIFACT_ROOT / "confirmatory_evaluation/paired_rows.jsonl",
            "frozen_rule_manifest": ARTIFACT_ROOT / "frozen_rule/manifest.json",
            "statistics": ARTIFACT_ROOT / "statistics/statistics.json",
            "primary_decision": ARTIFACT_ROOT / "statistics/primary_decision.json",
        }[label]
        if sha256_file(target) != digest:
            raise RuntimeError(f"primary lock checksum mismatch: {label}")
    # The live PAI artifact root is outside the clean source checkout. The
    # primary barrier is nevertheless required to be present in the immutable
    # source commit at this fixed repository-relative path.
    relative = "artifacts/stage2d/statistics/primary_manifest.json"
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=PROJECT_ROOT)
    if hashlib.sha256(committed).hexdigest() != sha256_file(path):
        raise RuntimeError("primary manifest must be committed unchanged before oracle appendix")
    return manifest


def pool_and_events() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    pool = [
        item
        for item in load_jsonl(ARTIFACT_ROOT / "actor_events/formal_event_pool.jsonl")
        if item["split"] == "evaluation"
    ]
    events = {
        event["event_id"]: event
        for event in load_jsonl(ARTIFACT_ROOT / "actor_events/events.jsonl")
    }
    return pool, events


def shard_path(item: dict[str, Any], prefix_k: int) -> Path:
    return ARTIFACT_ROOT / "oracle_appendix/shards" / f"{item['event_instance_id']}__k{prefix_k:02d}.json"


def run_worker(
    worker_index: int,
    worker_count: int,
    device: str,
    max_new_shards: int | None = None,
) -> None:
    verify_primary_lock()
    pool, events = pool_and_events()
    new_shards = 0
    for item in pool:
        event = events[item["event_id"]]
        for prefix_k in PREFIX_INDICES:
            key = f"{item['event_instance_id']}__k{prefix_k:02d}"
            if int(hashlib.sha256(key.encode()).hexdigest(), 16) % worker_count != worker_index:
                continue
            path = shard_path(item, prefix_k)
            if path.is_file():
                print(f"ORACLE_RESUME {key}", flush=True)
                continue
            row = run_spawned_branch(
                event=event,
                parameter=item["parameter"],
                prefix_k=prefix_k,
                arm="CACHED_MATCHED",
                repeat=0,
                device=device,
            )
            row.update(
                {
                    "event_instance_id": item["event_instance_id"],
                    "appendix_only": True,
                    "primary_decision_effect": "FORBIDDEN",
                    "primary_manifest_sha256": sha256_file(
                        ARTIFACT_ROOT / "statistics/primary_manifest.json"
                    ),
                }
            )
            atomic_write_json(path, row)
            new_shards += 1
            print(f"ORACLE_DONE {key} error={int(bool(row.get('error')))}", flush=True)
            if max_new_shards is not None and new_shards >= max_new_shards:
                return


def objective(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(bool(row["safe_success"])),
        int(bool(row["cause_violation"])),
        int(row["actual_post_detection_actions"]),
        math.inf if row["completion_steps"] is None else int(row["completion_steps"]),
        int(row["actual_actor_calls"]),
        int(row["requested_prefix_k"]),
    )


def consolidate() -> dict[str, Any]:
    primary_lock = verify_primary_lock()
    primary_before = json.loads((ARTIFACT_ROOT / "statistics/primary_decision.json").read_text())
    pool, _ = pool_and_events()
    rows, missing = [], []
    for item in pool:
        for prefix_k in PREFIX_INDICES:
            path = shard_path(item, prefix_k)
            if path.is_file():
                rows.append(json.loads(path.read_text()))
            else:
                missing.append(str(path.relative_to(ARTIFACT_ROOT)))
    output = ARTIFACT_ROOT / "oracle_appendix"
    expected_rows = len(pool) * len(PREFIX_INDICES)
    terminal_path = output / "terminal_receipt.json"
    terminal_receipt = json.loads(terminal_path.read_text()) if terminal_path.is_file() else {
        "status": "MISSING",
        "complete_matrix": False,
        "source": "no terminal PAI receipt was imported before consolidation",
    }
    complete_matrix = not missing and len(rows) == expected_rows
    atomic_write_jsonl(output / "rows.jsonl", rows)
    by_event = defaultdict(list)
    for row in rows:
        if not row.get("error"):
            by_event[row["event_instance_id"]].append(row)
    oracle_rows = []
    for item in pool:
        candidates = by_event[item["event_instance_id"]]
        if len(candidates) != len(PREFIX_INDICES):
            continue
        best = min(candidates, key=objective)
        oracle_rows.append(
            {
                "event_instance_id": item["event_instance_id"],
                "task": item["task"],
                "actor_seed": item["actor_seed"],
                "parameter_id": item["parameter_id"],
                "oracle_k": int(best["requested_prefix_k"]),
                "safe_success": best["safe_success"],
                "cause_violation": best["cause_violation"],
                "actual_post_detection_actions": best["actual_post_detection_actions"],
                "objective": list(objective(best)),
            }
        )
    atomic_write_jsonl(output / "oracle_rows.jsonl", oracle_rows)
    confirm = {
        row["event_instance_id"]: row
        for row in load_jsonl(ARTIFACT_ROOT / "confirmatory_evaluation/paired_rows.jsonl")
    }
    gaps = []
    for row in oracle_rows:
        paired = confirm.get(row["event_instance_id"])
        if not paired or "EVENT_ALIGNED_CACHED" not in paired["methods"]:
            continue
        rule = paired["methods"]["EVENT_ALIGNED_CACHED"]
        gaps.append(
            {
                "event_instance_id": row["event_instance_id"],
                "rule_minus_oracle_post_actions": float(rule["actual_post_detection_actions"])
                - float(row["actual_post_detection_actions"]),
                "rule_safe_success": rule["safe_success"],
                "oracle_safe_success": row["safe_success"],
            }
        )
    atomic_write_jsonl(output / "rule_oracle_gaps.jsonl", gaps)
    primary_after = json.loads((ARTIFACT_ROOT / "statistics/primary_decision.json").read_text())
    primary_unchanged = primary_before == primary_after
    summary = {
        "schema_version": 1,
        "status": "PASS" if complete_matrix and len(oracle_rows) == len(pool) else "INCOMPLETE",
        "expected_rows": expected_rows,
        "rows": len(rows),
        "oracle_events": len(oracle_rows),
        "missing_shards": missing,
        "error_count": sum(bool(row.get("error")) for row in rows),
        "mean_rule_minus_oracle_post_actions": float(
            np.mean([row["rule_minus_oracle_post_actions"] for row in gaps])
        )
        if gaps
        else None,
        "primary_decision_unchanged": primary_unchanged,
        "primary_decision_effect": "NONE",
        "primary_manifest": primary_lock,
        "diagnostic_only": DIAGNOSTIC_ONLY_GLOBAL,
        "formal_positive_evidence_allowed": FORMAL_POSITIVE_EVIDENCE_ALLOWED,
        "complete_matrix": complete_matrix,
        "terminal_receipt": terminal_receipt,
        "statistics_eligible": bool(complete_matrix and terminal_receipt.get("status") == "SUCCEEDED"),
    }
    atomic_write_json(output / "summary.json", summary)
    (output / "report.md").write_text(
        "# Evaluation oracle appendix\n\n"
        f"Rows: {len(rows)} / {summary['expected_rows']}; complete oracle events: "
        f"{len(oracle_rows)} / {len(pool)}. This scan ran only after the primary decision "
        f"was committed; primary decision unchanged: {primary_unchanged}.\n"
    )
    if MIRROR_EXPERIMENT_OUTPUTS:
        mirror = EXPERIMENT_ROOT / "oracle_appendix"
        mirror.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "report.md"):
            (mirror / name).write_bytes((output / name).read_bytes())
    print(json.dumps({"status": summary["status"], "primary_unchanged": primary_unchanged}, sort_keys=True))
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
        run_worker(args.worker_index, args.worker_count, args.device, args.max_new_shards)


if __name__ == "__main__":
    main()
