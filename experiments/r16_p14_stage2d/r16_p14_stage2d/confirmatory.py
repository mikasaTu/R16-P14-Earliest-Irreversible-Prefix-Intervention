from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .fresh_process import run_spawned_branch
from .freeze_rule import rule_k
from .io_utils import atomic_write_json, atomic_write_jsonl, load_jsonl, sha256_file
from .settings import (
    ACTOR_SEEDS,
    ARTIFACT_ROOT,
    CONFIRMATORY_METHODS,
    EXPERIMENT_ROOT,
    MIRROR_EXPERIMENT_OUTPUTS,
    PROJECT_ROOT,
    TASKS,
    DIAGNOSTIC_ONLY_GLOBAL,
    FORMAL_POSITIVE_EVIDENCE_ALLOWED,
)


def verify_frozen_rule() -> dict[str, Any]:
    root = ARTIFACT_ROOT / "frozen_rule"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("frozen_rule manifest is absent")
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["files"].items():
        if sha256_file(root / name) != digest:
            raise RuntimeError(f"frozen rule checksum mismatch: {name}")
    # ARTIFACT_ROOT points at the PAI run directory during formal execution,
    # while the immutable barrier copy is committed at this fixed repository
    # path. Compare the two byte-for-byte instead of assuming the run output is
    # physically nested under PROJECT_ROOT.
    relative = "artifacts/stage2d/frozen_rule/manifest.json"
    try:
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:{relative}"], cwd=PROJECT_ROOT
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("frozen_rule manifest must be committed before evaluation") from exc
    if hashlib.sha256(committed).hexdigest() != sha256_file(manifest_path):
        raise RuntimeError("committed frozen_rule manifest differs from working tree")
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


def replay_path(item: dict[str, Any], repeat: int) -> Path:
    return ARTIFACT_ROOT / "confirmatory_evaluation/replay_shards" / f"{item['event_instance_id']}__repeat{repeat}.json"


def method_path(item: dict[str, Any], method: str) -> Path:
    return ARTIFACT_ROOT / "confirmatory_evaluation/shards" / f"{item['event_instance_id']}__{method}.json"


def assignment(item: dict[str, Any], suffix: str, worker_count: int) -> int:
    return int(hashlib.sha256(f"{item['event_instance_id']}__{suffix}".encode()).hexdigest(), 16) % worker_count


def event_replay_status(item: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    rows = [json.loads(replay_path(item, repeat).read_text()) for repeat in range(3) if replay_path(item, repeat).is_file()]
    if len(rows) != 3 or any(row.get("error") for row in rows):
        return False, rows
    signatures = {row["detection_signature"]["complete_signature_hash"] for row in rows}
    exact = all(
        all(bool(value) for key, value in row["reconstruction"].items() if key != "max_anchor_state_error")
        and float(row["reconstruction"]["max_anchor_state_error"]) <= 1e-9
        for row in rows
    )
    return bool(exact and len(signatures) == 1), rows


def run_worker(
    worker_index: int,
    worker_count: int,
    device: str,
    phase: str,
    max_new_shards: int | None = None,
) -> None:
    verify_frozen_rule()
    pool, events = pool_and_events()
    rule = json.loads((ARTIFACT_ROOT / "frozen_rule/rule.json").read_text())
    new_shards = 0
    if phase in {"all", "replay"}:
        for item in pool:
            event = events[item["event_id"]]
            for repeat in range(3):
                if assignment(item, f"replay{repeat}", worker_count) != worker_index:
                    continue
                path = replay_path(item, repeat)
                if not path.is_file():
                    row = run_spawned_branch(
                        event=event,
                        parameter=item["parameter"],
                        prefix_k=2,
                        arm="RECONSTRUCT_ONLY",
                        repeat=repeat,
                        device=device,
                    )
                    row["event_instance_id"] = item["event_instance_id"]
                    row["replay_only"] = True
                    atomic_write_json(path, row)
                    new_shards += 1
                print(f"CONFIRM_REPLAY {item['event_instance_id']} repeat={repeat}", flush=True)
                if max_new_shards is not None and new_shards >= max_new_shards:
                    return
    # Synchronization is external when worker_count > 1: run a second resume pass
    # after both workers finish. Method rows remain diagnostic if replay shards are incomplete.
    if phase in {"all", "methods"}:
        for item in pool:
            event = events[item["event_id"]]
            admitted, _ = event_replay_status(item)
            k = rule_k(event, item["parameter"])
            for method in CONFIRMATORY_METHODS:
                if assignment(item, method, worker_count) != worker_index:
                    continue
                path = method_path(item, method)
                if path.is_file():
                    print(f"CONFIRM_RESUME {item['event_instance_id']} {method}", flush=True)
                    continue
                row = run_spawned_branch(
                    event=event,
                    parameter=item["parameter"],
                    prefix_k=k,
                    arm=method,
                    repeat=0,
                    device=device,
                )
                row.update(
                    {
                        "event_instance_id": item["event_instance_id"],
                        "rule_k": k,
                        "replay_admitted_3_of_3": admitted,
                        "diagnostic_only": bool(rule["diagnostic_only"] or not admitted),
                        "evaluation_all_k_scanned_before_rule": False,
                        "frozen_rule_manifest_sha256": sha256_file(
                            ARTIFACT_ROOT / "frozen_rule/manifest.json"
                        ),
                    }
                )
                atomic_write_json(path, row)
                new_shards += 1
                print(f"CONFIRM_DONE {item['event_instance_id']} {method} error={int(bool(row.get('error')))}", flush=True)
                if max_new_shards is not None and new_shards >= max_new_shards:
                    return


def consolidate() -> dict[str, Any]:
    verify_frozen_rule()
    pool, _ = pool_and_events()
    rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    replay_status: dict[str, bool] = {}
    for item in pool:
        admitted, event_replays = event_replay_status(item)
        replay_status[item["event_instance_id"]] = admitted
        replay_rows.extend(event_replays)
        for repeat in range(3):
            if not replay_path(item, repeat).is_file():
                missing.append(str(replay_path(item, repeat).relative_to(ARTIFACT_ROOT)))
        for method in CONFIRMATORY_METHODS:
            path = method_path(item, method)
            if path.is_file():
                row = json.loads(path.read_text())
                row["replay_admitted_3_of_3"] = admitted
                row["diagnostic_only"] = bool(row["diagnostic_only"] or not admitted)
                rows.append(row)
            else:
                missing.append(str(path.relative_to(ARTIFACT_ROOT)))
    output = ARTIFACT_ROOT / "confirmatory_evaluation"
    expected_method_rows = len(pool) * len(CONFIRMATORY_METHODS)
    expected_replay_rows = len(pool) * 3
    complete_matrix = (
        not missing
        and len(rows) == expected_method_rows
        and len(replay_rows) == expected_replay_rows
    )
    terminal_path = output / "terminal_receipt.json"
    terminal_receipt = json.loads(terminal_path.read_text()) if terminal_path.is_file() else {
        "status": "MISSING",
        "complete_matrix": False,
        "source": "no terminal PAI receipt was imported before consolidation",
    }
    atomic_write_jsonl(output / "replay_rows.jsonl", replay_rows)
    atomic_write_jsonl(output / "method_rows.jsonl", rows)
    by_event: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_event[row["event_instance_id"]][row["requested_arm"]] = row
    paired = []
    for item in pool:
        methods = by_event[item["event_instance_id"]]
        paired.append(
            {
                "event_instance_id": item["event_instance_id"],
                "event_id": item["event_id"],
                "task": item["task"],
                "actor_seed": item["actor_seed"],
                "init_state_id": item["init_state_id"],
                "parameter_id": item["parameter_id"],
                "rule_k": next(iter(methods.values()))["rule_k"] if methods else None,
                "replay_admitted_3_of_3": replay_status[item["event_instance_id"]],
                "methods": {
                    method: {
                        key: methods[method].get(key)
                        for key in (
                            "safe_success",
                            "cause_violation",
                            "task_success",
                            "actual_post_detection_actions",
                            "completion_steps",
                            "actual_actor_calls",
                            "actual_inference_wall_time_s",
                            "total_branch_wall_time_s",
                            "eef_path_length_m",
                            "manipulated_object_path_length_m",
                            "task_progress_at_k",
                            "progress_regression_m",
                            "cached_actions_retained",
                            "cached_fresh_action_disagreement",
                            "injection_contact",
                            "error",
                        )
                    }
                    for method in CONFIRMATORY_METHODS
                    if method in methods
                },
            }
        )
    atomic_write_jsonl(output / "paired_rows.jsonl", paired)
    valid_events = [
        item
        for item in pool
        if replay_status[item["event_instance_id"]]
        and len(by_event[item["event_instance_id"]]) == len(CONFIRMATORY_METHODS)
        and not any(row.get("error") for row in by_event[item["event_instance_id"]].values())
    ]
    minimum = {}
    for task in TASKS:
        subset = [item for item in valid_events if item["task"] == task]
        severity = Counter(item["parameter_id"] for item in subset)
        seeds = Counter(item["actor_seed"] for item in subset)
        checks = {
            "events_ge30": len(subset) >= 30,
            "each_severity_ge12": len(severity) >= 2 and min(severity.values(), default=0) >= 12,
            "each_actor_seed_ge8": all(seeds[seed] >= 8 for seed in ACTOR_SEEDS),
            "distinct_init_clusters_ge24": len({item["init_state_id"] for item in subset}) >= 24,
        }
        minimum[task] = {
            "valid_events": len(subset),
            "by_severity": dict(severity),
            "by_actor_seed": {str(seed): seeds[seed] for seed in ACTOR_SEEDS},
            "distinct_init_clusters": len({item["init_state_id"] for item in subset}),
            "checks": checks,
            "passes": all(checks.values()),
        }
    summary = {
        "schema_version": 1,
        "status": "PASS" if complete_matrix and all(value["passes"] for value in minimum.values()) else "BLOCKED_BY_MINIMUM_DATA",
        "expected_method_rows": expected_method_rows,
        "method_rows": len(rows),
        "expected_replay_rows": expected_replay_rows,
        "replay_rows": len(replay_rows),
        "replay_admitted_events": sum(replay_status.values()),
        "error_count": sum(bool(row.get("error")) for row in rows + replay_rows),
        "missing_shards": missing,
        "minimum_data": minimum,
        "diagnostic_only": DIAGNOSTIC_ONLY_GLOBAL or any(row.get("diagnostic_only") for row in rows),
        "formal_positive_evidence_allowed": FORMAL_POSITIVE_EVIDENCE_ALLOWED,
        "complete_matrix": complete_matrix,
        "statistics_eligible": bool(
            complete_matrix
            and not any(row.get("error") for row in rows + replay_rows)
            and terminal_receipt.get("status") == "SUCCEEDED"
        ),
        "terminal_receipt": terminal_receipt,
        "evaluation_all_k_scanned_before_rule": False,
    }
    atomic_write_json(output / "summary.json", summary)
    (output / "report.md").write_text(
        "# Confirmatory paired evaluation\n\n"
        f"Status: **{summary['status']}**; method rows {len(rows)} / {summary['expected_method_rows']}; "
        f"replay-admitted events {summary['replay_admitted_events']} / {len(pool)}. The frozen rule "
        "was applied before any all-k evaluation scan.\n"
    )
    if MIRROR_EXPERIMENT_OUTPUTS:
        mirror = EXPERIMENT_ROOT / "confirmatory_evaluation"
        mirror.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "report.md"):
            (mirror / name).write_bytes((output / name).read_bytes())
    print(json.dumps({"status": summary["status"], "rows": len(rows), "replay_admitted": summary["replay_admitted_events"]}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phase", choices=("all", "replay", "methods"), default="all")
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
            args.phase,
            args.max_new_shards,
        )


if __name__ == "__main__":
    main()
