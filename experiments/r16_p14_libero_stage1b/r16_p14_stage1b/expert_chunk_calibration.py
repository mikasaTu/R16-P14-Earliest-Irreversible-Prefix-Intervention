from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from r16_p14_stage1.io_utils import atomic_write_json

from .expert_common import (
    BRANCH_BUDGET,
    CHUNK_LENGTH,
    EXPERT_TASKS,
    INSERTION_PREFIX,
    ExpertCandidate,
    build_candidate,
    config_id,
    drawer_grid,
    json_dumps,
    make_expert_env,
    reconstruct_prefix,
    run_full_replan,
    run_local_repair,
    run_nominal,
    target_shift_grid,
)


DEFAULT_TASKS = (
    "put_the_bowl_on_the_plate",
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_cream_cheese_in_the_bowl",
)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json_dumps(record) + "\n" for record in records))


def task_grid(task_name: str) -> list[dict[str, float | int]]:
    if EXPERT_TASKS[task_name].family == "target_shift":
        return target_shift_grid()
    return drawer_grid()


def is_delayed_qualifying_violation(record: dict[str, Any]) -> bool:
    offset = record["first_violation_post_detection_step"]
    minimum_offset = 2 if EXPERT_TASKS[record["task"]].family == "drawer_obstacle" else 1
    maximum_offset = 8 if EXPERT_TASKS[record["task"]].family == "drawer_obstacle" else CHUNK_LENGTH
    return bool(
        record["cause_violation"]
        and not record["immediate_violation"]
        and offset is not None
        and minimum_offset <= int(offset) <= maximum_offset
    )


def audit_candidate_windows(
    candidate: ExpertCandidate,
    env_full,
    env_local,
) -> dict[str, Any]:
    prefixes: list[dict[str, Any]] = []
    replay_passes = 0
    for prefix_k in range(INSERTION_PREFIX, CHUNK_LENGTH + 1):
        tracker_full, info_full = reconstruct_prefix(
            env_full, candidate, prefix_k=prefix_k
        )
        tracker_local, info_local = reconstruct_prefix(
            env_local, candidate, prefix_k=prefix_k
        )
        state_match = info_full["state_hash"] == info_local["state_hash"]
        contact_match = info_full["contacts_hash"] == info_local["contacts_hash"]
        prefix_violation_match = (
            tracker_full.violation,
            tracker_full.violation_type,
        ) == (
            tracker_local.violation,
            tracker_local.violation_type,
        )
        replay_pass = bool(state_match and contact_match and prefix_violation_match)
        replay_passes += int(replay_pass)
        prefix_cause_violation = bool(tracker_full.violation)
        full = run_full_replan(
            env_full,
            tracker_full,
            info_full,
            prefix_k=prefix_k,
            budget=BRANCH_BUDGET,
        )
        local = run_local_repair(
            env_local,
            tracker_local,
            info_local,
            prefix_k=prefix_k,
            budget=BRANCH_BUDGET,
        )
        prefixes.append(
            {
                "prefix_k": prefix_k,
                "prefix_cause_violation": prefix_cause_violation,
                "replay": {
                    "state_hash_match": state_match,
                    "contact_trace_match": contact_match,
                    "prefix_violation_match": prefix_violation_match,
                    "passed": replay_pass,
                    "state_hash": info_full["state_hash"],
                    "contacts_hash": info_full["contacts_hash"],
                },
                "full_replan": full,
                "cause_specific_local_repair": local,
                "deployable_recovery": bool(full["safe_success"] or local["safe_success"]),
            }
        )
    safe_prefixes = [
        item["prefix_k"] for item in prefixes if item["deployable_recovery"]
    ]
    has_recovery = bool(safe_prefixes)
    k_last_safe = max(safe_prefixes) if safe_prefixes else INSERTION_PREFIX
    chosen = next(item for item in prefixes if item["prefix_k"] == k_last_safe)
    immediate = next(item for item in prefixes if item["prefix_k"] == INSERTION_PREFIX)
    window = k_last_safe - INSERTION_PREFIX if has_recovery else 0
    return {
        "record_type": "calibration_prefix_audit",
        "candidate_id": candidate.candidate_id,
        "task": candidate.task,
        "demo_id": candidate.demo_id,
        "config_id": candidate.config_id,
        "parameters": candidate.parameters,
        "insertion_prefix": INSERTION_PREFIX,
        "chunk_length": CHUNK_LENGTH,
        "has_deployable_recovery": has_recovery,
        "k_last_safe": k_last_safe if has_recovery else None,
        "intervention_window": window,
        "post_detection_retention": window / (CHUNK_LENGTH - INSERTION_PREFIX),
        "replay_pass_count": replay_passes,
        "replay_branch_point_count": len(prefixes),
        "M0_immediate_full_replan": immediate["full_replan"],
        "M1_continue_then_full_replan": chosen["full_replan"],
        "M2_continue_then_local_repair": chosen["cause_specific_local_repair"],
        "prefixes": prefixes,
    }


def summarize_config(
    task_name: str,
    parameters: dict[str, float | int],
    nominal: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    expected_demo_count: int,
) -> dict[str, Any]:
    valid = [record for record in nominal if record["phase_contract_valid"]]
    immediate_count = sum(bool(record["immediate_violation"]) for record in valid)
    delayed_count = sum(is_delayed_qualifying_violation(record) for record in valid)
    immediate_rate = immediate_count / len(valid) if valid else 1.0
    delayed_rate = delayed_count / len(valid) if valid else 0.0
    windows = [record["intervention_window"] for record in audits]
    retentions = [record["post_detection_retention"] for record in audits]
    replay_pass_count = sum(record["replay_pass_count"] for record in audits)
    replay_count = sum(record["replay_branch_point_count"] for record in audits)
    median_window = statistics.median(windows) if windows else None
    median_retention = statistics.median(retentions) if retentions else None
    constraints = {
        "all_calibration_demos_valid": len(valid) == expected_demo_count,
        "immediate_violation_rate_le_0_10": immediate_rate <= 0.10,
        "delayed_violation_rate_in_0_30_0_80": 0.30 <= delayed_rate <= 0.80,
        "median_actual_window_in_2_8": bool(
            median_window is not None and 2 <= median_window <= 8
        ),
        "replay_ge_0_99": bool(
            replay_count and replay_pass_count / replay_count >= 0.99
        ),
    }
    magnitude = float(
        parameters.get(
            "target_shift_x_magnitude_meters",
            parameters.get("obstacle_clearance_meters", 0.0),
        )
    )
    return {
        "task": task_name,
        "config_id": config_id(task_name, parameters),
        "parameters": parameters,
        "nominal_record_count": len(nominal),
        "valid_phase_count": len(valid),
        "immediate_violation_count": immediate_count,
        "immediate_violation_rate": immediate_rate,
        "qualifying_delayed_violation_count": delayed_count,
        "nominal_delayed_violation_rate": delayed_rate,
        "prefix_audit_candidate_count": len(audits),
        "median_intervention_window": median_window,
        "median_post_detection_retention": median_retention,
        "replay_pass_rate": replay_pass_count / replay_count if replay_count else None,
        "constraints": constraints,
        "qualifies": all(constraints.values()),
        "selection_rank": [
            0 if all(constraints.values()) else 1,
            abs(delayed_rate - 0.55),
            immediate_rate,
            magnitude,
            config_id(task_name, parameters),
        ],
    }


def write_grid_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "config_id",
        "nominal_record_count",
        "valid_phase_count",
        "immediate_violation_rate",
        "nominal_delayed_violation_rate",
        "prefix_audit_candidate_count",
        "median_intervention_window",
        "median_post_detection_retention",
        "replay_pass_rate",
        "qualifies",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in summaries:
            writer.writerow({key: record.get(key) for key in fields})
    temporary.replace(path)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase C calibration — expert action chunks",
        "",
        "No BC policy, learned risk model, or policy checkpoint is used. Parameters are selected only on demo IDs 0–9 with the preregistered fixed grid.",
        "",
        "## Frozen task decisions",
        "",
        "- `put_the_bowl_on_the_plate`: primary target-shift task.",
        "- `open_the_middle_drawer_of_the_cabinet`: revised obstacle-placement task.",
        "- `put_the_wine_bottle_on_the_rack`: inapplicable because the rack is static model geometry and direct manipulated-bottle teleport is forbidden.",
        "- `put_the_cream_cheese_in_the_bowl`: the single bounded replacement, selected before calibration as the first suite-order task with a stable grasp, late release, and movable target joint.",
        "",
        "## Grid result",
        "",
        "| Task | Selected configuration | Qualified configurations | Status |",
        "| --- | --- | ---: | --- |",
    ]
    for task_name, task_summary in summary["tasks"].items():
        selected = task_summary["selected_config_id"] or "none"
        lines.append(
            f"| {task_name} | {selected} | {task_summary['qualified_config_count']} | {task_summary['status']} |"
        )
    lines.extend(
        [
            "",
            "A qualifying perturbation must have at most 10% injection-instant violations, 30–80% nominal cause violations occurring 2–8 actions later, an actual median recoverable window of 2–8 actions, and at least 99% independent prefix reconstruction agreement. Violation-onset delay is not substituted for the recovery window.",
            "",
            "Configurations without a qualifying result are left unselected; no evaluation-demo retuning is permitted.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=sorted(EXPERT_TASKS), default=list(DEFAULT_TASKS))
    parser.add_argument("--demo-start", type=int, default=0)
    parser.add_argument("--demo-count", type=int, default=10)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--only-config-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    records_path = output_dir / "calibration_records.jsonl"
    if records_path.exists():
        raise FileExistsError(f"refusing to overwrite calibration records: {records_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = (
        args.repo_root.resolve()
        / "experiments/r16_p14_libero_stage1/libero_config"
    )
    demo_ids = list(range(args.demo_start, args.demo_start + args.demo_count))
    all_records: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    frozen: dict[str, Any] = {
        "schema_version": 1,
        "calibration_demo_ids": demo_ids,
        "selection_uses_evaluation_or_heldout_demos": False,
        "tasks": {},
    }
    started = time.time()
    for task_name in args.tasks:
        # Independent instances use the same paired environment seed. Different
        # seeds can change static placements that are not encoded in MuJoCo's
        # flattened state and would invalidate a reconstruction comparison.
        env_nominal = make_expert_env(task_name, config_dir, seed=0)
        env_full = make_expert_env(task_name, config_dir, seed=0)
        env_local = make_expert_env(task_name, config_dir, seed=0)
        try:
            grid = task_grid(task_name)
            if args.only_config_id:
                grid = [
                    parameters
                    for parameters in grid
                    if config_id(task_name, parameters) == args.only_config_id
                ]
                if not grid:
                    raise ValueError(
                        f"unknown config for {task_name}: {args.only_config_id}"
                    )
            if args.max_configs is not None:
                grid = grid[: args.max_configs]
            task_summaries: list[dict[str, Any]] = []
            for config_index, parameters in enumerate(grid):
                nominal_records: list[dict[str, Any]] = []
                candidates: dict[int, ExpertCandidate] = {}
                for demo_id in demo_ids:
                    try:
                        candidate = build_candidate(
                            env_nominal, task_name, demo_id, parameters
                        )
                    except ValueError as error:
                        invalid = {
                            "record_type": "calibration_nominal",
                            "candidate_id": None,
                            "task": task_name,
                            "demo_id": demo_id,
                            "config_id": config_id(task_name, parameters),
                            "parameters": parameters,
                            "phase_contract": {},
                            "phase_contract_valid": False,
                            "invalid_reason": str(error),
                            "immediate_violation": False,
                            "cause_violation": False,
                            "violation_type": None,
                            "first_violation_post_detection_step": None,
                            "task_success": False,
                            "safe_success": False,
                        }
                        nominal_records.append(invalid)
                        all_records.append(invalid)
                        continue
                    candidates[demo_id] = candidate
                    nominal = run_nominal(env_nominal, candidate)
                    nominal["record_type"] = "calibration_nominal"
                    nominal_records.append(nominal)
                    all_records.append(nominal)
                valid = [r for r in nominal_records if r["phase_contract_valid"]]
                immediate_rate = (
                    sum(bool(r["immediate_violation"]) for r in valid) / len(valid)
                    if valid
                    else 1.0
                )
                delayed_rate = (
                    sum(is_delayed_qualifying_violation(r) for r in valid) / len(valid)
                    if valid
                    else 0.0
                )
                audits: list[dict[str, Any]] = []
                if (
                    len(valid) == len(demo_ids)
                    and immediate_rate <= 0.10
                    and 0.30 <= delayed_rate <= 0.80
                ):
                    for nominal in nominal_records:
                        if not is_delayed_qualifying_violation(nominal):
                            continue
                        audit = audit_candidate_windows(
                            candidates[nominal["demo_id"]], env_full, env_local
                        )
                        audits.append(audit)
                        all_records.append(audit)
                config_summary = summarize_config(
                    task_name,
                    parameters,
                    nominal_records,
                    audits,
                    len(demo_ids),
                )
                task_summaries.append(config_summary)
                all_summaries.append(config_summary)
                print(
                    "CALIBRATION_CONFIG_COMPLETE "
                    f"task={task_name} index={config_index + 1}/{len(grid)} "
                    f"config={config_summary['config_id']} immediate={immediate_rate:.3f} "
                    f"delayed={delayed_rate:.3f} window={config_summary['median_intervention_window']} "
                    f"qualifies={int(config_summary['qualifies'])}",
                    flush=True,
                )
            qualified = sorted(
                [item for item in task_summaries if item["qualifies"]],
                key=lambda item: item["selection_rank"],
            )
            selected = qualified[0] if qualified else None
            frozen["tasks"][task_name] = {
                "status": "qualified" if selected else "no_qualifying_configuration",
                "selected_config_id": selected["config_id"] if selected else None,
                "parameters": selected["parameters"] if selected else None,
                "selection_rank": selected["selection_rank"] if selected else None,
                "qualified_config_count": len(qualified),
            }
        finally:
            env_nominal.close()
            env_full.close()
            env_local.close()

    summary = {
        "schema_version": 1,
        "phase": "C_expert_chunk_calibration",
        "status": "complete",
        "policy_used": False,
        "learned_risk_used": False,
        "calibration_demo_ids": demo_ids,
        "evaluation_demo_ids_inspected": [],
        "heldout_demo_ids_inspected": [],
        "tasks": frozen["tasks"],
        "qualified_task_count": sum(
            item["status"] == "qualified" for item in frozen["tasks"].values()
        ),
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_jsonl(records_path, all_records)
    atomic_write_json(output_dir / "grid_summary.json", all_summaries)
    write_grid_csv(output_dir / "grid_summary.csv", all_summaries)
    atomic_write_json(output_dir / "frozen_parameters.json", frozen)
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_text(output_dir / "report.md", render_report(summary))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
