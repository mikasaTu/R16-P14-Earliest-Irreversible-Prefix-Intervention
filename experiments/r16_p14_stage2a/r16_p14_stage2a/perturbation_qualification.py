from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .envs import make_env
from .io_utils import atomic_write_csv, atomic_write_json, atomic_write_jsonl, atomic_write_text
from .mechanics import build_candidate, nominal_record, reconstruct_prefix, replay_equal
from .settings import (
    CALIBRATION_DEMOS,
    TASK_NAMES,
    TASK_SPECS,
    config_id,
    perturbation_grid,
)


def delayed_qualifies(record: dict[str, Any]) -> bool:
    if record.get("immediate_violation") or not record.get("cause_violation"):
        return False
    offset = record.get("first_violation_post_detection_step")
    if offset is None:
        return False
    family = TASK_SPECS[record["task"]].family
    minimum, maximum = ((1, 16) if family in {"target_shift", "target_region_blocker"} else (2, 8))
    return minimum <= int(offset) <= maximum


def severity(parameters: dict[str, float | int]) -> float:
    for key in (
        "shift_magnitude_m",
        "obstacle_clearance_m",
        "blocker_lateral_offset_m",
    ):
        if key in parameters:
            return float(parameters[key])
    return 0.0


def summarize(
    task_name: str,
    parameters: dict[str, float | int],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    valid = [record for record in records if record.get("phase_contract_valid")]
    immediate = sum(bool(record["immediate_violation"]) for record in valid)
    delayed = sum(delayed_qualifies(record) for record in valid)
    replay_passes = sum(bool(record["replay"]["passed"]) for record in valid)
    count = len(valid)
    immediate_rate = immediate / count if count else 1.0
    delayed_rate = delayed / count if count else 0.0
    replay_rate = replay_passes / count if count else 0.0
    constraints = {
        "all_calibration_events_phase_valid": count == len(CALIBRATION_DEMOS),
        "immediate_violation_rate_le_0_10": immediate_rate <= 0.10,
        "delayed_nominal_violation_rate_in_0_30_0_80": 0.30 <= delayed_rate <= 0.80,
        "fresh_replay_rate_ge_0_99": replay_rate >= 0.99,
    }
    identifier = records[0]["config_id"] if records else "unknown"
    qualifies = all(constraints.values())
    return {
        "task": task_name,
        "family": TASK_SPECS[task_name].family,
        "config_id": identifier,
        "parameters": parameters,
        "record_count": len(records),
        "valid_phase_count": count,
        "immediate_violation_count": immediate,
        "immediate_violation_rate": immediate_rate,
        "qualifying_delayed_violation_count": delayed,
        "nominal_delayed_violation_rate": delayed_rate,
        "replay_pass_count": replay_passes,
        "replay_pass_rate": replay_rate,
        "constraints": constraints,
        "qualifies": qualifies,
        "selection_rank": [
            0 if qualifies else 1,
            abs(delayed_rate - 0.55),
            immediate_rate,
            severity(parameters),
            identifier,
        ],
        "selection_used_proposed_method_gain": False,
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Frozen perturbation qualification",
        "",
        "Selection uses calibration demos 0–9 only: injection-instant violation, delayed nominal target-cause violation, and fresh replay. It never uses proposed-method gain.",
        "",
        "| Task | Selected configuration | Qualified configs | Immediate | Delayed nominal | Replay |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for task, item in summary["tasks"].items():
        lines.append(
            f"| {task} | {item['selected_config_id'] or 'none'} | "
            f"{item['qualified_config_count']} | {item['selected_immediate_violation_rate']} | "
            f"{item['selected_delayed_violation_rate']} | {item['selected_replay_rate']} |"
        )
    lines.extend(
        [
            "",
            "A task with no qualifying fixed-grid configuration remains a negative result. Evaluation and held-out splits were not inspected or retuned.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    selected_tasks: dict[str, Any] = {}
    for task_name in TASK_NAMES:
        env_nominal, _ = make_env(task_name, config_dir=args.config_dir, seed=0)
        env_replay, _ = make_env(task_name, config_dir=args.config_dir, seed=0)
        task_summaries: list[dict[str, Any]] = []
        try:
            grid = perturbation_grid(task_name)
            for grid_index, parameters in enumerate(grid, start=1):
                config_records: list[dict[str, Any]] = []
                for demo_id in CALIBRATION_DEMOS:
                    try:
                        candidate = build_candidate(env_nominal, task_name, demo_id, parameters)
                        record = nominal_record(env_nominal, candidate)
                        _, left = reconstruct_prefix(env_nominal, candidate, prefix_k=16)
                        _, right = reconstruct_prefix(env_replay, candidate, prefix_k=16)
                        record["replay"] = replay_equal(left, right)
                    except Exception as error:
                        record = {
                            "record_type": "calibration_nominal",
                            "task": task_name,
                            "demo_id": demo_id,
                            "config_id": config_id(task_name, parameters),
                            "parameters": parameters,
                            "phase_contract_valid": False,
                            "immediate_violation": False,
                            "cause_violation": False,
                            "first_violation_post_detection_step": None,
                            "invalid_reason": f"{type(error).__name__}: {error}",
                            "replay": {"passed": False},
                        }
                    config_records.append(record)
                    all_records.append(record)
                item = summarize(task_name, parameters, config_records)
                task_summaries.append(item)
                all_summaries.append(item)
                print(
                    "PERTURBATION_CONFIG_COMPLETE "
                    f"task={task_name} index={grid_index}/{len(grid)} "
                    f"config={item['config_id']} immediate={item['immediate_violation_rate']:.3f} "
                    f"delayed={item['nominal_delayed_violation_rate']:.3f} "
                    f"replay={item['replay_pass_rate']:.3f} qualifies={int(item['qualifies'])}",
                    flush=True,
                )
        finally:
            env_nominal.close()
            env_replay.close()
        qualified = sorted(
            (item for item in task_summaries if item["qualifies"]),
            key=lambda item: item["selection_rank"],
        )
        selected = qualified[0] if qualified else None
        selected_tasks[task_name] = {
            "status": "qualified" if selected else "no_qualifying_configuration",
            "selected_config_id": selected["config_id"] if selected else None,
            "selected_parameters": selected["parameters"] if selected else None,
            "qualified_config_count": len(qualified),
            "selected_immediate_violation_rate": (
                selected["immediate_violation_rate"] if selected else None
            ),
            "selected_delayed_violation_rate": (
                selected["nominal_delayed_violation_rate"] if selected else None
            ),
            "selected_replay_rate": selected["replay_pass_rate"] if selected else None,
        }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "calibration_demo_ids": list(CALIBRATION_DEMOS),
        "evaluation_demo_ids_inspected": [],
        "held_out_demo_ids_inspected": [],
        "selection_uses_proposed_method_gain": False,
        "tasks": selected_tasks,
        "qualified_task_count": sum(
            item["status"] == "qualified" for item in selected_tasks.values()
        ),
    }
    atomic_write_jsonl(args.output_dir / "calibration_records.jsonl", all_records)
    atomic_write_json(args.output_dir / "grid_summary.json", all_summaries)
    csv_rows = [
        {
            "task": item["task"],
            "family": item["family"],
            "config_id": item["config_id"],
            "valid_phase_count": item["valid_phase_count"],
            "immediate_violation_rate": item["immediate_violation_rate"],
            "nominal_delayed_violation_rate": item["nominal_delayed_violation_rate"],
            "replay_pass_rate": item["replay_pass_rate"],
            "qualifies": item["qualifies"],
        }
        for item in all_summaries
    ]
    atomic_write_csv(
        args.output_dir / "grid_summary.csv",
        csv_rows,
        list(csv_rows[0]) if csv_rows else [],
    )
    atomic_write_json(args.output_dir / "frozen_parameters.json", summary)
    atomic_write_json(
        args.output_dir / "perturbation_schedules.json",
        {
            "calibration_demo_ids": list(CALIBRATION_DEMOS),
            "evaluation_demo_ids": list(range(10, 40)),
            "held_out_demo_ids": list(range(40, 50)),
            "selected": selected_tasks,
        },
    )
    atomic_write_text(args.output_dir / "report.md", render(summary))
    atomic_write_text(
        args.output_dir / "negative_results.md",
        "# Perturbation negative results\n\n"
        + "\n".join(
            f"- `{task}`: no qualifying fixed-grid configuration."
            for task, item in selected_tasks.items()
            if item["status"] != "qualified"
        )
        + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
