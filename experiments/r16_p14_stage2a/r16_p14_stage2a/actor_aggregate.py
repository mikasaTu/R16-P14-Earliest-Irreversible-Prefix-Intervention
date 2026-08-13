from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import atomic_write_csv, atomic_write_json, atomic_write_text
from .settings import TASK_NAMES


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# History-Conditioned State ACT qualification",
        "",
        f"Parameter count: `{summary['parameter_count']}` (limit `10,000,000`). The frozen actor uses four state observations, three executed actions, task ID, and sixteen action queries; it uses no RGB, world model, or perturbation metadata.",
        "",
        "| Task | Clean success | Grasp/open | Lift/transport | Pre-release/contact | Smoothness | Policy calls | Qualified |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for task, item in summary["tasks"].items():
        lines.append(
            f"| {task} | {item['clean_success_rate']:.3f} | "
            f"{item['grasp_or_open_phase_reach_rate']:.3f} | "
            f"{item['lift_or_transport_phase_reach_rate']:.3f} | "
            f"{item['pre_release_or_contact_phase_reach_rate']:.3f} | "
            f"{item['mean_chunk_smoothness']:.4f} | {item['mean_policy_calls']:.1f} | "
            f"{item['qualification_status']} |"
        )
    lines.extend(
        [
            "",
            f"Qualified tasks: `{summary['qualified_task_count']}/6`; actor gate: **{summary['actor_gate']}**.",
            "",
            "Qualification A is clean success in [0.30, 0.90]. Qualification B requires pre-release/contact reach >=0.70 and either no failures or at least half of failures occurring after that phase. No hyperparameter sweep or post-outcome retuning was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    args = parser.parse_args()
    records: list[dict[str, Any]] = []
    parameter_counts = set()
    for seed in args.seeds:
        records.extend(load_jsonl(args.input_dir / f"seed_{seed}.episodes.jsonl"))
        summary = json.loads((args.input_dir / f"seed_{seed}.summary.json").read_text())
        parameter_counts.add(int(summary["parameter_count"]))
    if len(parameter_counts) != 1:
        raise ValueError(f"parameter count mismatch: {parameter_counts}")
    tasks: dict[str, Any] = {}
    for task in TASK_NAMES:
        selected = [record for record in records if record["task"] == task]
        failures = [record for record in selected if not record["success"]]
        seed_direction = {}
        for seed in args.seeds:
            seed_records = [record for record in selected if record["policy_seed"] == seed]
            seed_direction[str(seed)] = {
                "success_rate": sum(record["success"] for record in seed_records) / len(seed_records),
                "late_phase_reach_rate": sum(record["pre_release_or_contact_phase_reached"] for record in seed_records) / len(seed_records),
            }
        success_rate = sum(record["success"] for record in selected) / len(selected)
        late_rate = sum(record["pre_release_or_contact_phase_reached"] for record in selected) / len(selected)
        late_failure_fraction = (
            sum(record["failure_after_target_phase"] for record in failures) / len(failures)
            if failures
            else None
        )
        qualifies_a = 0.30 <= success_rate <= 0.90
        qualifies_b = late_rate >= 0.70 and (
            not failures or float(late_failure_fraction) >= 0.50
        )
        tasks[task] = {
            "episodes": len(selected),
            "policy_seeds": list(args.seeds),
            "clean_success_rate": success_rate,
            "grasp_or_open_phase_reach_rate": sum(record["grasp_or_open_phase_reached"] for record in selected) / len(selected),
            "lift_or_transport_phase_reach_rate": sum(record["lift_or_transport_phase_reached"] for record in selected) / len(selected),
            "pre_release_or_contact_phase_reach_rate": late_rate,
            "failed_episode_count": len(failures),
            "failure_after_target_phase_fraction": late_failure_fraction,
            "mean_chunk_smoothness": float(np.mean([record["chunk_smoothness"] for record in selected])),
            "mean_policy_calls": float(np.mean([record["policy_calls"] for record in selected])),
            "mean_episode_length": float(np.mean([record["episode_length"] for record in selected])),
            "qualifies_clean_success_A": qualifies_a,
            "qualifies_late_phase_B": qualifies_b,
            "qualification_status": "qualified" if qualifies_a or qualifies_b else "not_qualified",
            "per_seed": seed_direction,
        }
    qualified_count = sum(item["qualification_status"] == "qualified" for item in tasks.values())
    summary = {
        "schema_version": 1,
        "actor": "HistoryConditionedStateACT",
        "parameter_count": parameter_counts.pop(),
        "training_seeds": list(args.seeds),
        "clean_rollout_episode_count": len(records),
        "episodes_per_task_seed": len(records) // (len(TASK_NAMES) * len(args.seeds)),
        "qualified_task_count": qualified_count,
        "minimum_qualified_tasks": 3,
        "actor_gate": "PASS" if qualified_count >= 3 else "BLOCKED_BY_ACTOR",
        "tasks": tasks,
    }
    rows = [
        {
            "task": task,
            "clean_success_rate": item["clean_success_rate"],
            "grasp_or_open_phase_reach_rate": item["grasp_or_open_phase_reach_rate"],
            "lift_or_transport_phase_reach_rate": item["lift_or_transport_phase_reach_rate"],
            "pre_release_or_contact_phase_reach_rate": item["pre_release_or_contact_phase_reach_rate"],
            "failure_after_target_phase_fraction": item["failure_after_target_phase_fraction"],
            "mean_chunk_smoothness": item["mean_chunk_smoothness"],
            "mean_policy_calls": item["mean_policy_calls"],
            "qualification_status": item["qualification_status"],
        }
        for task, item in tasks.items()
    ]
    atomic_write_json(args.output_dir / "qualification_summary.json", summary)
    atomic_write_csv(args.output_dir / "qualification_summary.csv", rows, list(rows[0]))
    atomic_write_text(args.output_dir / "qualification_report.md", render(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
