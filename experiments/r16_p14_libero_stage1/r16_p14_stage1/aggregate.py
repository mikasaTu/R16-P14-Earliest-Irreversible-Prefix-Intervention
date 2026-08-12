from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json
from .settings import DEVELOPMENT_TASKS, EXECUTION_HORIZONS, TRAIN_SEEDS


TASK_LABELS = {
    "open_the_middle_drawer_of_the_cabinet": "Open middle drawer / mechanism obstruction",
    "put_the_bowl_on_the_plate": "Bowl on plate / target shift",
    "put_the_wine_bottle_on_the_rack": "Wine bottle on rack / grasp slip",
}


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-episodes", type=int, default=10)
    parser.add_argument("--oracle-candidates", type=int, default=30)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object at {path}")
    return value


def baseline_task_summary(raw: dict[str, Any], task_name: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for horizon in EXECUTION_HORIZONS:
        groups = [
            value
            for key, value in raw["groups"].items()
            if key.startswith(f"{task_name}|") and key.endswith(f"|h={horizon}")
        ]
        episodes = sum(int(group["episodes"]) for group in groups)
        successes = sum(int(group["successes"]) for group in groups)
        output[str(horizon)] = {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes if episodes else None,
        }
    return output


def main() -> None:
    args = parse_args()
    baseline: dict[str, Any] = {}
    oracle: dict[str, Any] = {}
    checkpoint_evidence: list[dict[str, Any]] = []
    for task_name in DEVELOPMENT_TASKS:
        shard = args.shard_root / task_name / args.run_id
        baseline_raw = load_json(shard / "baseline_eval" / "summary.json")
        oracle_raw = load_json(
            shard / "oracle_audit" / task_name / "summary.json"
        )
        baseline[task_name] = baseline_task_summary(baseline_raw, task_name)
        oracle[task_name] = oracle_raw
        for seed in TRAIN_SEEDS:
            lineage = (
                args.checkpoint_root / args.run_id / task_name / f"seed_{seed}"
            )
            complete = load_json(lineage / ".training_complete.json")
            checkpoint_evidence.append(
                {
                    "task": task_name,
                    "seed": seed,
                    "step": int(complete["step"]),
                    "loss": float(complete["loss"]),
                    "checkpoint": complete["latest_checkpoint"],
                }
            )

    usable_gate = all(
        int(value["candidate_count"]) >= args.oracle_candidates
        and int(value["usable_unsafe_or_near_unsafe_chunks"]) >= 30
        for value in oracle.values()
    )
    window_task_count = sum(
        value["median_intervention_window"] is not None
        and float(value["median_intervention_window"]) >= 2
        for value in oracle.values()
    )
    no_intervention = sum(
        int(value["no_intervention_violation_count"]) for value in oracle.values()
    )
    oracle_unsafe = sum(int(value["oracle_violation_count"]) for value in oracle.values())
    violation_reduction = (
        (no_intervention - oracle_unsafe) / no_intervention
        if no_intervention
        else None
    )
    rework_values = [
        float(value["relative_rework_reduction_vs_same_trigger_full_replan"])
        for value in oracle.values()
        if value["relative_rework_reduction_vs_same_trigger_full_replan"] is not None
    ]
    rework_reduction = statistics.median(rework_values) if rework_values else None
    retention_values = [
        float(value["median_safe_prefix_retention"])
        for value in oracle.values()
        if value["median_safe_prefix_retention"] is not None
    ]
    median_retention = statistics.median(retention_values) if retention_values else None
    replay_rate = min(float(value["replay_determinism_rate"]) for value in oracle.values())
    cause_winners = []
    for value in oracle.values():
        wins = value["operator_wins"]
        winner, count = max(wins.items(), key=lambda item: (item[1], item[0]))
        if count:
            cause_winners.append(winner)

    gates = {
        "minimum_30_usable_per_task": usable_gate,
        "at_least_two_tasks_median_window_ge_2": window_task_count >= 2,
        "relative_unsafe_outcome_reduction_ge_30pct": (
            violation_reduction is not None and violation_reduction >= 0.30
        ),
        "relative_rework_reduction_ge_20pct": (
            rework_reduction is not None and rework_reduction >= 0.20
        ),
        "median_safe_prefix_retention_ge_30pct": (
            median_retention is not None and median_retention >= 0.30
        ),
        "two_distinct_cause_operator_winners": len(set(cause_winners)) >= 2,
        "replay_determinism_ge_99pct": replay_rate >= 0.99,
    }
    if all(gates.values()):
        decision = "PROCEED_TO_LEARNED_RISK"
    elif window_task_count == 0 and all(
        int(value["usable_unsafe_or_near_unsafe_chunks"]) >= 30
        for value in oracle.values()
    ):
        decision = "KILL_PREFIX_HYPOTHESIS"
    else:
        decision = "REVISE_TASKS_OR_PERTURBATIONS"

    summary = {
        "schema_version": 1,
        "status": "complete",
        "run_id": args.run_id,
        "completed_at_unix": time.time(),
        "scope": "preliminary_stage1_oracle_pilot_not_final_algorithm_claim",
        "training": checkpoint_evidence,
        "baseline": baseline,
        "oracle": oracle,
        "aggregate_metrics": {
            "tasks_with_median_window_ge_2": window_task_count,
            "no_intervention_unsafe_outcomes": no_intervention,
            "oracle_unsafe_outcomes": oracle_unsafe,
            "relative_unsafe_outcome_reduction": violation_reduction,
            "median_relative_rework_reduction": rework_reduction,
            "median_safe_prefix_retention": median_retention,
            "minimum_replay_determinism_rate": replay_rate,
            "cause_operator_winners": cause_winners,
        },
        "gates": gates,
        "decision": decision,
        "pilot_volume": {
            "clean_episodes_per_seed_per_horizon": args.baseline_episodes,
            "policy_seeds": list(TRAIN_SEEDS),
            "horizons": list(EXECUTION_HORIZONS),
            "oracle_candidates_per_task": args.oracle_candidates,
            "oracle_policy_seed": 7,
            "prefix_stride": 2,
            "branch_budget": 96,
        },
        "limitations": [
            "The clean pilot uses 10 rather than the preregistered final 50 episodes per seed and horizon.",
            "The oracle audit uses policy seed 7; the clean baseline covers all three policy seeds.",
            "Physical recoverability is a privileged scripted-recovery proxy, not an exhaustive dynamics proof.",
            "The policy is state-observation chunked BC; no learned risk head, VLA, or world model is evaluated.",
        ],
    }
    atomic_write_json(args.output_dir / "experiment_summary.json", summary)
    atomic_write_json(
        args.output_dir / "decision.json",
        {"decision": decision, "gates": gates, "run_id": args.run_id},
    )

    lines = [
        "# R16-P14 Stage-1 Oracle Prefix Feasibility Audit — LIBERO pilot",
        "",
        f"Decision: `{decision}`",
        "",
        "## Clean baseline",
        "",
        "| Task | h=1 | h=4 | h=8 | h=16 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task_name in DEVELOPMENT_TASKS:
        cells = [
            f"{100 * baseline[task_name][str(h)]['success_rate']:.1f}%"
            for h in EXECUTION_HORIZONS
        ]
        lines.append(f"| {TASK_LABELS[task_name]} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Oracle prefix audit",
            "",
            "| Task | Candidates | Usable unsafe | Median window | Safe-prefix retention | Replay |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for task_name in DEVELOPMENT_TASKS:
        value = oracle[task_name]
        retention = value["median_safe_prefix_retention"]
        retention_text = "n/a" if retention is None else f"{100 * retention:.1f}%"
        lines.append(
            f"| {TASK_LABELS[task_name]} | {value['candidate_count']} | "
            f"{value['usable_unsafe_or_near_unsafe_chunks']} | "
            f"{value['median_intervention_window']} | "
            f"{retention_text} | {100 * value['replay_determinism_rate']:.1f}% |"
        )
    lines.extend(["", "## Gates", ""])
    for name, passed in gates.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.append("")
    atomic_write_text(args.output_dir / "report.md", "\n".join(lines))
    atomic_write_json(
        args.output_dir / "EVALUATION_COMPLETE.json",
        {
            "status": "complete",
            "run_id": args.run_id,
            "decision": decision,
            "summary": str(args.output_dir / "experiment_summary.json"),
            "report": str(args.output_dir / "report.md"),
            "uid": os.getuid(),
            "gid": os.getgid(),
            "completed_at_unix": time.time(),
        },
    )
    print(
        f"R16P14_EVALUATION_COMPLETE decision={decision} "
        f"summary={args.output_dir / 'experiment_summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
