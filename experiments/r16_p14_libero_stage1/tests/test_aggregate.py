from __future__ import annotations

import json
import sys
from pathlib import Path

from r16_p14_stage1.aggregate import main
from r16_p14_stage1.io_utils import atomic_write_json
from r16_p14_stage1.settings import DEVELOPMENT_TASKS, EXECUTION_HORIZONS, TRAIN_SEEDS


def test_aggregate_produces_gate_decision_and_report(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "aggregate-test"
    shard_root = tmp_path / "shards"
    checkpoint_root = tmp_path / "checkpoints"
    output_dir = tmp_path / "output"
    winners = [
        "hold_and_replan",
        "cause_specific_local_repair",
        "hold_and_replan",
    ]
    for task_name, winner in zip(DEVELOPMENT_TASKS, winners, strict=True):
        groups = {}
        for seed in TRAIN_SEEDS:
            for horizon in EXECUTION_HORIZONS:
                groups[f"{task_name}|seed={seed}|h={horizon}"] = {
                    "episodes": 10,
                    "successes": 5,
                    "success_rate": 0.5,
                    "steps": 100,
                    "mean_episode_length": 10,
                }
            lineage = checkpoint_root / run_id / task_name / f"seed_{seed}"
            atomic_write_json(
                lineage / ".training_complete.json",
                {
                    "step": 8000,
                    "loss": 0.01,
                    "latest_checkpoint": str(lineage / "step_00008000"),
                },
            )
        shard = shard_root / task_name / run_id
        atomic_write_json(
            shard / "baseline_eval" / "summary.json",
            {"groups": groups},
        )
        operator_wins = {
            "trim_and_replan": 0,
            "hold_and_replan": 0,
            "bounded_rollback_and_replan": 0,
            "cause_specific_local_repair": 0,
        }
        operator_wins[winner] = 20
        atomic_write_json(
            shard / "oracle_audit" / task_name / "summary.json",
            {
                "candidate_count": 30,
                "usable_unsafe_or_near_unsafe_chunks": 30,
                "replay_determinism_rate": 1.0,
                "median_intervention_window": 3,
                "median_safe_prefix_retention": 0.5,
                "no_intervention_violation_count": 30,
                "oracle_violation_count": 10,
                "relative_rework_reduction_vs_same_trigger_full_replan": 0.25,
                "operator_wins": operator_wins,
            },
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate",
            "--run-id",
            run_id,
            "--shard-root",
            str(shard_root),
            "--checkpoint-root",
            str(checkpoint_root),
            "--output-dir",
            str(output_dir),
        ],
    )
    main()
    summary = json.loads((output_dir / "experiment_summary.json").read_text())
    assert summary["decision"] == "PROCEED_TO_LEARNED_RISK"
    assert all(summary["gates"].values())
    assert (output_dir / "report.md").read_text().startswith("# R16-P14")
    complete = json.loads((output_dir / "EVALUATION_COMPLETE.json").read_text())
    assert complete["status"] == "complete"
