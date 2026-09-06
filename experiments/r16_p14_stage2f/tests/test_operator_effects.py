from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from s1.consolidate import (  # noqa: E402
    BUDGETS,
    OPERATORS,
    PHASE1_PREFIXES,
    SEEDS,
    TASKS,
    consolidate_phase1,
)
from s1.summarize_operator_effects import (  # noqa: E402
    PAIR_DEFINITIONS,
    summarize_operator_effects,
)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _row(
    task: str,
    event_index: int,
    init_state_id: str,
    budget: tuple[int, int, int],
    operator: str,
    prefix: int,
    seed: int,
    *,
    structural: bool = False,
    hash_override: str | None = None,
    same_fresh: bool = False,
) -> dict:
    tail, action, cap = budget
    # The first nineteen cream events share one init cluster and the last
    # event has a second init cluster.  Their values make event weighting and
    # equal-init-cluster weighting visibly different.
    if task == TASKS[0] and init_state_id == "cream-a":
        values = {
            "fresh_h4": 0.1,
            "fresh_h16": 0.2,
            "hold_1+fresh_h4": 0.3,
            "rollback_1+fresh_h16": 0.4,
        }
    elif task == TASKS[0]:
        values = {
            "fresh_h4": 0.9,
            "fresh_h16": 0.8,
            "hold_1+fresh_h4": 0.7,
            "rollback_1+fresh_h16": 0.6,
        }
    else:
        values = {
            "fresh_h4": 0.2,
            "fresh_h16": 0.3,
            "hold_1+fresh_h4": 0.4,
            "rollback_1+fresh_h16": 0.5,
        }
    if same_fresh and tail == 4 and operator == "fresh_h16":
        values["fresh_h16"] = values["fresh_h4"]
    event_id = f"event-{task}-{event_index}"
    stream_hash = hash_override or f"stream-{task}-{event_index}-{prefix}-{seed}"
    row = {
        "event_instance_id": event_id,
        "task": task,
        "init_state_id": init_state_id,
        "split": "calibration",
        "generator_actor_seed": 7,
        "recovery_actor_seed": seed,
        "operator": operator,
        "prefix_k": prefix,
        "tail_horizon": tail,
        "action_budget": action,
        "policy_call_cap": cap,
        "safe_success": 0.0 if structural else values[operator],
        "pid": 1234,
        "env_hash": "env-hash",
        "chunk_hash": "chunk-hash",
        "status": "BLOCKED" if structural else "COMPLETE",
        "action_stream_hash": stream_hash,
        "is_reference": False,
    }
    if structural:
        row["error_type"] = "PrefixOutsideTaskHorizon"
        row["structurally_excluded"] = True
        row["safe_success"] = 0.0
    return row


def _fixture(
    root: Path,
    *,
    excluded: bool = False,
    hash_mismatch: bool = False,
    incomplete_summary: bool = False,
    same_fresh: bool = False,
    counts: tuple[int, int] = (20, 20),
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    excluded_events = []
    for task_index, task in enumerate(TASKS):
        for event_index in range(counts[task_index]):
            if task == TASKS[0]:
                init_state_id = "cream-a" if event_index < 19 else "cream-b"
            else:
                init_state_id = "bowl-a"
            is_excluded = (
                excluded
                and task == TASKS[1]
                and event_index == counts[1] - 1
            )
            if is_excluded:
                excluded_events.append(
                    {
                        "task": task,
                        "event_instance_id": f"event-{task}-{event_index}",
                        "init_state_id": init_state_id,
                        "split": "calibration",
                        "generator_actor_seed": 7,
                    }
                )
            for budget in BUDGETS:
                for operator in OPERATORS:
                    for prefix in PHASE1_PREFIXES:
                        for seed in SEEDS:
                            override = None
                            if (
                                hash_mismatch
                                and budget[0] == 4
                                and budget[1] == 8
                                and task == TASKS[0]
                                and event_index == 0
                                and operator == "fresh_h16"
                                and prefix == PHASE1_PREFIXES[0]
                                and seed == SEEDS[0]
                            ):
                                override = "mismatching-stream"
                            rows.append(
                                _row(
                                    task,
                                    event_index,
                                    init_state_id,
                                    budget,
                                    operator,
                                    prefix,
                                    seed,
                                    structural=is_excluded,
                                    hash_override=override,
                                    same_fresh=same_fresh,
                                )
                            )
    phase = root / "phase1"
    phase.mkdir(parents=True, exist_ok=True)
    (phase / "grid_rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "analysis": "grid",
        "status": (
            "COMPLETE_AVAILABLE_REQUESTS_SAMPLE_SHORTFALL"
            if incomplete_summary
            else "COMPLETE"
        ),
        "blocked": False,
        "split": "calibration",
        "sample_complete": not incomplete_summary,
        "planned_sample_complete": not incomplete_summary,
        "planned_events": 20,
        "observed_events_by_task": {
            task: counts[index] for index, task in enumerate(TASKS)
        },
        "shortfall": {
            task: 20 - counts[index] for index, task in enumerate(TASKS)
        },
        "grid_row_count": len(rows),
        "support_grid_row_count": sum(
            not (
                excluded
                and row["task"] == TASKS[1]
                and row["event_instance_id"].endswith(f"-{counts[1] - 1}")
            )
            for row in rows
        ),
        "structurally_excluded_events": excluded_events,
        "structurally_excluded_event_count": len(excluded_events),
        "completeness": {
            "status": "COMPLETE",
            "planned_sample_complete": not incomplete_summary,
            "observed_events_by_task": {
                task: counts[index] for index, task in enumerate(TASKS)
            },
        },
        "selection": (
            {
                "status": "BLOCKED",
                "selected_budget": None,
            }
            if incomplete_summary
            else {
                "status": "SELECTED",
                "selected_budget": {
                    "tail_horizon": 4,
                    "action_budget": 8,
                    "policy_call_cap": 8,
                },
            }
        ),
    }
    _write(phase / "summary.json", summary)
    return rows, summary


def test_full_support_uses_equal_init_clusters_and_paired_events(tmp_path):
    _fixture(tmp_path, same_fresh=True)
    result = summarize_operator_effects(tmp_path, replicates=100, seed=216214)
    assert result["status"] == "COMPLETE"
    assert result["selection_performed"] is False
    assert result["evaluation_read"] is False
    assert len(result["budgets"]) == 9

    budget_name = "tail_horizon=4/action_budget=8/policy_call_cap=8"
    cream = result["budgets"][budget_name]["by_task"][TASKS[0]]
    # Event mean is (19*.1+.9)/20=.14, while two init clusters are (.1+.9)/2=.5.
    assert cream["absolute"]["fresh_h4"]["estimate"] == pytest.approx(0.5)
    assert cream["event_count"] == 20
    assert cream["init_count"] == 2
    assert cream["branch_count"] == 20 * 4 * 5 * 3
    pair = cream["paired_differences"]["hold_1+fresh_h4-minus-fresh_h4"]
    assert pair["estimate"] == pytest.approx(0.0)
    assert pair["paired_event_count"] == 20
    assert pair["paired_init_count"] == 2
    assert pair["left_branch_count"] == 20 * 5 * 3
    assert pair["right_branch_count"] == 20 * 5 * 3
    assert pair["bootstrap_replicates"] == 100
    assert pair["bootstrap_seed"] == 216214
    assert result["action_stream_check"]["status"] == "EQUIVALENT_OBSERVED"
    assert result["action_stream_check"]["checked_pairs"] == 3 * 2 * 20 * 5 * 3


def test_available_shortfall_and_structural_blocked_rows_are_excluded_not_zero_failures(tmp_path):
    _fixture(
        tmp_path,
        excluded=True,
        incomplete_summary=True,
        counts=(19, 11),
    )
    result = summarize_operator_effects(tmp_path, replicates=40)
    assert result["status"] == "COMPLETE"
    assert result["summary_status"] == "COMPLETE_AVAILABLE_REQUESTS_SAMPLE_SHORTFALL"
    assert result["sample_complete"] is False
    bowl = result["budgets"][
        "tail_horizon=4/action_budget=8/policy_call_cap=8"
    ]["by_task"][TASKS[1]]
    assert bowl["raw_event_count"] == 11
    assert bowl["event_count"] == 10
    assert bowl["excluded_event_count"] == 1
    assert bowl["absolute"]["fresh_h4"]["estimate"] == pytest.approx(0.2)
    assert bowl["branch_count"] == 10 * 4 * 5 * 3
    assert result["structurally_excluded_events"][0]["event_instance_id"].endswith("-10")
    check = result["action_stream_check"]
    assert check["excluded_branch_pairs_skipped"] == 45
    assert check["excluded_branch_rows_skipped"] == 90



def test_accepts_real_available_consolidation_schema(tmp_path):
    # Reuse the existing phase1 collector fixture so this exercises the exact
    # COMPLETE_AVAILABLE_REQUESTS_SAMPLE_SHORTFALL summary emitted by
    # consolidate_phase1, including 19/11 source identities and one structural
    # bowl event.  No evaluation tree is created or read.
    from test_partial_sample_s1 import _phase1_input

    _phase1_input(tmp_path, (19, 11), structural_last_bowl=True)
    consolidated = consolidate_phase1(tmp_path, tmp_path)
    assert consolidated["status"] == "COMPLETE_AVAILABLE_REQUESTS_SAMPLE_SHORTFALL"
    result = summarize_operator_effects(tmp_path, replicates=20)
    assert result["status"] == "COMPLETE"
    assert result["sample_complete"] is False
    assert result["support_event_count_by_task"] == {
        TASKS[0]: 19,
        TASKS[1]: 10,
    }
    assert result["budgets"][
        "tail_horizon=4/action_budget=8/policy_call_cap=8"
    ]["by_task"][TASKS[1]]["excluded_event_count"] == 1

def test_action_stream_check_keeps_mismatch_evidence(tmp_path):
    _fixture(tmp_path, hash_mismatch=True, same_fresh=True)
    result = summarize_operator_effects(tmp_path, replicates=20)
    check = result["action_stream_check"]
    assert check["status"] == "MISMATCH"
    assert check["safe_success_mismatch_count"] == 0
    assert check["action_stream_hash_mismatch_count"] == 1
    assert check["checked_pairs"] == 3 * 2 * 20 * 5 * 3
    assert any("action stream hash differs" in item["reasons"] for item in check["mismatches"])


def test_nonstructural_blocked_row_cannot_masquerade_as_failure(tmp_path):
    _fixture(tmp_path)
    path = tmp_path / "phase1" / "grid_rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["status"] = "BLOCKED_BY_RUNTIME_ERROR"
    rows[0]["error_type"] = "RuntimeError"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = summarize_operator_effects(tmp_path, replicates=10)
    assert result["status"] == "BLOCKED"
    assert any("non-COMPLETE status" in reason for reason in result["blocking_reasons"])


def test_partial_summary_denies_grid_read(tmp_path):
    phase = tmp_path / "phase1"
    phase.mkdir(parents=True)
    (phase / "summary.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE_AVAILABLE_REQUESTS_SAMPLE_SHORTFALL",
                "blocked": False,
                "sample_complete": False,
                "completeness": {"status": "INCOMPLETE"},
            }
        )
    )
    (phase / "grid_rows.jsonl").write_text("{not-json\n")
    result = summarize_operator_effects(tmp_path, replicates=10)
    assert result["status"] == "BLOCKED"
    assert result["grid_rows_read"] is False
    assert any(
        "INCOMPLETE" in reason or "sample_complete" in reason
        for reason in result["blocking_reasons"]
    )


def test_cli_writes_explicit_output_path(tmp_path):
    _fixture(tmp_path)
    output = tmp_path / "reports" / "operator_effects.json"
    from s1.summarize_operator_effects import main

    code = main(
        [
            "--input-root",
            str(tmp_path),
            "--output",
            str(output),
            "--replicates",
            "10",
            "--seed",
            "216214",
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text())
    assert payload["status"] == "COMPLETE"
    assert payload["bootstrap_seed"] == 216214
