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
    REFERENCE_PREFIXES,
    SEEDS,
    TASKS,
    consolidate_phase1,
)
from s1.statistics import select_budget  # noqa: E402


ARM_VALUES = {
    "fresh_h4": 0.30,
    "fresh_h16": 0.40,
    "hold_1+fresh_h4": 0.50,
    "rollback_1+fresh_h16": 0.70,
}


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _phase1_input(
    root: Path,
    counts: tuple[int, int],
    *,
    structural_last_bowl: bool = False,
    drop_branch: tuple[str, int, str, int, int] | None = None,
) -> None:
    rows_by_task = {task: [] for task in TASKS}
    source_events = []
    for task_index, (task, count) in enumerate(zip(TASKS, counts)):
        for event_index in range(count):
            init_state_id = str(10 + task_index * 100 + event_index)
            event_id = f"event-{task_index}-{event_index}"
            source_events.append(
                {
                    "task": task,
                    "event_instance_id": event_id,
                    "init_state_id": init_state_id,
                    "split": "calibration",
                    "actor_seed": 7,
                }
            )
            for recovery_seed in SEEDS:
                for tail_horizon, action_budget, policy_call_cap in BUDGETS:
                    for operator in OPERATORS:
                        for prefix_k in PHASE1_PREFIXES:
                            is_structural = (
                                structural_last_bowl
                                and task_index == 1
                                and event_index == count - 1
                                and prefix_k in (8, 12, 16)
                            )
                            key = (task, event_index, operator, prefix_k, recovery_seed)
                            if drop_branch == key:
                                continue
                            row = {
                                "event_instance_id": event_id,
                                "task": task,
                                "init_state_id": init_state_id,
                                "split": "calibration",
                                "generator_actor_seed": 7,
                                "recovery_actor_seed": recovery_seed,
                                "operator": operator,
                                "prefix_k": prefix_k,
                                "tail_horizon": tail_horizon,
                                "action_budget": action_budget,
                                "policy_call_cap": policy_call_cap,
                                "safe_success": ARM_VALUES[operator],
                                "pid": 1000,
                                "env_hash": "env-hash",
                                "chunk_hash": "chunk-hash",
                                "status": "BLOCKED" if is_structural else "COMPLETE",
                                "is_reference": False,
                            }
                            if is_structural:
                                row["error_type"] = "PrefixOutsideTaskHorizon"
                                row["env_hash"] = None
                            rows_by_task[task].append(row)
                    for operator, prefix_k in REFERENCE_PREFIXES.items():
                        is_structural = (
                            structural_last_bowl
                            and task_index == 1
                            and event_index == count - 1
                            and prefix_k in (6, 10)
                        )
                        rows_by_task[task].append(
                            {
                                "event_instance_id": event_id,
                                "task": task,
                                "init_state_id": init_state_id,
                                "split": "calibration",
                                "generator_actor_seed": 7,
                                "recovery_actor_seed": recovery_seed,
                                "operator": operator,
                                "prefix_k": prefix_k,
                                "tail_horizon": tail_horizon,
                                "action_budget": action_budget,
                                "policy_call_cap": policy_call_cap,
                                "safe_success": 0.5,
                                "pid": 1000,
                                "env_hash": "env-hash",
                                "chunk_hash": "chunk-hash",
                                "status": "BLOCKED" if is_structural else "COMPLETE",
                                "error_type": "PrefixOutsideTaskHorizon" if is_structural else None,
                                "env_hash": None if is_structural else "env-hash",
                                "is_reference": True,
                            }
                        )
    (root / "phase0b" / "calibration_events.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (root / "phase0b" / "calibration_events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in source_events),
        encoding="utf-8",
    )
    for task, count in zip(TASKS, counts):
        _write(root / "phase1" / "shards" / task / "rows.json", {"rows": rows_by_task[task]})
        _write(
            root / "phase1" / f"completion_{task}.json",
            {
                "task": task,
                "phase": "grid",
                "events": count,
                "planned_events": 20,
                "planned_sample_complete": count == 20,
                "missing_planned_events": max(0, 20 - count),
                "status": "COMPLETE" if count == 20 else "COMPLETE_AVAILABLE_REQUESTS_SAMPLE_SHORTFALL",
            },
        )


def _assert_metrics(summary, counts: tuple[int, int], support_counts: tuple[int, int] | None = None) -> None:
    assert summary["status"] == "COMPLETE_AVAILABLE_REQUESTS_SAMPLE_SHORTFALL"
    assert summary["sample_complete"] is False
    assert summary["observed_events_by_task"] == {
        TASKS[0]: counts[0],
        TASKS[1]: counts[1],
    }
    assert summary["shortfall"] == {
        TASKS[0]: 20 - counts[0],
        TASKS[1]: 20 - counts[1],
    }
    records = summary["grid"].values() if isinstance(summary["grid"], dict) else summary["grid"]
    support_counts = counts if support_counts is None else support_counts
    for record in records:
        for task, expected_count in zip(TASKS, support_counts):
            metrics = record["by_task"][task]
            assert metrics["event_count"] == expected_count
            assert metrics["oracle_best"] == pytest.approx(0.70)
            assert metrics["weakest_arm"] == pytest.approx(0.30)
            assert metrics["oracle_minus_weakest"] == pytest.approx(0.40)


def test_partial_19_11_reports_real_grid_and_blocks_receipt(tmp_path):
    counts = (19, 11)
    _phase1_input(tmp_path, counts)
    result = consolidate_phase1(tmp_path, tmp_path)
    _assert_metrics(result, counts)
    assert result["selection"]["status"] == "BLOCKED"
    assert result["selection"]["selected_budget"] is None
    receipt = json.loads((tmp_path / "phase1" / "selection_receipt.json").read_text())
    assert receipt["status"] == "BLOCKED"
    assert receipt["selected_budget"] is None
    assert len((tmp_path / "phase1" / "grid_rows.jsonl").read_text().splitlines()) == 19 * 540 + 11 * 540


def test_structural_horizon_rows_are_retained_but_excluded_from_support(tmp_path):
    _phase1_input(tmp_path, (19, 11), structural_last_bowl=True)
    result = consolidate_phase1(tmp_path, tmp_path)
    _assert_metrics(result, (19, 11), (19, 10))
    assert result["support_event_count_by_task"] == {TASKS[0]: 19, TASKS[1]: 10}
    assert result["structurally_excluded_event_count"] == 1
    assert result["structurally_excluded_row_count"] == 378
    assert result["selection"]["status"] == "BLOCKED"
    raw_core = (tmp_path / "phase1" / "grid_rows.jsonl").read_text().splitlines()
    raw_reference = (tmp_path / "phase1" / "reference_rows.jsonl").read_text().splitlines()
    assert sum('"status": "BLOCKED"' in line for line in raw_core) == 324
    assert sum('"status": "BLOCKED"' in line for line in raw_reference) == 54


def test_completion_receipt_cannot_replace_source_identity(tmp_path):
    _phase1_input(tmp_path, (19, 11))
    path = tmp_path / "phase1" / f"completion_{TASKS[0]}.json"
    payload = json.loads(path.read_text())
    payload["events"] = 20
    payload["planned_sample_complete"] = True
    payload["missing_planned_events"] = 0
    path.write_text(json.dumps(payload))
    result = consolidate_phase1(tmp_path, tmp_path)
    assert result["status"] == "BLOCKED"
    assert any("disagrees with source event count" in reason for reason in result["blocking_reasons"])
    assert not (tmp_path / "phase1" / "selection_receipt.json").exists()


def test_missing_branch_blocks_without_filling_zero(tmp_path):
    _phase1_input(
        tmp_path,
        (19, 11),
        drop_branch=(TASKS[0], 0, "fresh_h4", 2, 7),
    )
    result = consolidate_phase1(tmp_path, tmp_path)
    assert result["status"] == "BLOCKED"
    assert any("missing" in reason for reason in result["blocking_reasons"])
    assert not (tmp_path / "phase1" / "selection_receipt.json").exists()


def test_n20_n20_path_can_select_with_complete_source(tmp_path):
    _phase1_input(tmp_path, (20, 20))
    result = consolidate_phase1(tmp_path, tmp_path)
    assert result["status"] == "COMPLETE"
    assert result["sample_complete"] is True
    assert result["selection"]["status"] == "SELECTED"
    assert result["selection"]["selected_budget"] == {
        "tail_horizon": 4,
        "action_budget": 8,
        "policy_call_cap": 8,
    }


def test_n0_source_is_blocked_and_not_declared_complete(tmp_path):
    _phase1_input(tmp_path, (19, 0))
    result = consolidate_phase1(tmp_path, tmp_path)
    assert result["status"] == "BLOCKED"
    assert any("no events" in reason or "expected both tasks" in reason for reason in result["blocking_reasons"])
    assert not (tmp_path / "phase1" / "selection_receipt.json").exists()


def test_direct_selection_guard_rejects_annotated_shortfall():
    receipt = select_budget(
        {
            "status": "COMPLETE",
            "blocked": False,
            "sample_complete": False,
            "observed_events_by_task": dict(zip(TASKS, (19, 11))),
            "shortfall": dict(zip(TASKS, (1, 9))),
            "grid": [],
            "tasks": list(TASKS),
        }
    )
    assert receipt["status"] == "BLOCKED"
    assert receipt["selected_budget"] is None
