from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from s1.statistics import (  # noqa: E402
    FAMILY_A,
    FAMILY_B,
    OPERATORS,
    PHASE1_PREFIXES,
    PHASE2_PREFIXES,
    analyze_crossing,
    select_budget,
    summarize_grid,
)


def _row(
    *,
    task="task_a",
    event=0,
    init=None,
    split="evaluation",
    recovery_seed=0,
    operator="fresh_h4",
    prefix=2,
    tail=4,
    action=8,
    safe=0.0,
    status="ok",
):
    return {
        "event_instance_id": f"event-{task}-{event}",
        "task": task,
        "init_state_id": f"init-{task}-{event if init is None else init}",
        "split": split,
        "generator_actor_seed": 7,
        "recovery_actor_seed": recovery_seed,
        "operator": operator,
        "prefix_k": prefix,
        "tail_horizon": tail,
        "action_budget": action,
        "policy_call_cap": 2,
        "safe_success": safe,
        "pid": 1000 + event,
        "env_hash": "env-hash",
        "chunk_hash": "chunk-hash",
        "status": status,
    }


def _crossing_rows(kind="crossing"):
    rows = []
    for task in ("task_a", "task_b"):
        for event in range(6):
            if kind == "monotone":
                boundary_a = boundary_b = 8
            elif kind == "none" and event == 0:
                boundary_a, boundary_b = None, 16
            else:
                boundary_a, boundary_b = ((16, 8) if event % 2 == 0 else (4, 14))
            for prefix in PHASE2_PREFIXES:
                for operator in OPERATORS:
                    family_boundary = boundary_a if operator in FAMILY_A else boundary_b
                    for recovery_seed in (0, 1, 2):
                        safe = float(
                            family_boundary is not None and prefix <= family_boundary
                        )
                        rows.append(
                            _row(
                                task=task,
                                event=event,
                                recovery_seed=recovery_seed,
                                operator=operator,
                                prefix=prefix,
                                safe=safe,
                            )
                        )
    return rows


def _grid_rows(split="calibration"):
    rows = []
    arm_values = {
        "fresh_h4": 0.30,
        "fresh_h16": 0.40,
        "hold_1+fresh_h4": 0.50,
        "rollback_1+fresh_h16": 0.70,
    }
    for budget_index, (tail, action) in enumerate(((4, 8), (16, 32))):
        for task in ("task_a", "task_b"):
            for event in range(2):
                for prefix in PHASE1_PREFIXES:
                    for operator in OPERATORS:
                        for recovery_seed in (0, 1, 2):
                            value = arm_values[operator] - (0.1 if budget_index else 0.0)
                            rows.append(
                                _row(
                                    task=task,
                                    event=event + budget_index * 100,
                                    split=split,
                                    recovery_seed=recovery_seed,
                                    operator=operator,
                                    prefix=prefix,
                                    tail=tail,
                                    action=action,
                                    safe=value,
                                )
                            )
    return rows


def test_grid_reports_cluster_means_and_selects_ranked_budget():
    summary = summarize_grid(_grid_rows())
    assert summary["status"] == "COMPLETE"
    receipt = select_budget(summary)
    assert receipt["status"] == "SELECTED"
    assert receipt["selected_budget"] == {
        "tail_horizon": 4,
        "action_budget": 8,
        "policy_call_cap": 2,
    }
    selected = summary["budgets"]["tail_horizon=4/action_budget=8/policy_call_cap=2"]
    for task in ("task_a", "task_b"):
        metrics = selected["by_task"][task]
        assert metrics["oracle_best"] == pytest.approx(0.7)
        assert metrics["weakest_arm"] == pytest.approx(0.3)
        assert metrics["oracle_minus_weakest"] == pytest.approx(0.4)
        assert metrics["cluster_count"] == 2
        assert all(item["cluster_count"] == 2 for item in metrics["per_prefix"])
        assert len(metrics["event_prefix_summaries"]) == 10


def test_grid_hard_rejects_evaluation_for_selection():
    with pytest.raises(ValueError, match="calibration rows only"):
        summarize_grid(_grid_rows(split="evaluation"))


def test_missing_provenance_is_blocked_and_not_a_science_failure():
    rows = _grid_rows()[:]
    rows[0] = dict(rows[0])
    rows[0].pop("env_hash")
    summary = summarize_grid(rows)
    assert summary["status"] == "BLOCKED"
    assert "provenance" in " ".join(summary["blocking_reasons"])


def test_crossing_detects_two_directions_and_bootstrap_is_reproducible():
    rows = _crossing_rows()
    first = analyze_crossing(rows, replicates=100, seed=216214)
    second = analyze_crossing(rows, replicates=100, seed=216214)
    assert first["status"] == "COMPLETE"
    assert first["crossing"]["overall"]["both_defined"] == 12
    assert first["crossing"]["overall"]["crossing_A_rate"] == pytest.approx(0.5)
    assert first["crossing"]["overall"]["crossing_B_rate"] == pytest.approx(0.5)
    assert first["crossing"]["overall"]["minority_crossing_rate"] == pytest.approx(0.5)
    assert first["crossing"]["overall"]["spearman"] == pytest.approx(-1.0)
    assert first["crossing"]["overall"]["bootstrap"] == second["crossing"]["overall"]["bootstrap"]
    assert len(first["split_half_null"]["family_A"]["partitions"]) == 3
    assert len(first["split_half_null"]["family_B"]["partitions"]) == 3
    assert first["split_half_null"]["by_task"]["task_a"]["bootstrap_null_p95"] == 0.0


def test_monotone_rescaling_has_no_fabricated_spearman_or_crossing():
    result = analyze_crossing(_crossing_rows("monotone"), replicates=50)
    overall = result["crossing"]["overall"]
    assert overall["crossing_A_rate"] == 0.0
    assert overall["crossing_B_rate"] == 0.0
    assert overall["minority_crossing_rate"] == 0.0
    assert overall["spearman"] is None
    assert result["k3"]["status"] == "FAIL"


def test_undefined_boundaries_are_reported_separately():
    result = analyze_crossing(_crossing_rows("none"), replicates=30)
    overall = result["crossing"]["overall"]
    assert overall["missing_one"] == 2
    assert overall["missing_both"] == 0
    assert overall["both_defined"] == 10
    assert result["boundaries"][0]["boundary_A"] is None
    assert result["boundaries"][0]["boundary_B"] == 16


def test_crossing_missing_matrix_cell_is_blocked():
    rows = _crossing_rows()
    rows = [row for row in rows if not (row["event_instance_id"].endswith("task_a-0") and row["prefix_k"] == 2 and row["operator"] == "fresh_h4" and row["recovery_actor_seed"] == 1)]
    result = analyze_crossing(rows, replicates=10)
    assert result["status"] == "BLOCKED"
    assert any("incomplete Phase-2 cell" in reason for reason in result["blocking_reasons"])
