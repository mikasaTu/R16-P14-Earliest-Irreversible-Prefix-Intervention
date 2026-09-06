from __future__ import annotations

import copy
import sys

import numpy as np
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
    _crossing_scope,
    _family_success,
    _reference_summary,
)


def _row(
    *,
    task="task_a",
    event=0,
    init=None,
    split="evaluation",
    recovery_seed=7,
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
                    for recovery_seed in (7, 17, 29):
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


def _split_null_counterexample_rows():
    """Make the family bootstrap tails differ from the merged null tail."""
    rows = []
    for task in ("task_a", "task_b"):
        for event in range(20):
            for operator in OPERATORS:
                for recovery_seed in (7, 17, 29):
                    if operator in FAMILY_A and event < 2:
                        bound = ({7: 16, 17: 4, 29: 4} if event == 0 else {7: 4, 17: 16, 29: 16})[recovery_seed]
                    elif operator in FAMILY_B and event < 10:
                        bound = ({7: 16, 17: 4, 29: 4} if event % 2 == 0 else {7: 4, 17: 16, 29: 16})[recovery_seed]
                    else:
                        bound = 8
                    for prefix in PHASE2_PREFIXES:
                        rows.append(
                            _row(
                                task=task,
                                event=event,
                                recovery_seed=recovery_seed,
                                operator=operator,
                                prefix=prefix,
                                safe=float(prefix <= bound),
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
                        for recovery_seed in (7, 17, 29):
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
    rows = [row for row in rows if not (row["event_instance_id"].endswith("task_a-0") and row["prefix_k"] == 2 and row["operator"] == "fresh_h4" and row["recovery_actor_seed"] == 17)]
    result = analyze_crossing(rows, replicates=10)
    assert result["status"] == "BLOCKED"
    assert any("incomplete Phase-2 cell" in reason for reason in result["blocking_reasons"])


def test_crossing_bootstrap_keeps_event_mean_with_unequal_cluster_sizes():
    rows = []
    for init_state_id, count, boundary_a, boundary_b in (
        ("init-a", 3, 16, 4),
        ("init-b", 1, 4, 16),
    ):
        for event in range(count):
            rows.append(
                {
                    "task": "task_a",
                    "init_state_id": init_state_id,
                    "event_instance_id": f"{init_state_id}-{event}",
                    "boundary_A": boundary_a,
                    "boundary_B": boundary_b,
                }
            )
    result = _crossing_scope(rows, "task_a", replicates=100, seed=216214)
    assert result["minority_crossing_rate"] == pytest.approx(.25)
    assert result["bootstrap"]["estimate"] == pytest.approx(.25)
    assert result["bootstrap"]["cluster_count"] == 2


def test_split_half_null_merges_family_draws_once_and_excludes_point_p95():
    result = analyze_crossing(_split_null_counterexample_rows(), replicates=200, seed=216214)
    null = result["split_half_null"]["by_task"]["task_a"]
    family_a = result["split_half_null"]["family_A"]["by_task"]["task_a"]
    family_b = result["split_half_null"]["family_B"]["by_task"]["task_a"]
    assert len(null["bootstrap_null_draws"]) == 6 * 200
    assert null["p95"] == pytest.approx(0.20)
    assert null["p95"] == pytest.approx(
        __import__("numpy").quantile(null["bootstrap_null_draws"], .95)
    )
    assert null["p95"] != max(
        family_a["bootstrap_null_p95"],
        family_b["bootstrap_null_p95"],
        null["point_p95"],
    )


def test_wrong_recovery_seed_triplet_is_blocked():
    rows = _crossing_rows()
    for row in rows:
        row["recovery_actor_seed"] = {7: 0, 17: 1, 29: 2}[row["recovery_actor_seed"]]
    result = analyze_crossing(rows, replicates=10)
    assert result["status"] == "BLOCKED"
    assert any("expected recovery actor seeds" in reason for reason in result["blocking_reasons"])


def _family_event(*, task, init_state_id, event_instance_id, value, operators):
    return {
        "task": task,
        "init_state_id": init_state_id,
        "event_instance_id": event_instance_id,
        "values": {
            prefix: {
                operator: {"7": value, "17": value, "29": value}
                for operator in operators
            }
            for prefix in PHASE2_PREFIXES
        },
    }


def test_phase2_family_ci_resamples_init_clusters_not_independent_events():
    events = [
        _family_event(
            task="task_a",
            init_state_id="init-a",
            event_instance_id=f"event-a-{index}",
            value=1.0,
            operators=FAMILY_A,
        )
        for index in range(3)
    ]
    events.append(
        _family_event(
            task="task_a",
            init_state_id="init-b",
            event_instance_id="event-b-0",
            value=0.0,
            operators=FAMILY_A,
        )
    )

    first = _family_success(events, FAMILY_A)
    second = _family_success(events, FAMILY_A)
    scope = first["by_task"]["task_a"]
    bootstrap = scope["bootstrap"]

    assert scope["oracle"] == pytest.approx(0.5)
    assert scope["arm_means"]["fresh_h4"] == pytest.approx(0.5)
    assert bootstrap["cluster_count"] == 2
    assert bootstrap["cluster_unit"] == ["task", "init_state_id"]
    assert bootstrap["replicates"] == 10000
    assert bootstrap["seed"] == 216214
    assert bootstrap["oracle"]["estimate"] == pytest.approx(0.5)
    assert bootstrap["oracle"]["ci95"] == second["by_task"]["task_a"]["bootstrap"]["oracle"]["ci95"]
    assert bootstrap["arm_means"]["fresh_h4"]["ci95"] == bootstrap["oracle"]["ci95"]

    # An independent-event bootstrap sees three repeated observations from
    # init-a and therefore has a different lower tail from the prescribed
    # two-init cluster bootstrap.
    rng = np.random.default_rng(216214)
    event_values = [1.0, 1.0, 1.0, 0.0]
    independent_draws = [
        float(np.mean([event_values[int(index)] for index in rng.integers(0, 4, size=4)]))
        for _ in range(10000)
    ]
    independent_ci = [
        float(np.quantile(independent_draws, 0.025)),
        float(np.quantile(independent_draws, 0.975)),
    ]
    assert bootstrap["oracle"]["ci95"][0] != pytest.approx(independent_ci[0])


def test_phase2_reference_ci_is_reported_per_task_and_operator():
    rows = []
    for init_state_id, event_instance_id, value in (
        ("init-a", "event-a-0", 1.0),
        ("init-a", "event-a-1", 1.0),
        ("init-b", "event-b-0", 0.0),
    ):
        for recovery_seed in (7, 17, 29):
            rows.append(
                {
                    "task": "task_a",
                    "init_state_id": init_state_id,
                    "event_instance_id": event_instance_id,
                    "prefix_k": 2,
                    "operator": "immediate_fresh",
                    "recovery_actor_seed": recovery_seed,
                    "safe_success": value,
                }
            )

    summary = _reference_summary(
        rows,
        ("immediate_fresh",),
        (2,),
        include_bootstrap=True,
    )
    metrics = summary["by_task"]["task_a"]
    arm = metrics["bootstrap"]["arm_means"]["immediate_fresh"]

    assert metrics["oracle"] == pytest.approx(0.5)
    assert arm["estimate"] == pytest.approx(0.5)
    assert arm["ci95"] == metrics["bootstrap"]["oracle"]["ci95"]
    assert arm["defined_cluster_count"] == 2
    assert metrics["bootstrap"]["cluster_unit"] == ["task", "init_state_id"]
    assert metrics["bootstrap"]["replicates"] == 10000
    assert metrics["bootstrap"]["seed"] == 216214
    assert summary["bootstrap_replicates"] == 10000
    assert summary["bootstrap_seed"] == 216214
