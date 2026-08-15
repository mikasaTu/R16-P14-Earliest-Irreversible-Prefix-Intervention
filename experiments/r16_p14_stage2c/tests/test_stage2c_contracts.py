from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import r16_p14_stage2c.runtime as stage2c_runtime
from r16_p14_stage2b.io_utils import write_once_jsonl
from r16_p14_stage2c import goal_geometry
from r16_p14_stage2c.checksums import build_manifest
from r16_p14_stage2c.events import exclusion_metrics, persist_first_completed_event_marker
from r16_p14_stage2c.mechanism_reverse import paired_prefix_rows, target_shift_activation
from r16_p14_stage2c.runtime import numeric_difference
from r16_p14_stage2c.contracts import (
    admit_event_replay,
    assert_no_physical_irreversibility_label,
    freeze_strongest_baseline,
    hierarchical_cluster_bootstrap,
    last_recoverable_prefix,
    matched_branch_contract,
    monotone_observed_safety,
    pending_shard_ids,
    persistent_irreversibility_prefix,
    positive_label_allowed,
    qualify_cell,
    recovery_operator_contract,
    replay_cell_summary,
    select_crossfit_prefix,
    select_second_failure_family,
)


def replay_attempt(order: int, **overrides):
    row = {
        "order_slot": order,
        "error": None,
        "replay_valid": True,
        "anchor_state_exact": True,
        "state_history_exact": True,
        "action_history_exact": True,
        "original_chunk_exact": True,
        "branch_order_invariant": True,
    }
    row.update(overrides)
    return row


def safety_rows(values):
    return [{"prefix_k": k, "S_obs": value} for k, value in zip(range(2, 17), values)]


def qualification_rows(count=20, seeds=(7, 17), delayed=True):
    rows = []
    for i in range(count):
        rows.append({
            "event_id": f"e{i}",
            "actor_seed": seeds[i % len(seeds)],
            "replay_valid": True,
            "error": None,
            "immediate_cause_violation": False,
            "delayed_cause_violation": delayed if i < 10 else False,
            "first_violation_offset": 5 if delayed and i < 10 else None,
        })
    return rows


def test_01_replay_errors_count_in_denominator():
    summary = replay_cell_summary([
        {"replay_valid": True, "error": None},
        {"replay_valid": False, "error": "boom"},
    ])
    assert summary["replay_rate"] == 0.5 and summary["error_count"] == 1


def test_02_any_replay_error_blocks_formal_cell():
    rows = [replay_attempt(0), replay_attempt(1), replay_attempt(2, error="boom")]
    assert not admit_event_replay(rows)["admitted"]


def test_03_s_obs_cannot_transition_zero_to_one():
    values = [1] * 4 + [0] * 4 + [1] + [0] * 6
    result = monotone_observed_safety(safety_rows(values))
    assert not result["passed"] and result["failure_label"] == "NONMONOTONIC_CAUSE_PREFIX"


def test_04_nonmonotonic_event_has_no_last_safe():
    result = monotone_observed_safety(safety_rows([1, 0, 1] + [0] * 12))
    assert result["k_last_observed_safe"] is None


def test_05_goal_distance_source_does_not_load_demo0():
    source = inspect.getsource(goal_geometry)
    assert "load_demo" not in source and "joint_trajectory" not in source
    assert "parsed_problem" in source and "object_sites_dict" in source


def test_06_replay_admission_reads_no_outcomes():
    result = admit_event_replay([replay_attempt(i, safe_success=bool(i)) for i in range(3)])
    assert result["admitted"] and not result["outcome_fields_read"]


def test_07_used_event_requires_three_of_three_exact():
    assert admit_event_replay([replay_attempt(i) for i in range(3)])["admitted"]
    assert not admit_event_replay([replay_attempt(i) for i in range(2)])["admitted"]


def test_08_cached_fresh_equal_prefix_lengths():
    for k in range(2, 17):
        pair = matched_branch_contract(k, 100)
        assert pair["CACHED_MATCHED"]["prefix_actions"] == pair["FRESH_MATCHED"]["prefix_actions"]


def test_09_cached_fresh_equal_pre_k_actor_calls():
    pair = matched_branch_contract(9, 100)
    assert pair["CACHED_MATCHED"]["pre_k_actor_calls"] == pair["FRESH_MATCHED"]["pre_k_actor_calls"] == 1


def test_10_both_primary_branches_use_h16_tail():
    pair = matched_branch_contract(8, 100)
    assert {row["tail_execution_horizon"] for row in pair.values()} == {16}


def test_11_old_and_fresh_actions_consume_same_budget():
    pair = matched_branch_contract(16, 97)
    for row in pair.values():
        assert row["prefix_actions"] + row["tail_action_budget"] == 97


def test_12_cached_matched_sham_chunk_is_not_executed():
    assert matched_branch_contract(12, 100)["CACHED_MATCHED"]["d_call_executed"] is False


def test_13_demonstration_chunks_are_forbidden_by_preregistration():
    prereg = (Path(__file__).parents[1] / "preregistration.yaml").read_text()
    assert "frozen_ACT" in prereg and "retraining: forbidden" in prereg


def test_14_last_observed_safe_can_differ_from_last_recoverable():
    safe = monotone_observed_safety(safety_rows([1] * 6 + [0] * 9))["k_last_observed_safe"]
    atlas = [{"prefix_k": k, "R_U": k <= 10} for k in range(2, 17)]
    assert safe == 7 and last_recoverable_prefix(atlas) == 10


def test_15_k_irrev_uses_persistent_crossing():
    atlas = [{"prefix_k": k, "R_U": k in {2, 3, 5}} for k in range(2, 17)]
    assert persistent_irreversibility_prefix(atlas) == 6


def test_16_no_physical_irreversibility_label():
    assert_no_physical_irreversibility_label({"label": "policy-family-relative irreversibility"})
    with pytest.raises(ValueError):
        assert_no_physical_irreversibility_label({"label": "physical irreversibility"})


def test_17_held_out_actor_cannot_enter_selection():
    rows = [{"prefix_k": 2, "recovery_actor_seed": 29, "safe_success": 1, "new_non_nominal_actions": 2, "actor_calls": 1}]
    with pytest.raises(ValueError, match="held-out"):
        select_crossfit_prefix(rows, 29)


def test_18_strongest_baseline_uses_calibration_only():
    with pytest.raises(ValueError, match="calibration"):
        freeze_strongest_baseline([{"split": "evaluation", "method": "x", "safe_success": 1, "new_non_nominal_actions": 1, "actor_calls": 1}])


def test_19_bootstrap_clusters_task_init_state():
    rows = [
        {"task": "a", "init_state_id": 40, "actor_seed": seed, "delta": 1.0}
        for seed in (7, 17, 29)
    ] + [{"task": "a", "init_state_id": 41, "actor_seed": 7, "delta": -1.0}]
    result = hierarchical_cluster_bootstrap(rows, lambda row: row["delta"], replicates=100, seed=1)
    assert result["cluster_count"] == 2 and result["row_count"] == 4


def test_20_actor_seed_rows_are_not_independent_units():
    rows = [{"task": "a", "init_state_id": 40, "delta": value} for value in (0, 1, 1)]
    result = hierarchical_cluster_bootstrap(rows, lambda row: row["delta"], replicates=20, seed=2)
    assert result["cluster_count"] == 1


def test_21_minimum_data_gate_fails_closed():
    summary = qualify_cell(qualification_rows(count=19))
    assert not summary["qualifies"] and not summary["checks"]["valid_events_ge_20"]


def test_22_second_family_selection_forbids_method_gain():
    with pytest.raises(ValueError, match="method outcomes"):
        select_second_failure_family([{"task": "stove", "qualifies": True, "method_gain": 0.1}], ["stove"])


def test_23_completed_evidence_cannot_be_overwritten(tmp_path):
    path = tmp_path / "complete.jsonl"
    write_once_jsonl(path, [{"value": 1}])
    with pytest.raises(FileExistsError):
        write_once_jsonl(path, [{"value": 2}])


def test_24_resume_only_incomplete_event_shards():
    assert pending_shard_ids(["a", "b", "c"], ["a", "c"]) == ["b"]


def test_25_no_positive_label_after_upstream_failure():
    assert not positive_label_allowed({"contract": True, "replay": False, "second_family": True})


def test_26_recovery_operators_have_identical_budgets():
    contracts = recovery_operator_contract(103)
    assert {row["action_budget"] for row in contracts.values()} == {103}
    assert {row["policy_call_budget"] for row in contracts.values()} == {26}


def test_27_crossfit_lexicographic_tie_uses_lower_k():
    rows = [
        {"prefix_k": k, "recovery_actor_seed": seed, "safe_success": 1, "new_non_nominal_actions": 3, "actor_calls": 2}
        for k in (4, 7) for seed in (7, 17)
    ]
    assert select_crossfit_prefix(rows, 29) == 4


def test_28_second_family_uses_frozen_order_and_two_severities():
    rows = [
        {"task": "second", "severity": i, "qualifies": True} for i in (1, 2)
    ] + [{"task": "first", "severity": 1, "qualifies": True}]
    assert select_second_failure_family(rows, ["first", "second"]) == "second"


def test_29_complete_monotone_grid_reports_last_safe():
    result = monotone_observed_safety(safety_rows([1] * 10 + [0] * 5))
    assert result["passed"] and result["k_last_observed_safe"] == 11


def test_30_first_work_marker_rejects_exception_attempt(tmp_path):
    marker = tmp_path / "first_completed_event.json"
    failed = {"event": None, "attempt": {"eligible": False, "reason": "exception", "error": "EGL failure"}}
    assert not persist_first_completed_event_marker(failed, "task", 7, 30, "calibration", marker=marker)
    assert not marker.exists()
    passed = {"event": None, "attempt": {"eligible": False, "reason": "no_actor_generated_anchor", "error": None}}
    assert persist_first_completed_event_marker(passed, "task", 7, 30, "calibration", marker=marker)
    payload = json.loads(marker.read_text())
    assert payload["schema_version"] == 2
    assert payload["record_type"] == "error_free_completed_actor_event_attempt"
    assert payload["attempt_error"] is None


def test_31_checksum_manifest_excludes_itself(tmp_path):
    (tmp_path / "result.json").write_text("{}\n")
    (tmp_path / "SHA256SUMS").write_text("stale\n")
    manifest = build_manifest(tmp_path)
    assert "result.json" in manifest
    assert "SHA256SUMS" not in manifest


def test_32_ineligible_event_is_not_replay_instability():
    attempts = [
        {"admission": {"admitted": True, "failure_label": None}},
        {"admission": {"admitted": False, "failure_label": "NO_ACTOR_EVENT"}},
        {"admission": {"admitted": False, "failure_label": "REPLAY_ERROR"}},
    ]
    metrics = exclusion_metrics(attempts, admitted_count=1)
    assert metrics["ineligible_event_exclusions"] == 1
    assert metrics["unstable_replay_exclusions"] == 1
    assert metrics["unstable_event_exclusion_rate"] == 0.5
    assert metrics["total_event_exclusion_rate"] == pytest.approx(2 / 3)


def test_33_replay_difference_is_exact_and_locates_first_delta():
    saved = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    exact = numeric_difference(saved, saved.copy())
    assert exact["byte_exact"] and exact["nonzero_count"] == 0
    fresh = saved.copy()
    fresh[0, 1] += np.float32(0.25)
    changed = numeric_difference(saved, fresh)
    assert not changed["byte_exact"]
    assert changed["nonzero_count"] == 1
    assert changed["max_abs_difference"] == pytest.approx(0.25)
    assert changed["first_difference"]["index"] == [0, 1]


def test_34_trace_capture_flag_does_not_change_observable_refresh_schedule(monkeypatch):
    observation = {
        "robot0_proprio-state": np.zeros(39, dtype=np.float32),
        "object-state": np.zeros(56, dtype=np.float32),
    }

    class FakeEnv:
        def seed(self, _seed):
            return None

        def reset(self):
            return observation

        def step(self, _action):
            return observation, 0.0, False, {}

        def get_sim_state(self):
            return np.asarray([1.0], dtype=np.float64)

        def close(self):
            return None

    class FakeBundle:
        def predict(self, _states, _actions, _task):
            return np.zeros((16, 7), dtype=np.float32)

    refreshes = []

    def trace_row(_env, _history, *, global_step, action):
        refreshes.append((global_step, action is None))
        return {"global_step": global_step}

    states = np.zeros((4, 95), dtype=np.float32)
    actions = np.zeros((3, 7), dtype=np.float32)
    chunk = np.zeros((16, 7), dtype=np.float32)
    event = {
        "task": "put_the_cream_cheese_in_the_bowl",
        "init_state": [0.0],
        "pre_anchor_actions": np.zeros((2, 7), dtype=np.float32).tolist(),
        "anchor_state": [1.0],
        "anchor_state_hash": stage2c_runtime.state_sha256(np.asarray([1.0], dtype=np.float64)),
        "state_history_hash": stage2c_runtime.sha256_array(states, np.float32),
        "action_history_hash": stage2c_runtime.sha256_array(actions, np.float32),
        "original_chunk_hash": stage2c_runtime.chunk_hash(chunk),
    }
    monkeypatch.setattr(stage2c_runtime, "validate_event_bytes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(stage2c_runtime, "make_env", lambda *_args, **_kwargs: (FakeEnv(), None))
    monkeypatch.setattr(stage2c_runtime, "restore_state", lambda *_args, **_kwargs: observation)
    monkeypatch.setattr(stage2c_runtime, "runtime_trace_row", trace_row)
    env, _, audit = stage2c_runtime.reconstruct_anchor(event, FakeBundle(), capture_trace=False)
    env.close()
    assert audit["passed"]
    assert audit["trace"] == []
    assert refreshes == [(0, True), (1, False), (2, False)]


def test_35_mechanism_reverse_pairs_only_matched_cached_and_fresh():
    common = {
        "event_instance_id": "event",
        "task": "put_the_cream_cheese_in_the_bowl",
        "split": "evaluation",
        "parameter_id": "shift_040mm",
        "generator_actor_seed": 7,
        "init_state_id": 40,
        "prefix_k": 6,
        "task_success": True,
        "cause_violation": False,
        "task_progress_retained_at_k": 0.1,
        "completion_steps": 100,
        "eef_path_length": 1.0,
        "object_path_length": 0.5,
        "contact_count": 2,
        "cached_fresh_action_displacement": 0.25,
        "allocated_actor_call_budget": 8,
        "total_post_detection_action_budget": 100,
    }
    cached = {**common, "branch": "CACHED_MATCHED", "safe_success": True, "new_non_nominal_actions": 80}
    fresh = {**common, "branch": "FRESH_MATCHED", "safe_success": False, "new_non_nominal_actions": 84}
    pairs = paired_prefix_rows([cached, fresh])
    assert len(pairs) == 1
    assert pairs[0]["safe_success_delta"] == 1
    assert pairs[0]["fresh_minus_cached_new_actions"] == 4
    assert pairs[0]["compute_matched"] and pairs[0]["budget_matched"]


def test_36_target_shift_activation_requires_release_after_lift():
    base = {
        "split": "calibration",
        "task": "put_the_cream_cheese_in_the_bowl",
        "event_id": "event",
        "actor_seed": 7,
        "init_state_id": 30,
        "action_history": np.zeros((3, 7), dtype=np.float32).tolist(),
        "original_chunk": np.zeros((16, 7), dtype=np.float32).tolist(),
        "initial_manipulated_qpos": np.zeros(7, dtype=np.float64).tolist(),
        "nominal_trace": {"positions": np.zeros((17, 7), dtype=np.float64).tolist()},
        "phase": {"ever_lifted": False},
    }
    base["action_history"][-1][-1] = 1.0
    for index in range(4):
        base["original_chunk"][index][-1] = 1.0
    base["original_chunk"][4][-1] = -1.0
    audit = target_shift_activation([base])
    assert audit["events_with_post_detection_release"] == 1
    assert audit["events_with_cause_eligible_release"] == 0
