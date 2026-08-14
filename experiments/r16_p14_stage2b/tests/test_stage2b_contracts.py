from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from r16_p14_stage2a.envs import state_sha256

from r16_p14_stage2b import actor_perturbation_qualification, atlas_runner, operator_audit
from r16_p14_stage2b.atlas_runner import (
    assign_severity,
    branch_contract,
    complete_with_h1_replan,
    event_budget,
    pending_event_ids,
    select_evaluation_events,
)
from r16_p14_stage2b.baselines import k_best, k_last_safe, method_positions, strongest_baseline, validate_prefix_index
from r16_p14_stage2b.bootstrap import paired_event_bootstrap
from r16_p14_stage2b.io_utils import atomic_write_csv, atomic_write_json, sha256_array, write_once_jsonl
from r16_p14_stage2b.runtime import (
    ActorHistory,
    CauseTracker,
    chunk_hash,
    is_catastrophic_object_drop,
    validate_event_contract,
)
from r16_p14_stage2b.settings import (
    ACTOR_QUALIFICATION_IDS,
    ACTION_DIM,
    ACTION_HISTORY,
    ATLAS_EVALUATION_IDS,
    CHUNK_LENGTH,
    DETECTION_PREFIX,
    OBS_HISTORY,
    PERTURBATION_CALIBRATION_IDS,
    RESERVED_IDS,
)


def fake_event() -> dict:
    states = np.arange(OBS_HISTORY * 95, dtype=np.float32).reshape(OBS_HISTORY, 95) / 1000
    actions = np.arange(ACTION_HISTORY * ACTION_DIM, dtype=np.float32).reshape(ACTION_HISTORY, ACTION_DIM) / 100
    chunk = np.arange(CHUNK_LENGTH * ACTION_DIM, dtype=np.float32).reshape(CHUNK_LENGTH, ACTION_DIM) / 10
    pre = np.arange(2 * ACTION_DIM, dtype=np.float32).reshape(2, ACTION_DIM) / 10
    init_state = np.arange(24, dtype=np.float64)
    anchor_state = np.arange(24, dtype=np.float64) + 0.5
    return {
        "event_id": "put_the_cream_cheese_in_the_bowl__seed07__init20",
        "task": "put_the_cream_cheese_in_the_bowl",
        "actor_seed": 7,
        "init_state_id": 20,
        "anchor_global_step": 50,
        "effective_horizon": 8,
        "checkpoint_sha256": "checkpoint-a",
        "state_history": states.tolist(),
        "state_history_hash": sha256_array(states, np.float32),
        "action_history": actions.tolist(),
        "action_history_hash": sha256_array(actions, np.float32),
        "original_chunk": chunk.tolist(),
        "original_chunk_hash": chunk_hash(chunk),
        "pre_anchor_actions": pre.tolist(),
        "pre_anchor_actions_hash": sha256_array(pre, np.float32),
        "init_state": init_state.tolist(),
        "init_state_hash": sha256_array(init_state, np.float64),
        "anchor_state": anchor_state.tolist(),
        "anchor_state_hash": state_sha256(anchor_state),
        "source_is_actor_generated_chunk": True,
        "source_is_demonstration_chunk": False,
    }


def fake_branches() -> list[dict]:
    return [
        {"prefix_k": 2, "S_k": 1, "safe_success": 0, "new_non_nominal_actions": 20, "policy_calls": 20, "cached_fresh_normalized_disagreement": 0.1, "error": None},
        {"prefix_k": 3, "S_k": 1, "safe_success": 1, "new_non_nominal_actions": 10, "policy_calls": 10, "cached_fresh_normalized_disagreement": 2.0, "error": None},
        {"prefix_k": 4, "S_k": 0, "safe_success": 0, "new_non_nominal_actions": 0, "policy_calls": 0, "cached_fresh_normalized_disagreement": 3.0, "error": None},
        {"prefix_k": 5, "S_k": 1, "safe_success": 1, "new_non_nominal_actions": 10, "policy_calls": 10, "cached_fresh_normalized_disagreement": 0.2, "error": None},
        {"prefix_k": 6, "S_k": 0, "safe_success": 0, "new_non_nominal_actions": 0, "policy_calls": 0, "cached_fresh_normalized_disagreement": 4.0, "error": None},
        {"prefix_k": 7, "S_k": 0, "safe_success": 0, "new_non_nominal_actions": 0, "policy_calls": 0, "cached_fresh_normalized_disagreement": 4.0, "error": None},
        {"prefix_k": 8, "S_k": 0, "safe_success": 0, "new_non_nominal_actions": 0, "policy_calls": 0, "cached_fresh_normalized_disagreement": None, "error": None},
    ]


def test_01_prefix_indexing_rejects_off_by_one() -> None:
    validate_prefix_index(2, 2, 8)
    validate_prefix_index(8, 2, 8)
    with pytest.raises(ValueError):
        validate_prefix_index(1, 2, 8)
    with pytest.raises(ValueError):
        validate_prefix_index(9, 2, 8)


def test_02_k_d_is_exact_immediate_replan() -> None:
    row = branch_contract(DETECTION_PREFIX, 8, 200)
    assert row["old_nominal_actions_retained_after_detection"] == 0
    assert row["new_action_budget"] == 200


def test_03_k_h_executes_full_validated_prefix() -> None:
    row = branch_contract(8, 8, 200)
    assert row["old_nominal_actions_retained_after_detection"] == 6
    assert row["nominal_actions_discarded"] == 0


def test_04_cause_violation_is_absorbing() -> None:
    tracker = CauseTracker("put_the_cream_cheese_in_the_bowl", 0.0, 1.0)
    tracker.mark("first")
    tracker.mark("second")
    assert tracker.violation and tracker.violation_type == "first"


def test_05_timeout_is_not_cause_violation() -> None:
    tracker = CauseTracker("put_the_bowl_on_the_stove", 0.0, 1.0)
    tracker.post_detection_steps = 200
    assert tracker.violation is False and tracker.violation_type is None


def test_06_k_last_safe_contract() -> None:
    assert k_last_safe(fake_branches()) == 5


def test_07_k_best_lexicographic_tie_is_lowest_index() -> None:
    assert k_best(fake_branches()) == 3


def test_08_branch_order_permutation_is_invariant() -> None:
    rows = fake_branches()
    assert k_last_safe(rows) == k_last_safe(list(reversed(rows)))
    assert k_best(rows) == k_best([rows[index] for index in (3, 0, 6, 2, 5, 1, 4)])


def test_09_changed_checkpoint_rejected() -> None:
    with pytest.raises(ValueError, match="checkpoint_hash"):
        validate_event_contract(fake_event(), SimpleNamespace(checkpoint_sha256="checkpoint-b", seed=7))


def test_10_changed_state_history_rejected() -> None:
    event = fake_event()
    event["state_history"][0][0] += 1
    with pytest.raises(ValueError, match="state_history_bytes"):
        validate_event_contract(event)


def test_11_changed_action_history_rejected() -> None:
    event = fake_event()
    event["action_history"][0][0] += 1
    with pytest.raises(ValueError, match="action_history_bytes"):
        validate_event_contract(event)


def test_12_changed_original_chunk_bytes_rejected() -> None:
    event = fake_event()
    event["original_chunk"][0][0] += 1
    with pytest.raises(ValueError, match="original_chunk_bytes"):
        validate_event_contract(event)


def test_13_formal_atlas_forbids_demonstration_nominal() -> None:
    event = fake_event()
    event["source_is_actor_generated_chunk"] = False
    event["source_is_demonstration_chunk"] = True
    with pytest.raises(ValueError, match="actor generated"):
        select_evaluation_events([event])


def test_14_event_splits_are_pairwise_disjoint() -> None:
    splits = [set(ACTOR_QUALIFICATION_IDS), set(PERTURBATION_CALIBRATION_IDS), set(ATLAS_EVALUATION_IDS), set(RESERVED_IDS)]
    assert all(left.isdisjoint(right) for index, left in enumerate(splits) for right in splits[index + 1 :])


def test_15_calibration_module_does_not_import_evaluation_split() -> None:
    source = inspect.getsource(actor_perturbation_qualification)
    assert "ATLAS_EVALUATION_IDS" not in source
    assert "N_oracle" not in source and "k_best(" not in source


def test_16_interrupted_atlas_only_resumes_incomplete_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(atlas_runner, "ARTIFACT_ROOT", tmp_path)
    event = fake_event()
    assert pending_event_ids([event]) == [event["event_id"]]
    path = atlas_runner.event_output_path(event, smoke=False)
    write_once_jsonl(path, [{"prefix_k": k} for k in range(2, 9)])
    assert pending_event_ids([event]) == [event["event_id"]]
    atomic_write_json(atlas_runner.event_complete_path(event, smoke=False), {"complete": True, "branch_count": 7})
    assert pending_event_ids([event]) == []


def test_17_nonempty_completed_evidence_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "evidence.jsonl"
    write_once_jsonl(path, [{"value": 1}])
    write_once_jsonl(path, [{"value": 1}])
    with pytest.raises(FileExistsError):
        write_once_jsonl(path, [{"value": 2}])


def test_18_event_budget_identical_across_prefixes() -> None:
    event = fake_event()
    budget = event_budget(event)
    contracts = [branch_contract(k, 8, budget) for k in range(2, 9)]
    assert len({row["post_detection_budget"] for row in contracts}) == 1
    assert all(row["old_nominal_actions_retained_after_detection"] + row["new_action_budget"] == budget for row in contracts)


def test_19_all_methods_derive_from_same_branch_set() -> None:
    rows = fake_branches()
    chunk = np.zeros((16, 7), dtype=np.float32)
    positions = method_positions(rows, chunk=chunk, d=2, horizon=8, disagreement_threshold=1.0)
    available = {row["prefix_k"] for row in rows}
    assert all(position is None or position in available for position in positions.values())
    assert positions["B0"] == 2 and positions["A_original"] == 5 and positions["N_oracle"] == 3


def test_20_bootstrap_rejects_branch_rows_with_duplicate_event_ids() -> None:
    rows = [{"event_id": "e1", "task": "a", "value": 1.0}, {"event_id": "e1", "task": "a", "value": 0.0}]
    with pytest.raises(ValueError, match="one paired row per independent event"):
        paired_event_bootstrap(rows, lambda row: row["value"], resamples=10, seed=1)


def test_21_bootstrap_reports_event_unit() -> None:
    rows = [{"event_id": "e1", "task": "a", "value": 1.0}, {"event_id": "e2", "task": "b", "value": 0.0}]
    result = paired_event_bootstrap(rows, lambda row: row["value"], resamples=20, seed=1)
    assert result["unit"] == "event" and result["event_count"] == 2


def test_22_event_source_contract_rejects_demo_flag() -> None:
    event = fake_event()
    event["source_is_demonstration_chunk"] = True
    with pytest.raises(ValueError, match="demonstration_nominal_forbidden"):
        validate_event_contract(event)


def test_23_severity_assignment_alternates_without_outcomes() -> None:
    severities = {"put_the_cream_cheese_in_the_bowl": (0.03, 0.06)}
    event = fake_event()
    event["init_state_id"] = 20
    assert assign_severity(event, severities) == 0.03
    event["init_state_id"] = 21
    assert assign_severity(event, severities) == 0.06


def test_24_strongest_baseline_is_deterministic() -> None:
    rows = [
        {"method": "B0", "available": True, "safe_success": 1, "cause_violation": 0, "new_non_nominal_actions": 5, "policy_calls": 5},
        {"method": "B1", "available": True, "safe_success": 1, "cause_violation": 0, "new_non_nominal_actions": 5, "policy_calls": 5},
    ]
    assert strongest_baseline(rows) == "B0"


def test_25_changed_anchor_state_bytes_rejected() -> None:
    event = fake_event()
    event["anchor_state"][0] += 0.25
    with pytest.raises(ValueError, match="anchor_state_bytes"):
        validate_event_contract(event)


def test_26_action_free_injection_refreshes_current_observation_in_place() -> None:
    history = ActorHistory(
        states=[np.zeros(95, dtype=np.float32) for _ in range(4)],
        actions=[np.zeros(7, dtype=np.float32) for _ in range(3)],
    )
    observation = {
        "robot0_proprio-state": np.ones(10, dtype=np.float32),
        "object-state": np.full(85, 2.0, dtype=np.float32),
    }
    before_actions = history.action_array().copy()
    history.refresh_latest_observation(observation)
    assert np.all(history.state_array()[:3] == 0)
    assert np.all(history.state_array()[-1, :10] == 1)
    assert np.all(history.state_array()[-1, 10:] == 2)
    assert np.array_equal(history.action_array(), before_actions)


def test_27_post_replan_cause_label_does_not_truncate_common_budget() -> None:
    observation = {
        "robot0_proprio-state": np.zeros(10, dtype=np.float32),
        "object-state": np.zeros(85, dtype=np.float32),
    }

    class FakeEnv:
        def __init__(self) -> None:
            self.sim = SimpleNamespace(data=SimpleNamespace(ncon=0), model=SimpleNamespace())

        def check_success(self) -> bool:
            return False

        def step(self, action):
            return observation, 0.0, False, {}

    class FakeTracker:
        task = "put_the_cream_cheese_in_the_bowl"
        violation = True
        violation_type = "absorbing_test_cause"

        def observe_action(self, env, action) -> None:
            return None

    class FakeBundle:
        def predict(self, states, actions, task):
            return np.zeros((16, 7), dtype=np.float32)

    context = SimpleNamespace(
        env=FakeEnv(),
        history=ActorHistory(
            states=[np.zeros(95, dtype=np.float32) for _ in range(4)],
            actions=[np.zeros(7, dtype=np.float32) for _ in range(3)],
        ),
        tracker=FakeTracker(),
    )
    result = complete_with_h1_replan(
        context,
        event={"task": "put_the_cream_cheese_in_the_bowl"},
        bundle=FakeBundle(),
        first_chunk=np.zeros((16, 7), dtype=np.float32),
        action_budget=2,
    )
    assert result["cause_violation"] is True
    assert result["new_non_nominal_actions"] == 2
    assert result["policy_calls"] == 2


def test_28_intended_stove_placement_descent_is_not_a_catastrophic_drop() -> None:
    common = {
        "task": "put_the_bowl_on_the_stove",
        "initial_object_z": 0.90,
        "current_object_z": 0.90,
        "ever_stably_lifted": True,
        "valid_release": False,
        "success": False,
    }
    assert not is_catastrophic_object_drop(target_distance_value=0.08, **common)
    assert is_catastrophic_object_drop(target_distance_value=0.20, **common)


def test_29_csv_and_qualification_errors_are_unambiguous(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    atomic_write_csv(path, ["value"], [{"value": 1}])
    assert path.read_bytes() == b"value\n1\n"

    source = inspect.getsource(actor_perturbation_qualification)
    assert source.count('"proposed_method_outcome_read": False') == 3
    assert source.count('"immediate_replan_outcome_read": False') == 3
    assert source.count('"last_safe_or_k_best_read": False') == 3


def test_30_failed_upstream_gate_blocks_operator_positive_labels() -> None:
    permitted, checks = operator_audit.positive_label_gate(
        {
            "all_upstream_gates_pass": False,
            "minimum_valid_data": {"passed": True},
            "invalid_events": {},
        }
    )
    assert not permitted
    assert checks["minimum_valid_data_pass"]
