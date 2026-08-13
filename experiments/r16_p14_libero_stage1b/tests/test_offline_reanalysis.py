from __future__ import annotations

from r16_p14_stage1b.offline_reanalysis import branch_metrics, reanalyze_record


def _branch(*, violation: bool, success: bool, steps: int) -> dict[str, object]:
    return {
        "violation": violation,
        "violation_type": "target_cause" if violation else None,
        "task_recoverable": success,
        "safe_success": success and not violation,
        "steps": steps,
        "rework_steps": steps,
        "extra_path_length": float(steps),
        "policy_calls": 2,
    }


def test_budget_failure_is_not_cause_violation() -> None:
    metrics = branch_metrics(
        _branch(violation=False, success=False, steps=96),
        branch_budget=96,
        retained_nominal_actions=0,
        discarded_nominal_actions=10,
    )
    assert metrics["cause_violation"] is False
    assert metrics["task_failure"] is True
    assert metrics["timeout"] is True


def test_post_detection_retention_and_rework_are_corrected() -> None:
    fail = _branch(violation=True, success=False, steps=4)
    success_long = _branch(violation=False, success=True, steps=20)
    success_short = _branch(violation=False, success=True, steps=8)
    record = {
        "candidate_id": "candidate",
        "task": "put_the_bowl_on_the_plate",
        "demo_id": 1,
        "anchor_step": 10,
        "cause_type": "target_shift_or_premature_release",
        "severity": "medium",
        "insertion_prefix": 4,
        "chunk_length": 16,
        "intervention_window_policy": 8,
        "prefixes": [
            {
                "prefix_k": 4,
                "branches": {
                    "nominal_continue": fail,
                    "trim_and_replan": success_long,
                    "hold_and_replan": fail,
                    "bounded_rollback_and_replan": fail,
                    "cause_specific_local_repair": fail,
                },
            },
            {
                "prefix_k": 10,
                "branches": {
                    "nominal_continue": fail,
                    "trim_and_replan": success_short,
                    "hold_and_replan": fail,
                    "bounded_rollback_and_replan": fail,
                    "cause_specific_local_repair": success_short,
                },
            },
        ],
    }
    value = reanalyze_record(record, branch_budget=96)
    assert value["k_last_safe"] == 10
    assert value["intervention_window_after_detection"] == 6
    assert value["retention_after_detection"] == 0.5
    m1 = value["paired_methods"]["M1_continue_to_last_safe_then_full_replan"]
    assert m1["retained_nominal_actions"] == 6
    assert m1["new_non_nominal_actions_after_detection"] == 8
    assert m1["discarded_nominal_actions"] == 6
    assert m1["completion_steps_after_detection"] == 14
    timing = value["signed_gains"]["timing_gain_M1_minus_M0"]
    assert timing["new_non_nominal_action_savings"] == 12
