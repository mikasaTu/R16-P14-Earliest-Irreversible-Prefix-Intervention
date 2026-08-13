from __future__ import annotations

from r16_p14_stage1b.expert_gate import paired_row


def test_paired_gain_signs_are_method_minus_baseline() -> None:
    record = {
        "task": "task",
        "config_id": "config",
        "demo_id": 1,
        "candidate_id": "candidate",
        "insertion_prefix": 2,
        "k_last_safe": 7,
        "intervention_window": 5,
        "post_detection_retention": 5 / 14,
        "M0_immediate_full_replan": {
            "cause_violation": True,
            "safe_success": False,
            "new_non_nominal_actions": 30,
        },
        "M1_continue_then_full_replan": {
            "cause_violation": False,
            "safe_success": True,
            "new_non_nominal_actions": 20,
        },
        "M2_continue_then_local_repair": {
            "cause_violation": False,
            "safe_success": True,
            "new_non_nominal_actions": 8,
        },
    }
    row = paired_row(record)
    assert row["timing_gain_cause_violation_M1_minus_M0"] == -1
    assert row["timing_gain_safe_success_M1_minus_M0"] == 1
    assert row["timing_gain_new_actions_M1_minus_M0"] == -10
    assert row["operator_gain_new_actions_M2_minus_M1"] == -12
