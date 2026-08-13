from __future__ import annotations

from r16_p14_stage1b.expert_chunk_calibration import (
    is_delayed_qualifying_violation,
    summarize_config,
)
from r16_p14_stage1b.expert_common import config_id, drawer_grid, target_shift_grid


def test_fixed_grids_and_stable_ids() -> None:
    targets = target_shift_grid()
    drawers = drawer_grid()
    assert len(targets) == 16
    assert len(drawers) == 12
    assert (
        config_id("put_the_bowl_on_the_plate", targets[-1])
        == "lead14_shift100mm"
    )
    assert (
        config_id("open_the_middle_drawer_of_the_cabinet", drawers[0])
        == "offset-02_clear035mm"
    )


def test_delayed_violation_contract_is_task_specific() -> None:
    target = {
        "task": "put_the_bowl_on_the_plate",
        "cause_violation": True,
        "immediate_violation": False,
        "first_violation_post_detection_step": 12,
    }
    drawer = {
        "task": "open_the_middle_drawer_of_the_cabinet",
        "cause_violation": True,
        "immediate_violation": False,
        "first_violation_post_detection_step": 12,
    }
    assert is_delayed_qualifying_violation(target)
    assert not is_delayed_qualifying_violation(drawer)
    drawer["first_violation_post_detection_step"] = 5
    assert is_delayed_qualifying_violation(drawer)


def test_config_requires_actual_recovery_window() -> None:
    parameters = {
        "release_lead_actions": 12,
        "target_shift_x_magnitude_meters": 0.06,
    }
    nominal = []
    for demo_id in range(10):
        nominal.append(
            {
                "task": "put_the_bowl_on_the_plate",
                "demo_id": demo_id,
                "phase_contract_valid": True,
                "cause_violation": demo_id < 5,
                "immediate_violation": False,
                "first_violation_post_detection_step": 12 if demo_id < 5 else None,
            }
        )
    audits = [
        {
            "intervention_window": 4,
            "post_detection_retention": 4 / 14,
            "replay_pass_count": 15,
            "replay_branch_point_count": 15,
        }
        for _ in range(5)
    ]
    summary = summarize_config(
        "put_the_bowl_on_the_plate", parameters, nominal, audits, 10
    )
    assert summary["qualifies"]
    audits[0]["intervention_window"] = 0
    for audit in audits[1:]:
        audit["intervention_window"] = 1
    summary = summarize_config(
        "put_the_bowl_on_the_plate", parameters, nominal, audits, 10
    )
    assert not summary["qualifies"]
