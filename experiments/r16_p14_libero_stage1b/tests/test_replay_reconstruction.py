from __future__ import annotations

from copy import deepcopy

from r16_p14_stage1b.replay_reconstruction import compare_runs


def _run() -> dict[str, object]:
    branch = {
        "full_simulator_state_hash": "a",
        "policy_observation_feature_hash": "b",
        "object_joint_poses_hash": "c",
        "robot_state_hash": "d",
        "contact_pairs_hash": "e",
        "perturbation_state_hash": "f",
        "controller_visible_observation_hash": "g",
    }
    return {
        "prefix_contact_sequence_hash": "prefix",
        "branch_point": branch,
        "suffix": {
            "contact_sequence_hash": "suffix",
            "task_success": False,
            "cause_violation": True,
            "violation_type": "target",
            "final_simulator_state": [1.0, 2.0],
        },
    }


def test_exact_runs_pass() -> None:
    result = compare_runs([_run(), _run()], 1e-9)
    assert result["passed"] is True
    assert result["max_final_state_abs_error"] == 0.0


def test_hidden_final_divergence_fails() -> None:
    first = _run()
    second = deepcopy(first)
    second["suffix"]["final_simulator_state"][1] = 2.1
    result = compare_runs([first, second], 1e-9)
    assert result["passed"] is False
    assert result["max_final_state_abs_error"] > 0.09
