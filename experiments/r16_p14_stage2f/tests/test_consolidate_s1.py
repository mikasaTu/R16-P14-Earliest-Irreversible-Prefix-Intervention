from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from s1.consolidate import consolidate_phase0b, consolidate_phase1, consolidate_phase2, main  # noqa: E402


TASKS = (
    "put_the_cream_cheese_in_the_bowl",
    "put_the_bowl_on_the_plate",
)


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _row(task: str, event: int = 0):
    return {
        "event_instance_id": f"event-{task}-{event}",
        "task": task,
        "init_state_id": str(event + 10),
        "split": "calibration",
        "generator_actor_seed": 7,
        "recovery_actor_seed": 7,
        "operator": "fresh_h4",
        "prefix_k": 2,
        "tail_horizon": 4,
        "action_budget": 8,
        "policy_call_cap": 8,
        "safe_success": 0.5,
        "pid": 123,
        "env_hash": "env",
        "chunk_hash": "chunk",
        "status": "COMPLETE",
        "is_reference": False,
    }


def test_phase1_partial_matrix_is_blocked_without_selection(tmp_path):
    for task in TASKS:
        _write(tmp_path / "phase1" / "shards" / task / "one.json", _row(task))
    result = consolidate_phase1(tmp_path, tmp_path)
    assert result["status"] == "BLOCKED"
    assert not (tmp_path / "phase1" / "selection_receipt.json").exists()
    assert any("missing" in reason for reason in result["blocking_reasons"])


def test_phase0b_incomplete_qualification_is_blocked(tmp_path):
    task = TASKS[0]
    _write(
        tmp_path / "phase0b" / "qualification" / task / "one.json",
        {
            "event_instance_id": "event",
            "task": task,
            "init_state_id": "10",
            "actor_seed": 7,
            "split": "calibration",
            "qualified_natural_failure": True,
            "status": "COMPLETE",
        },
    )
    result = consolidate_phase0b(tmp_path, tmp_path)
    assert result["status"] == "BLOCKED"
    assert result["evaluation_clean_read"] is False
    assert result["k1"]["source"] == "qualification metadata only"
    assert result["gate_status"] == "BLOCKED_BY_NATURAL_EVENT_YIELD"


def test_phase2_open_deny_happens_before_shard_read(tmp_path):
    _write(
        tmp_path / "phase1" / "selection_receipt.json",
        {
            "status": "SELECTED",
            "selection_source": "calibration_only",
            "selected_budget": {"tail_horizon": 4, "action_budget": 8, "policy_call_cap": 8},
        },
    )
    _write(
        tmp_path / "phase1" / "selection_authorization.json",
        {"git_commit": "not-a-real-commit", "selection_receipt_sha256": "wrong"},
    )
    # A malformed shard proves the denial path does not inspect phase2 data.
    for task in TASKS:
        shard = tmp_path / "phase2" / "shards" / task / "evaluation.json"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_text("{", encoding="utf-8")
    result = consolidate_phase2(tmp_path, tmp_path)
    assert result["status"] == "BLOCKED"
    assert result["evaluation_read"] is False
    assert any("OPEN_DENY" in reason for reason in result["blocking_reasons"])


def test_cli_requires_explicit_input_and_output_paths(tmp_path):
    code = main(["--phase", "phase1", "--input-root", str(tmp_path), "--output-root", str(tmp_path)])
    assert code == 2
    assert (tmp_path / "phase1" / "summary.json").exists()
