from __future__ import annotations

import json
from pathlib import Path

from r16_p14_stage2a.atlas_preflight import BRANCH_PREFIXES
from r16_p14_stage2a.settings import TASK_NAMES, TASK_SPECS, perturbation_grid


def test_roster_and_grid_are_frozen() -> None:
    assert len(TASK_NAMES) == 6
    assert set(TASK_NAMES) == set(TASK_SPECS)
    assert all(len(perturbation_grid(task)) in {12, 16} for task in TASK_NAMES)


def test_stage1b_kill_is_immutable_in_stage2_decision() -> None:
    decision_path = Path(__file__).resolve().parents[1] / "decision.json"
    decision = json.loads(decision_path.read_text())
    assert decision["stage1b_universal_hypothesis"] == "KILLED_IMMUTABLE"
    assert decision["accepted"] is False
    assert decision["full_atlas_launched"] is False


def test_atlas_branch_prefixes_are_stride_one_and_stop_before_chunk_end() -> None:
    assert BRANCH_PREFIXES == tuple(range(2, 16))
    assert len(BRANCH_PREFIXES) == 14
