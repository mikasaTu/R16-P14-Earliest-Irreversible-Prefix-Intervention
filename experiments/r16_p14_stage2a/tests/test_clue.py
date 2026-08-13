from __future__ import annotations

from pathlib import Path

from r16_p14_stage2a.clue import reproduce


def test_current_clue_reproduces_exactly() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "artifacts/stage1b/expert_chunk_calibration/selected_config_paired_metrics.csv"
    )
    result = reproduce(source)
    assert result["status"] == "PASS"
    assert result["facts"]["M0_safe_success_count"] == 5
    assert result["facts"]["M1_safe_success_count"] == 5
    assert result["facts"]["M2_safe_success_count"] == 2
    assert result["facts"]["M0_M1_union_safe_count"] == 6
    assert result["facts"]["M2_unique_safe_when_M0_M1_fail_count"] == 0
