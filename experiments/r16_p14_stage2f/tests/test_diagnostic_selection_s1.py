
from __future__ import annotations

import hashlib
import json
import sys

import pytest

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from s1.diagnostic_selection import (  # noqa: E402
    BUDGETS,
    DEFAULT_DIAGNOSTIC_CONTINUATION_DOC_SHA256,
    TASKS,
    build_diagnostic_selection_receipt,
    write_diagnostic_selection_receipt,
)


def _metric(oracle=0.70, weakest=0.30, events=19):
    return {
        "oracle_best": oracle,
        "weakest_arm": weakest,
        "oracle_minus_weakest": oracle - weakest,
        "gap": oracle - weakest,
        "event_count": events,
        "completeness": {
            "status": "COMPLETE",
            "complete": True,
            "expected_event_prefix_cells": events * 5,
            "complete_event_prefix_cells": events * 5,
        },
    }


def _summary(
    *,
    oracle_by_budget=None,
    weakest=0.30,
    status="COMPLETE_AVAILABLE_REQUESTS_SAMPLE_SHORTFALL",
    sample_complete=False,
    structural_count=1,
):
    oracle_by_budget = oracle_by_budget or {}
    grid = []
    for tail, action, cap in BUDGETS:
        values = oracle_by_budget.get((tail, action), (0.70, 0.70))
        by_task = {
            task: _metric(oracle, weakest, 19 if task == TASKS[0] else 10)
            for task, oracle in zip(TASKS, values)
        }
        grid.append({
            "budget": {
                "tail_horizon": tail,
                "action_budget": action,
                "policy_call_cap": cap,
            },
            "by_task": by_task,
            "task_metrics": by_task,
            "selection_eligible": False,
        })
    return {
        "schema_version": 1,
        "analysis": "grid",
        "phase": "phase1",
        "status": status,
        "blocked": False,
        "split": "calibration",
        "selection_source": "calibration_only",
        "tasks": list(TASKS),
        "operators": [
            "fresh_h4",
            "fresh_h16",
            "hold_1+fresh_h4",
            "rollback_1+fresh_h16",
        ],
        "prefixes": [2, 4, 8, 12, 16],
        "grid": grid,
        "budgets": {
            f"tail_horizon={tail}/action_budget={action}/policy_call_cap={cap}": row
            for (tail, action, cap), row in zip(BUDGETS, grid)
        },
        "support_event_count_by_task": {TASKS[0]: 19, TASKS[1]: 10},
        "support_grid_row_count": 9 * (19 + 10) * 5,
        "grid_row_count": 9 * (19 + 10) * 5,
        "reference_row_count": 9 * (19 + 10) * 4,
        "sample": {
            "planned_events": 20,
            "observed_events_by_task": {TASKS[0]: 19, TASKS[1]: 11},
            "shortfall": {TASKS[0]: 1, TASKS[1]: 9},
            "planned_sample_complete": sample_complete,
            "sample_complete": sample_complete,
        },
        "planned_events": 20,
        "observed_events_by_task": {TASKS[0]: 19, TASKS[1]: 11},
        "shortfall": {TASKS[0]: 1, TASKS[1]: 9},
        "planned_sample_complete": sample_complete,
        "sample_complete": sample_complete,
        "structurally_excluded_event_count": structural_count,
        "structurally_excluded_row_count": 378 if structural_count else 0,
        "excluded_error_row_count": 0,
        "selection": {
            "status": "BLOCKED" if (not sample_complete or structural_count) else "SELECTED",
            "selected_budget": None,
        },
    }


def _sha(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def test_k2_candidate_has_priority_over_higher_ranked_numeric_failure():
    summary = _summary(
        oracle_by_budget={(4, 8): (0.90, 0.90), (8, 32): (0.80, 0.80)},
    )
    receipt = build_diagnostic_selection_receipt(
        summary, input_summary_sha256=_sha(summary), input_summary_path="/fixture/summary.json"
    )
    assert receipt["status"] == "DIAGNOSTIC_SELECTED"
    assert receipt["selected_budget"] == {
        "tail_horizon": 8, "action_budget": 32, "policy_call_cap": 8
    }
    assert receipt["k2_numeric_status"] == "PASS"
    assert receipt["fallback_used"] is False
    assert receipt["formal_selection_status"] == "BLOCKED"
    assert receipt["selection_source"] == "calibration_only"
    assert receipt["evaluation_read"] is False
    assert receipt["diagnostic_only"] is True
    assert receipt["input_sha256"] == receipt["input_summary_sha256"]
    assert receipt["protocol_sha256"] == receipt["diagnostic_continuation_doc_sha256"]
    assert receipt["original_rank"] == receipt["selected_candidate"]["original_rank"]
    assert receipt["selection_authorized"] is False
    assert receipt["gitproof_path"] is None
    assert len(receipt["all_candidates"]) == 9


def test_no_k2_candidate_uses_ranked_fallback():
    summary = _summary(
        oracle_by_budget={(4, 8): (0.90, 0.90), (8, 32): (0.65, 0.65)},
        weakest=0.60,
    )
    receipt = build_diagnostic_selection_receipt(summary, input_summary_sha256=_sha(summary))
    assert receipt["status"] == "DIAGNOSTIC_SELECTED"
    assert receipt["fallback_used"] is True
    assert receipt["k2_numeric_status"] == "FAIL"
    assert receipt["selected_budget"] == {
        "tail_horizon": 4, "action_budget": 8, "policy_call_cap": 8
    }
    assert any("K2 numerical" in item for item in receipt["prerequisite_failures"])


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda summary: summary["grid"].pop(), "exactly 9"),
        (lambda summary: summary.update({"excluded_error_row_count": 1}), "excluded_error_row_count"),
        (lambda summary: summary.update({"split": "evaluation"}), "split"),
        (lambda summary: summary.update({"selection_source": "calibration_only_diagnostic"}), "selection_source"),
    ],
)
def test_missing_or_error_or_wrong_split_blocks(mutate, needle):
    summary = _summary()
    mutate(summary)
    receipt = build_diagnostic_selection_receipt(summary, input_summary_sha256=_sha(summary))
    assert receipt["status"] == "BLOCKED"
    assert receipt["blocked"] is True
    assert receipt["selected_budget"] is None
    assert any(needle in item for item in receipt["prerequisite_failures"])


def test_short_sample_and_structural_exclusion_are_diagnostic_only(tmp_path):
    summary = _summary()
    path = tmp_path / "phase1" / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(summary), encoding="utf-8")
    receipt = write_diagnostic_selection_receipt(path)
    assert receipt["status"] == "DIAGNOSTIC_SELECTED"
    assert receipt["calibration_only"] is True
    assert receipt["confirmatory"] is False
    assert receipt["diagnostic_continuation_doc_sha256"] == DEFAULT_DIAGNOSTIC_CONTINUATION_DOC_SHA256
    saved = json.loads((path.parent / "diagnostic_selection_receipt.json").read_text())
    assert saved["input_summary_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert saved["evaluation_authorized"] is False
    assert saved["proof_written"] is False

@pytest.mark.parametrize("bad_status", [
    "BLOCKED_BY_RUNTIME_ERROR",
    "FAILED_WITH_OUTPUT",
    "RUNNING",
])
def test_runtime_error_and_nonterminal_statuses_fail_closed(bad_status):
    summary = _summary()
    summary["grid"][0]["status"] = bad_status
    receipt = build_diagnostic_selection_receipt(summary, input_summary_sha256=_sha(summary))
    assert receipt["status"] == "BLOCKED"
    assert receipt["selected_budget"] is None
    assert any("status" in item for item in receipt["prerequisite_failures"])

def test_complete_flag_cannot_mask_runtime_error_status():
    summary = _summary()
    summary["grid"][0]["by_task"][TASKS[0]]["status"] = "FAILED_WITH_OUTPUT"
    summary["grid"][0]["by_task"][TASKS[0]]["completeness"]["status"] = "BLOCKED_BY_RUNTIME_ERROR"
    receipt = build_diagnostic_selection_receipt(summary, input_summary_sha256=_sha(summary))
    assert receipt["status"] == "BLOCKED"
    assert receipt["selected_budget"] is None
    assert any("status" in item for item in receipt["prerequisite_failures"])


def test_actual_phase1_consolidation_schema_can_bind_omitted_phase_to_path(tmp_path):
    # This mirrors the committed Phase-1 consolidator output: analysis=grid,
    # split=calibration, shortfall status, and no redundant phase key.
    summary = _summary()
    summary.pop("phase")
    path = tmp_path / "artifacts" / "stage2f" / "phase1" / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(summary), encoding="utf-8")

    receipt = write_diagnostic_selection_receipt(path)

    assert receipt["status"] == "DIAGNOSTIC_SELECTED"
    assert receipt["input_summary_path"] == str(path)
    assert not any("summary.phase must be phase1" in item for item in receipt["prerequisite_failures"])


def test_omitted_phase_without_phase1_path_stays_fail_closed():
    summary = _summary()
    summary.pop("phase")
    receipt = build_diagnostic_selection_receipt(
        summary,
        input_summary_sha256=_sha(summary),
        input_summary_path="/fixture/not_phase1/summary.json",
    )

    assert receipt["status"] == "BLOCKED"
    assert any("summary.phase must be phase1" in item for item in receipt["prerequisite_failures"])
