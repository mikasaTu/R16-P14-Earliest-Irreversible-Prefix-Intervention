from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from s1 import consolidate_diagnostic_phase2 as diagnostic  # noqa: E402


TASK_BOWL = "put_the_bowl_on_the_plate"
TASK_CREAM = "put_the_cream_cheese_in_the_bowl"
BUDGET = {"tail_horizon": 4, "action_budget": 8, "policy_call_cap": 8}


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _receipt(root: Path) -> tuple[dict[str, int], str]:
    value = {
        "status": "DIAGNOSTIC_SELECTED",
        "blocked": False,
        "selection_source": "calibration_only",
        "calibration_only": True,
        "confirmatory": False,
        "diagnostic_only": True,
        "evaluation_read": False,
        "evaluation_authorized": False,
        "diagnostic_continuation_doc_sha256": diagnostic.DIAGNOSTIC_DOC_SHA256,
        "protocol_sha256": diagnostic.DIAGNOSTIC_DOC_SHA256,
        "selected_budget": dict(BUDGET),
    }
    path = root / "phase1" / "diagnostic_selection_receipt.json"
    _write(path, value)
    return BUDGET, hashlib.sha256(path.read_bytes()).hexdigest()


def _structural(anchor: int, prefix: int = 8, *, error: str = "PrefixOutsideTaskHorizon") -> dict[str, object]:
    return {
        "task": TASK_BOWL,
        "event_instance_id": "event-316",
        "init_state_id": "316",
        "split": "evaluation",
        "generator_actor_seed": 7,
        "recovery_actor_seed": 17,
        "operator": "fresh_h4",
        "prefix_k": prefix,
        "tail_horizon": 4,
        "action_budget": 8,
        "policy_call_cap": 8,
        "status": "BLOCKED",
        "error_type": error,
        "anchor_global_step": anchor,
        "diagnostic_continuation": True,
        "diagnostic_selection_receipt_sha256": "a" * 64,
    }


def test_open_deny_happens_before_evaluation_sources_or_shards(tmp_path, monkeypatch):
    def deny(_root):
        raise RuntimeError("proof is absent")

    monkeypatch.setattr(diagnostic.matrix, "selected_diagnostic_budget", deny, raising=False)
    monkeypatch.setattr(diagnostic.matrix, "diagnostic_selected_budget", None, raising=False)
    malformed = tmp_path / "phase0b" / "sealed_evaluation" / TASK_BOWL / "must-not-read.json"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{", encoding="utf-8")
    shard = tmp_path / "phase2" / "shards" / TASK_BOWL / "must-not-read.json"
    shard.parent.mkdir(parents=True)
    shard.write_text("{", encoding="utf-8")

    result = diagnostic.consolidate_diagnostic_phase2(tmp_path, tmp_path / "out")

    assert result["status"] == "BLOCKED"
    assert result["evaluation_read"] is False
    assert any("OPEN_DENY" in item for item in result["blocking_reasons"])
    assert not any("must-not-read" in item for item in result["blocking_reasons"])


def test_admission_accepts_canonical_alias_and_binds_receipt_hash(tmp_path, monkeypatch):
    _budget, receipt_sha = _receipt(tmp_path)
    monkeypatch.setattr(diagnostic.matrix, "selected_diagnostic_budget", None, raising=False)
    monkeypatch.setattr(
        diagnostic.matrix,
        "diagnostic_selected_budget",
        lambda _root: {"selected_budget": dict(BUDGET), "selection_receipt_sha256": receipt_sha},
        raising=False,
    )
    reasons: list[str] = []

    selected, observed_sha, metadata = diagnostic._admit_diagnostic_budget(tmp_path, reasons)

    assert not reasons
    assert selected == BUDGET
    assert observed_sha == receipt_sha
    assert metadata["receipt_sha256"] == receipt_sha


def test_structural_exclusion_uses_source_anchor_and_preserves_marker():
    source = {
        "task": TASK_BOWL,
        "event_instance_id": "event-316",
        "init_state_id": "316",
        "split": "evaluation",
        "generator_actor_seed": 7,
        "anchor_global_step": 316,
    }
    reasons: list[str] = []
    row = _structural(316)
    result = diagnostic._structural_row(row, source, "a" * 64, reasons, "row")
    assert result is not None
    assert result["structurally_excluded"] is True
    assert result["error_type"] == "PrefixOutsideTaskHorizon"
    assert result["prefix_k"] == 8
    assert not reasons

    feasible_reasons: list[str] = []
    feasible = diagnostic._structural_row(
        _structural(300), source, "a" * 64, feasible_reasons, "feasible"
    )
    assert feasible is None
    assert any("feasible" in item for item in feasible_reasons)


def test_structural_branch_excludes_the_entire_event_from_support():
    source_key = (TASK_BOWL, "event-316", "316", "evaluation", 7)
    common = {
        "task": TASK_BOWL,
        "event_instance_id": "event-316",
        "init_state_id": "316",
        "split": "evaluation",
        "generator_actor_seed": 7,
        "recovery_actor_seed": 17,
        "operator": "fresh_h4",
        "prefix_k": 8,
        "tail_horizon": 4,
        "action_budget": 8,
        "policy_call_cap": 8,
    }
    structural = {**common, "structurally_excluded": True}
    complete_sibling = {**common, "prefix_k": 2}
    reference_sibling = {
        **common, "operator": "immediate_fresh", "prefix_k": 2, "is_reference": True,
    }
    reasons: list[str] = []
    _core, _reference, support, excluded = diagnostic._check_grid(
        [structural, complete_sibling, reference_sibling],
        {source_key: {"task": TASK_BOWL, "anchor_global_step": 316}},
        BUDGET,
        {source_key},
        reasons,
    )
    assert support == []
    assert excluded and excluded[0]["event_instance_id"] == "event-316"

    # A separate complete event keeps its reference row for analyze_crossing's
    # independent reference summary.
    other_key = (TASK_BOWL, "event-317", "317", "evaluation", 7)
    other_source = {
        **{source_key: {"task": TASK_BOWL, "anchor_global_step": 316}},
        other_key: {"task": TASK_BOWL, "anchor_global_step": 300},
    }
    other = {**reference_sibling, "event_instance_id": "event-317", "init_state_id": "317"}
    _core, _reference, support, _excluded = diagnostic._check_grid(
        [other], other_source, BUDGET, set(), [],
    )
    assert len(support) == 1 and support[0]["is_reference"] is True


def test_evaluation_source_skips_nonqualified_episode_without_event(tmp_path):
    qualified_key = (TASK_BOWL, "qualified", "316", "evaluation", 7)
    directory = tmp_path / "phase0b" / "sealed_evaluation" / TASK_BOWL
    directory.mkdir(parents=True)
    _write(
        directory / "nonqualified.json",
        {
            "task": TASK_BOWL, "init_state_id": "315",
            "event_instance_id": "nonqualified", "split": "evaluation",
            "qualified_natural_failure": False, "status": "COMPLETE", "event": None,
        },
    )
    _write(
        directory / "qualified.json",
        {
            "task": TASK_BOWL, "init_state_id": "316",
            "event_instance_id": "qualified", "split": "evaluation",
            "qualified_natural_failure": True, "status": "COMPLETE",
            "event": {
                "task": TASK_BOWL, "init_state_id": "316",
                "event_instance_id": "qualified", "split": "evaluation",
                "actor_seed": 7, "generator_actor_seed": 7,
                "anchor_global_step": 316,
            },
        },
    )
    reasons: list[str] = []
    source = diagnostic._source_events(tmp_path, {qualified_key: {}}, reasons)
    assert set(source) == {qualified_key}
    assert not any("nonqualified" in item for item in reasons)


def test_runtime_provenance_rejects_unallowed_source_commit():
    row = {
        "diagnostic_continuation": True,
        "diagnostic_selection_receipt_sha256": "a" * 64,
        "source_commit": "b" * 40,
        "runtime_receipt_sha256": "c" * 64,
        "job_id": "job", "pai_run_id": "run", "pid": 1,
        "env_hash": "env", "chunk_hash": "chunk",
    }
    reasons: list[str] = []
    diagnostic._runtime_provenance(
        row, False, "a" * 64, reasons, "row",
        allowed_source_commits={"a" * 40},
    )
    assert any("allowed diagnostic source set" in item for item in reasons)


def test_reused_calibration_path_hash_and_key_are_verified(tmp_path):
    previous = {
        "task": TASK_BOWL, "event_instance_id": "event", "init_state_id": "316",
        "split": "calibration", "generator_actor_seed": 7, "recovery_actor_seed": 17,
        "operator": "fresh_h4", "prefix_k": 2, "tail_horizon": 4,
        "action_budget": 8, "policy_call_cap": 8, "status": "COMPLETE",
        "source_commit": "a" * 40, "runtime_receipt_sha256": "c" * 64,
    }
    path = tmp_path / "phase1" / "shards" / TASK_BOWL / "old.json"
    _write(path, previous)
    row = {**previous, "reused_calibration_shard": str(path), "reused_calibration_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    reasons: list[str] = []
    diagnostic._validate_reused_calibration(
        tmp_path, row, reasons, "row",
        allowed_source_commits={"a" * 40},
        expected_runtime_receipt_sha256="c" * 64,
    )
    assert reasons == []
    path.write_text(json.dumps({**previous, "prefix_k": 4}, sort_keys=True), encoding="utf-8")
    reasons = []
    diagnostic._validate_reused_calibration(
        tmp_path, row, reasons, "row",
        allowed_source_commits={"a" * 40},
        expected_runtime_receipt_sha256="c" * 64,
    )
    assert any("SHA-256 mismatch" in item for item in reasons)


def test_unknown_error_marker_cannot_be_counted_as_structural():
    source = {
        "task": TASK_BOWL,
        "event_instance_id": "event-316",
        "init_state_id": "316",
        "split": "evaluation",
        "generator_actor_seed": 7,
        "anchor_global_step": 316,
    }
    reasons: list[str] = []
    result = diagnostic._structural_row(
        _structural(316, error="Timeout"), source, "a" * 64, reasons, "unknown"
    )
    assert result is None
    assert any("non-structural" in item for item in reasons)


def test_statistics_are_called_separately_for_primary_and_descriptive_splits(monkeypatch):
    calls: list[tuple[str, int, int]] = []

    def fake(rows, *, split, replicates, seed):
        calls.append((split, replicates, seed))
        return {
            "status": "COMPLETE",
            "crossing": {"by_task": {}},
            "split_half_null": {"draws": []},
        }

    monkeypatch.setattr(diagnostic, "analyze_crossing", fake)
    reasons: list[str] = []
    assert diagnostic._run_statistics([], "evaluation", reasons)["status"] == "COMPLETE"
    assert diagnostic._run_statistics([], "calibration", reasons)["status"] == "COMPLETE"
    assert calls == [("evaluation", 10000, 216214), ("calibration", 10000, 216214)]
    assert reasons == []


def test_low_support_k3_does_not_claim_monotone():
    raw = {
        "status": "FAIL",
        "decision": "MONOTONE",
        "by_task": {
            "put_the_cream_cheese_in_the_bowl": {
                "checks": {"both_defined_at_least_30": False}
            },
            TASK_BOWL: {"checks": {"both_defined_at_least_30": True}},
        },
    }
    result = diagnostic._diagnostic_k3({"k3": raw})
    assert result["status"] == "INSUFFICIENT_SUPPORT"
    assert result["decision"] == "NOT_ESTABLISHED_LOW_SUPPORT"
    assert result["raw_k3"] == raw
