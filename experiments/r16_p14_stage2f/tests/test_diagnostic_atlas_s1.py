from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from s1 import matrix, selection  # noqa: E402


TASK = "put_the_cream_cheese_in_the_bowl"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _candidate(budget):
    return {
        "budget": dict(budget),
        "rank": 1,
        "original_rank": 1,
        "k2_qualifies": False,
        "qualifies": False,
        "min_oracle": 0.2,
        "min_gap": 0.1,
    }


def _diagnostic_receipt(protocol_sha: str) -> dict:
    budget = {"tail_horizon": 4, "action_budget": 8, "policy_call_cap": 8}
    return {
        "schema_version": 1,
        "receipt_type": "DIAGNOSTIC_SELECTION_RECEIPT",
        "status": "DIAGNOSTIC_SELECTED",
        "selection_source": "calibration_only",
        "confirmatory": False,
        "prerequisite_failures": ["planned calibration sample incomplete"],
        "input_sha256": "1" * 64,
        "input_summary_sha256": "1" * 64,
        "protocol_sha256": protocol_sha,
        "diagnostic_continuation_doc_sha256": protocol_sha,
        "selected_budget": budget,
        "original_rank": 1,
        "all_candidates": [_candidate(budget)],
        "selected_candidate": _candidate(budget),
    }


def _auth(raw: bytes) -> dict:
    return {
        "selection_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "git_commit": "a" * 40,
        "commit_object_base64": "ZmFrZQ==",
        "tree_path_proof": [],
        "selection_path": selection.DIAGNOSTIC_PATH,
        "diagnostic_atlas": True,
        "verified_origin_main": "b" * 40,
    }


def test_formal_admission_still_rejects_short_sample(tmp_path, monkeypatch):
    root = tmp_path
    receipt = root / "phase1" / "selection_receipt.json"
    auth = root / "phase1" / "selection_authorization.json"
    _write(
        receipt,
        {
            "status": "SELECTED",
            "selection_source": "calibration_only",
            "planned_sample_complete": False,
            "selected_budget": {
                "tail_horizon": 4,
                "action_budget": 8,
                "policy_call_cap": 8,
            },
            "ranked_candidates": [],
        },
    )
    _write(auth, {"selection_receipt_sha256": "unused"})
    monkeypatch.setattr(selection, "verify_commit_proof", lambda *args, **kwargs: True)
    with pytest.raises(RuntimeError, match="sample incomplete"):
        matrix.selected_budget(root)


def test_diagnostic_missing_or_fake_proof_is_open_deny(tmp_path):
    root = tmp_path
    with pytest.raises(RuntimeError, match="diagnostic selection not committed"):
        matrix.diagnostic_selected_budget(root)
    receipt = root / "phase1" / "diagnostic_selection_receipt.json"
    auth = root / "phase1" / "diagnostic_selection_authorization.json"
    _write(receipt, _diagnostic_receipt(matrix.file_sha(matrix._PROTOCOL)))
    _write(auth, _auth(receipt.read_bytes()))
    with pytest.raises(RuntimeError):
        matrix.diagnostic_selected_budget(root)


def test_diagnostic_proof_rejects_formal_or_arbitrary_receipt_path(tmp_path):
    receipt = tmp_path / "phase1" / "selection_receipt.json"
    _write(receipt, {})
    with pytest.raises(RuntimeError, match="path"):
        selection.verify_commit_proof(receipt, _auth(receipt.read_bytes()), diagnostic=True)
    arbitrary = tmp_path / "other" / "diagnostic_selection_receipt.json"
    _write(arbitrary, {})
    with pytest.raises(RuntimeError, match="phase1"):
        selection.verify_commit_proof(arbitrary, _auth(arbitrary.read_bytes()), diagnostic=True)


def test_diagnostic_missing_fields_fail_closed():
    with pytest.raises(RuntimeError, match="fields missing"):
        selection.validate_diagnostic_receipt(
            {"status": "DIAGNOSTIC_SELECTED"},
            protocol_sha256="2" * 64,
        )


def test_valid_diagnostic_selection_is_admitted_without_evaluation_read(
    tmp_path, monkeypatch
):
    root = tmp_path
    receipt = root / "phase1" / "diagnostic_selection_receipt.json"
    auth = root / "phase1" / "diagnostic_selection_authorization.json"
    data = _diagnostic_receipt(matrix.file_sha(matrix._PROTOCOL))
    _write(receipt, data)
    _write(auth, _auth(receipt.read_bytes()))
    monkeypatch.setattr(selection, "verify_commit_proof", lambda *args, **kwargs: True)
    assert matrix.diagnostic_selected_budget(root) == data["selected_budget"]


def test_reused_calibration_metadata_keeps_source_and_trace():
    binding = {
        "receipt_sha256": "a" * 64,
        "receipt_path": "phase1/diagnostic_selection_receipt.json",
        "input_sha256": "b" * 64,
        "protocol_sha256": "c" * 64,
        "original_rank": 1,
    }
    row = {"source_commit": "source", "trace_path": "/trace/original.jsonl.gz"}
    matrix._attach_diagnostic_binding(row, binding)
    assert row["source_commit"] == "source"
    assert row["trace_path"] == "/trace/original.jsonl.gz"
    assert row["diagnostic_continuation"] is True
    assert row["diagnostic_selection_receipt_sha256"] == "a" * 64


def test_diagnostic_completion_is_bound_before_empty_atlas(tmp_path, monkeypatch):
    root = tmp_path
    receipt = root / "phase1" / "diagnostic_selection_receipt.json"
    auth = root / "phase1" / "diagnostic_selection_authorization.json"
    data = _diagnostic_receipt(matrix.file_sha(matrix._PROTOCOL))
    _write(receipt, data)
    _write(auth, _auth(receipt.read_bytes()))
    monkeypatch.setattr(selection, "verify_commit_proof", lambda *args, **kwargs: True)
    result = matrix.run_task(
        "atlas",
        TASK,
        root,
        workers=1,
        device="cpu",
        diagnostic_atlas=True,
    )
    assert result["diagnostic_continuation"] is True
    assert result["diagnostic_selection_receipt_sha256"] == matrix.file_sha(receipt)
    completion = json.loads(
        (root / "phase2" / f"completion_{TASK}.json").read_text()
    )
    assert completion["diagnostic_continuation"] is True
    assert completion["diagnostic_selection_receipt_sha256"] == matrix.file_sha(receipt)
