from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from s1 import matrix, selection  # noqa: E402
from experiments.r16_p14_stage2f.pai import seal_selection  # noqa: E402


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


def _diagnostic_receipt(protocol_sha: str, input_sha: str = "1" * 64) -> dict:
    budget = {"tail_horizon": 4, "action_budget": 8, "policy_call_cap": 8}
    return {
        "schema_version": 1,
        "receipt_type": "DIAGNOSTIC_SELECTION_RECEIPT",
        "status": "DIAGNOSTIC_SELECTED",
        "selection_source": "calibration_only",
        "confirmatory": False,
        "prerequisite_failures": ["planned calibration sample incomplete"],
        "input_sha256": input_sha,
        "input_summary_sha256": input_sha,
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



def _prepare_diagnostic_root(root: Path, monkeypatch):
    summary_path = root / "phase1" / "summary.json"
    _write(summary_path, {"phase": "phase1", "fixture": "calibration"})
    summary_sha = matrix.file_sha(summary_path)
    data = _diagnostic_receipt(matrix.file_sha(matrix._PROTOCOL), summary_sha)
    receipt = root / "phase1" / "diagnostic_selection_receipt.json"
    auth = root / "phase1" / "diagnostic_selection_authorization.json"
    _write(receipt, data)
    _write(auth, _auth(receipt.read_bytes()))
    recompute_calls = []

    def recompute(path, digest, protocol):
        recompute_calls.append((Path(path), digest, protocol))
        return json.loads(json.dumps(data))

    monkeypatch.setattr(selection, "verify_commit_proof", lambda *args, **kwargs: True)
    monkeypatch.setattr(matrix, "_recompute_diagnostic_receipt", recompute)
    return summary_path, summary_sha, data, receipt, recompute_calls

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
    _, summary_sha, data, _, calls = _prepare_diagnostic_root(root, monkeypatch)
    assert matrix.diagnostic_selected_budget(root) == data["selected_budget"]
    assert calls == [
        (
            root / "phase1" / "summary.json",
            summary_sha,
            matrix.file_sha(matrix._PROTOCOL),
        )
    ]


def test_diagnostic_input_sha_mismatch_denies_before_recompute(tmp_path, monkeypatch):
    root = tmp_path
    summary_path = root / "phase1" / "summary.json"
    _write(summary_path, {"phase": "phase1", "fixture": "changed"})
    receipt = root / "phase1" / "diagnostic_selection_receipt.json"
    auth = root / "phase1" / "diagnostic_selection_authorization.json"
    data = _diagnostic_receipt(matrix.file_sha(matrix._PROTOCOL), "1" * 64)
    _write(receipt, data)
    _write(auth, _auth(receipt.read_bytes()))
    monkeypatch.setattr(selection, "verify_commit_proof", lambda *args, **kwargs: True)
    with pytest.raises(RuntimeError, match="input summary SHA256 mismatch"):
        matrix.diagnostic_selected_budget(root)



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
    _, _, data, receipt, _ = _prepare_diagnostic_root(root, monkeypatch)
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


def _tree_object(entries):
    data = bytearray()
    for mode, name, object_id in entries:
        data.extend(mode.encode())
        data.extend(b" ")
        data.extend(name.encode())
        data.extend(b"\0")
        data.extend(bytes.fromhex(object_id))
    return bytes(data)


def test_diagnostic_seal_tree_proof_binds_all_diagnostic_components(tmp_path):
    raw = b'{"status":"DIAGNOSTIC_SELECTED"}\n'
    objects = {}
    blob = selection.oid("blob", raw)
    objects[blob] = None
    phase_tree = _tree_object(
        [("100644", "diagnostic_selection_receipt.json", blob)]
    )
    phase_id = selection.oid("tree", phase_tree)
    objects[phase_id] = phase_tree
    stage2f_tree = _tree_object([("040000", "phase1", phase_id)])
    stage2f_id = selection.oid("tree", stage2f_tree)
    objects[stage2f_id] = stage2f_tree
    artifacts_tree = _tree_object([("040000", "stage2f", stage2f_id)])
    artifacts_id = selection.oid("tree", artifacts_tree)
    objects[artifacts_id] = artifacts_tree
    root_tree = _tree_object([("040000", "artifacts", artifacts_id)])
    root_id = selection.oid("tree", root_tree)
    objects[root_id] = root_tree
    commit_object = (
        f"tree {root_id}\nauthor test <test@example.com> 0 +0000\n"
        "committer test <test@example.com> 0 +0000\n\nfixture\n"
    ).encode()
    commit = selection.oid("commit", commit_object)
    proof = seal_selection._tree_path_proof(
        commit_object,
        seal_selection.DIAGNOSTIC_REL,
        lambda tree: objects[tree],
    )
    assert len(proof) == 4
    assert b"diagnostic_selection_receipt.json\0" in base64.b64decode(
        proof[-1]["object_base64"]
    )
    receipt = tmp_path / "phase1" / "diagnostic_selection_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(raw)
    authorization = {
        "selection_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "git_commit": commit,
        "commit_object_base64": base64.b64encode(commit_object).decode(),
        "tree_path_proof": proof,
        "selection_path": selection.DIAGNOSTIC_PATH,
        "diagnostic_atlas": True,
        "verified_origin_main": "b" * 40,
    }
    assert selection.verify_commit_proof(receipt, authorization, diagnostic=True)
