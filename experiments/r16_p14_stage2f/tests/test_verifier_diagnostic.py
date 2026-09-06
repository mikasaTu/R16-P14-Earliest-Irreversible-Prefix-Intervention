from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "verify_r16p14_stage2f.py"


def _verifier():
    spec = importlib.util.spec_from_file_location("stage2f_verifier_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Selection:
    def __init__(self, result=True):
        self.result = result
        self.calls = []

    def verify_commit_proof(self, receipt, authorization, diagnostic=False):
        self.calls.append(
            (Path(receipt), dict(authorization), diagnostic)
        )
        return self.result


class _Matrix:
    def __init__(self, budget):
        self.budget = dict(budget)
        self.paths = []

    def selected_diagnostic_budget(self, root):
        self.paths.append(Path(root))
        return dict(self.budget)


def _fixture(tmp_path, verifier):
    budget = {"tail_horizon": 4, "action_budget": 8, "policy_call_cap": 8}
    receipt = {
        "status": "DIAGNOSTIC_SELECTED",
        "selection_source": "calibration_only",
        "confirmatory": False,
        "input_sha256": "1" * 64,
        "protocol_sha256": "2" * 64,
        "selected_budget": dict(budget),
        "original_rank": 1,
    }
    receipt_path = tmp_path / verifier.DIAGNOSTIC_RECEIPT_REL
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    auth = {
        "selection_receipt_sha256": receipt_sha,
        "git_commit": "a" * 40,
        "selection_path": verifier.DIAGNOSTIC_RECEIPT_REL,
        "diagnostic_atlas": True,
        "verified_origin_main": "b" * 40,
    }
    auth_path = tmp_path / verifier.DIAGNOSTIC_AUTH_REL
    auth_path.write_text(json.dumps(auth, sort_keys=True))
    row = {
        "diagnostic_continuation": True,
        "diagnostic_selection_receipt_sha256": receipt_sha,
        "diagnostic_selection_receipt_path": verifier.DIAGNOSTIC_RECEIPT_REL,
        "diagnostic_selection_binding": {
            "input_sha256": receipt["input_sha256"],
            "protocol_sha256": receipt["protocol_sha256"],
            "original_rank": receipt["original_rank"],
        },
        "configured_budget": dict(budget),
        **budget,
        "safe_success": 1,
    }
    atlas = tmp_path / "phase2" / "atlas_rows.jsonl"
    atlas.parent.mkdir(parents=True, exist_ok=True)
    atlas.write_text(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "phase": "phase2",
        "diagnostic_continuation": True,
        "confirmatory": False,
        "diagnostic_selection_receipt_sha256": receipt_sha,
        "selected_budget": dict(budget),
    }
    return summary, atlas, receipt_path, auth_path, receipt_sha, row, budget


def test_diagnostic_selection_is_independent_and_rechecked(tmp_path, monkeypatch):
    verifier = _verifier()
    summary, atlas, _, _, receipt_sha, _, budget = _fixture(tmp_path, verifier)
    selection = _Selection()
    matrix = _Matrix(budget)
    monkeypatch.setattr(verifier, "_load_selection_module", lambda root: selection)
    monkeypatch.setattr(verifier, "_load_matrix_module", lambda root: matrix)

    result = verifier._verify_phase2_artifacts(
        summary, atlas, tmp_path / "phase2" / "reference_rows.jsonl", tmp_path
    )

    assert result == {
        "diagnostic": True,
        "receipt_sha256": receipt_sha,
        "row_count": 1,
    }
    assert selection.calls and selection.calls[0][2] is True
    assert matrix.paths == [tmp_path / "artifacts" / "stage2f"]
    assert verifier._scientific_success(
        {"K3": "NONTRIVIAL_OPERATOR_RELATIVITY"}, True
    ) is False


def test_diagnostic_requires_both_receipt_and_authorization(tmp_path):
    verifier = _verifier()
    summary, atlas, receipt, auth, *_ = _fixture(tmp_path, verifier)
    auth.unlink()
    try:
        verifier._verify_phase2_artifacts(
            summary, atlas, tmp_path / "phase2" / "reference_rows.jsonl", tmp_path
        )
    except RuntimeError as exc:
        assert "independent selection receipt/auth" in str(exc)
    else:
        raise AssertionError("missing diagnostic authorization was accepted")


def test_diagnostic_sha_and_proof_fail_closed(tmp_path, monkeypatch):
    verifier = _verifier()
    summary, atlas, _, auth, *_ = _fixture(tmp_path, verifier)
    auth.write_text(
        json.dumps(
            {
                "selection_receipt_sha256": "0" * 64,
                "git_commit": "a" * 40,
            }
        )
    )
    selection = _Selection()
    matrix = _Matrix(
        {"tail_horizon": 4, "action_budget": 8, "policy_call_cap": 8}
    )
    monkeypatch.setattr(verifier, "_load_selection_module", lambda root: selection)
    monkeypatch.setattr(verifier, "_load_matrix_module", lambda root: matrix)
    try:
        verifier._verify_phase2_artifacts(
            summary, atlas, tmp_path / "phase2" / "reference_rows.jsonl", tmp_path
        )
    except RuntimeError as exc:
        assert "SHA256 mismatch" in str(exc)
    else:
        raise AssertionError("bad diagnostic receipt SHA was accepted")
    assert not selection.calls


def test_diagnostic_git_proof_false_is_rejected(tmp_path, monkeypatch):
    verifier = _verifier()
    summary, atlas, *_ = _fixture(tmp_path, verifier)
    selection = _Selection(result=False)
    matrix = _Matrix(
        {"tail_horizon": 4, "action_budget": 8, "policy_call_cap": 8}
    )
    monkeypatch.setattr(verifier, "_load_selection_module", lambda root: selection)
    monkeypatch.setattr(verifier, "_load_matrix_module", lambda root: matrix)
    try:
        verifier._verify_phase2_artifacts(
            summary, atlas, tmp_path / "phase2" / "reference_rows.jsonl", tmp_path
        )
    except RuntimeError as exc:
        assert "Git object proof invalid" in str(exc)
    else:
        raise AssertionError("false diagnostic proof was accepted")
    assert selection.calls and selection.calls[0][2] is True


def test_every_diagnostic_row_must_bind_receipt(tmp_path, monkeypatch):
    verifier = _verifier()
    summary, atlas, *_ = _fixture(tmp_path, verifier)
    row = json.loads(atlas.read_text())
    row["diagnostic_selection_receipt_sha256"] = "f" * 64
    atlas.write_text(json.dumps(row) + "\n")
    selection = _Selection()
    matrix = _Matrix(
        {"tail_horizon": 4, "action_budget": 8, "policy_call_cap": 8}
    )
    monkeypatch.setattr(verifier, "_load_selection_module", lambda root: selection)
    monkeypatch.setattr(verifier, "_load_matrix_module", lambda root: matrix)
    try:
        verifier._verify_phase2_artifacts(
            summary, atlas, tmp_path / "phase2" / "reference_rows.jsonl", tmp_path
        )
    except RuntimeError as exc:
        assert "row 1 receipt SHA256 mismatch" in str(exc)
    else:
        raise AssertionError("unbound diagnostic row was accepted")


def test_diagnostic_summary_must_be_non_confirmatory(tmp_path):
    verifier = _verifier()
    summary, atlas, *_ = _fixture(tmp_path, verifier)
    summary["confirmatory"] = True
    try:
        verifier._verify_phase2_artifacts(
            summary, atlas, tmp_path / "phase2" / "reference_rows.jsonl", tmp_path
        )
    except RuntimeError as exc:
        assert "confirmatory=false" in str(exc)
    else:
        raise AssertionError("confirmatory diagnostic summary was accepted")


def test_no_atlas_keeps_diagnostic_branch_closed(tmp_path, monkeypatch):
    verifier = _verifier()
    called = []

    def forbidden(root):
        called.append(root)
        raise AssertionError("diagnostic admission should not run")

    monkeypatch.setattr(verifier, "_verify_diagnostic_selection", forbidden)
    result = verifier._verify_phase2_artifacts(
        {"execution_status": "NOT_RUN"},
        tmp_path / "phase2" / "atlas_rows.jsonl",
        tmp_path / "phase2" / "reference_rows.jsonl",
        tmp_path,
    )
    assert result["diagnostic"] is False
    assert called == []
    assert verifier._scientific_success(
        {"K3": "NONTRIVIAL_OPERATOR_RELATIVITY"}, False
    ) is True


def test_formal_atlas_uses_formal_proof_only(tmp_path, monkeypatch):
    verifier = _verifier()
    atlas = tmp_path / "phase2" / "atlas_rows.jsonl"
    atlas.parent.mkdir(parents=True, exist_ok=True)
    atlas.write_text(json.dumps({"safe_success": 1}) + "\n")
    receipt = tmp_path / verifier.FORMAL_RECEIPT_REL
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text('{"status":"SELECTED"}')
    auth = tmp_path / verifier.FORMAL_AUTH_REL
    auth.write_text(
        json.dumps(
            {
                "selection_receipt_sha256": hashlib.sha256(
                    receipt.read_bytes()
                ).hexdigest(),
                "git_commit": "a" * 40,
            }
        )
    )
    selection = _Selection()
    monkeypatch.setattr(verifier, "_load_selection_module", lambda root: selection)
    result = verifier._verify_phase2_artifacts(
        {"execution_status": "COMPLETE"},
        atlas,
        tmp_path / "phase2" / "reference_rows.jsonl",
        tmp_path,
    )
    assert result["diagnostic"] is False
    assert selection.calls and selection.calls[0][2] is False
