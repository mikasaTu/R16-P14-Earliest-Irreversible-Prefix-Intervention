from __future__ import annotations

import errno
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from experiments.r16_p14_stage2f.pai import resume


TASK = "put_the_cream_cheese_in_the_bowl"


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _source():
    return {
        "run_id": "s1-source",
        "root_run_id": "s1-source",
        "job_id": "dlcsource12345678",
        "phase": "collect",
        "task": TASK,
        "gpus": 2,
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "status": "Stopped",
        "stop_reason": "BLACKOUT",
        "artifact_dir": "/actual/source/artifact",
    }


def _setup(tmp_path, monkeypatch, *, jobs=None, claims=None):
    registry = tmp_path / "registry"
    manifest = tmp_path / "control" / "jobs.json"
    heartbeat = tmp_path / "control" / "heartbeat.json"
    at = datetime(2026, 9, 7, 2, 1, tzinfo=timezone.utc)  # 10:01 Beijing
    _write(manifest, {"jobs": jobs or [_source()], "submission_claims": claims or {}})
    _write(heartbeat, {"time": (at - timedelta(seconds=30)).isoformat(), "resume_allowed": True})
    monkeypatch.setattr(resume, "REG", registry)
    return manifest, heartbeat, at


def test_load_reopens_cpfs_json_after_estale_and_partial_json(tmp_path):
    target = tmp_path / "jobs.json"
    with patch.object(
        Path,
        "read_text",
        side_effect=[OSError(errno.ESTALE, "stale"), "{", '{"jobs": []}'],
    ) as read:
        assert resume._load(target, attempts=3, delay=0) == {"jobs": []}
    assert read.call_count == 3


def test_load_persistent_estale_is_not_success(tmp_path):
    target = tmp_path / "jobs.json"
    with patch.object(Path, "read_text", side_effect=OSError(errno.ESTALE, "stale")) as read:
        with pytest.raises(OSError):
            resume._load(target, attempts=3, delay=0)
    assert read.call_count == 3


def test_blackout_and_stale_heartbeat_deny_without_commands(tmp_path, monkeypatch):
    manifest, heartbeat, at = _setup(tmp_path, monkeypatch)
    calls = []
    blackout_at = datetime(2026, 9, 7, 1, 30, tzinfo=timezone.utc)  # 09:30 Beijing
    result = resume.run_once(manifest, heartbeat_path=heartbeat, command_runner=lambda *a, **k: calls.append(a), at=blackout_at)
    assert result["reason"] == "BLACKOUT"
    assert calls == []

    _write(heartbeat, {"time": (at - timedelta(seconds=61)).isoformat(), "resume_allowed": True})
    result = resume.run_once(manifest, heartbeat_path=heartbeat, command_runner=lambda *a, **k: calls.append(a), at=at)
    assert result["reason"] == "HEARTBEAT_STALE"
    assert calls == []


def test_cap_counts_active_jobs_and_unfilled_claims(tmp_path, monkeypatch):
    active = {"run_id": "active", "job_id": "dlcactive12345678", "status": "Running", "gpus": 2}
    manifest, heartbeat, at = _setup(tmp_path, monkeypatch, jobs=[_source(), active], claims={"pending-run": {"state": "precreate"}})
    calls = []
    result = resume.run_once(manifest, heartbeat_path=heartbeat, command_runner=lambda *a, **k: calls.append(a), at=at)
    assert result["reason"] == "CAP"
    assert result["active"] == 1
    assert result["pending_claims"] == 1
    assert calls == []


def test_success_fills_claim_registers_exact_resolved_artifact_once(tmp_path, monkeypatch):
    manifest, heartbeat, at = _setup(tmp_path, monkeypatch)
    calls = []
    actual_artifact = "/resolved/storage/path/that-is-not-inferred"

    def runner(args, **kwargs):
        calls.append(args)
        target = args[args.index("--run-id") + 1]
        target_dir = resume.REG / "runs" / target
        target_dir.mkdir(parents=True, exist_ok=True)
        if args[1] == "clone":
            _write(target_dir / "resolved.json", {"run_id": target, "artifact_dir": actual_artifact})
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        state = json.loads(manifest.read_text())
        state.setdefault("submission_claims", {})[target] = {"state": "precreate", "gpus": 2}
        _write(manifest, state)
        _write(target_dir / "result.json", {"run_id": target, "returncode": 0, "submission_state": "submitted_verified", "job_id": "dlcnew12345678"})
        _write(target_dir / "submission-state.json", {"state": "submitted_verified", "job_id": "dlcnew12345678"})
        return SimpleNamespace(returncode=0, stdout="dlcnew12345678\n", stderr="")

    result = resume.run_once(manifest, heartbeat_path=heartbeat, command_runner=runner, at=at)
    assert result["action"] == "registered"
    state = json.loads(manifest.read_text())
    assert state["submission_claims"][result["run_id"]]["job_id"] == "dlcnew12345678"
    registered = [job for job in state["jobs"] if job.get("run_id") == result["run_id"]]
    assert len(registered) == 1
    assert registered[0]["artifact_dir"] == actual_artifact
    assert registered[0]["resolved_path"].endswith(f"runs/{result['run_id']}/resolved.json")

    second = resume.run_once(manifest, heartbeat_path=heartbeat, command_runner=runner, at=at)
    assert second["action"] in {"idle", "blocked"}
    assert sum(args[1] == "submit-resolved" for args in calls) == 1
    state = json.loads(manifest.read_text())
    assert len([job for job in state["jobs"] if job.get("run_id") == result["run_id"]]) == 1


def test_uncertain_claim_is_logged_and_never_retried(tmp_path, monkeypatch):
    manifest, heartbeat, at = _setup(tmp_path, monkeypatch)
    calls = []

    def runner(args, **kwargs):
        calls.append(args)
        target = args[args.index("--run-id") + 1]
        target_dir = resume.REG / "runs" / target
        target_dir.mkdir(parents=True, exist_ok=True)
        if args[1] == "clone":
            _write(target_dir / "resolved.json", {"run_id": target, "artifact_dir": "/resolved/artifact"})
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        state = json.loads(manifest.read_text())
        state.setdefault("submission_claims", {})[target] = {"state": "precreate", "gpus": 2}
        _write(manifest, state)
        _write(target_dir / "result.json", {"run_id": target, "submission_state": "uncertain_timeout"})
        _write(target_dir / "submission-state.json", {"state": "uncertain_timeout"})
        return SimpleNamespace(returncode=1, stdout="", stderr="timeout")

    first = resume.run_once(manifest, heartbeat_path=heartbeat, command_runner=runner, at=at)
    assert first["action"] == "needs-reconciliation"
    state = json.loads(manifest.read_text())
    target = first["run_id"]
    assert state["submission_claims"][target]["state"] == "needs-reconciliation"
    second = resume.run_once(manifest, heartbeat_path=heartbeat, command_runner=runner, at=at)
    assert second["action"] in {"idle", "blocked"}
    assert sum(args[1] == "submit-resolved" for args in calls) == 1
    log = (manifest.parent / "resume_needs_reconciliation.jsonl").read_text()
    assert target in log


def test_admission_is_rechecked_after_clone(tmp_path, monkeypatch):
    manifest, heartbeat, at = _setup(tmp_path, monkeypatch)
    calls = []
    admission_calls = []
    original = resume.admission

    def closing_admission(hb, *, at=None):
        admission_calls.append(len(admission_calls))
        if len(admission_calls) >= 3:
            return False, "BLACKOUT", {"checked_at_utc": at.isoformat() if at else "test"}
        return original(hb, at=at)

    monkeypatch.setattr(resume, "admission", closing_admission)

    def runner(args, **kwargs):
        calls.append(args)
        target = args[args.index("--run-id") + 1]
        target_dir = resume.REG / "runs" / target
        target_dir.mkdir(parents=True, exist_ok=True)
        _write(target_dir / "resolved.json", {"run_id": target, "artifact_dir": "/resolved/artifact"})
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = resume.run_once(manifest, heartbeat_path=heartbeat, command_runner=runner, at=at)
    assert result["action"] == "blocked_after_clone"
    assert result["reason"] == "BLACKOUT"
    assert any(args[1] == "clone" for args in calls)
    assert not any(args[1] == "submit-resolved" for args in calls)


def test_existing_claim_job_id_is_registered_without_submission(tmp_path, monkeypatch):
    target = "s1-source-b1"
    claims = {target: {"job_id": "dlcrecovered12345678", "state": "created"}}
    manifest, heartbeat, at = _setup(tmp_path, monkeypatch, claims=claims)
    target_dir = resume.REG / "runs" / target
    target_dir.mkdir(parents=True, exist_ok=True)
    _write(target_dir / "resolved.json", {"run_id": target, "artifact_dir": "/resolved/actual"})
    # The source keeps the target relationship required to reconstruct the
    # exact registration, while the claim itself is already authoritative.
    state = json.loads(manifest.read_text())
    state["jobs"][0].update({"resume_target_run_id": target, "resume_sequence": 1, "resume_claimed": "now"})
    _write(manifest, state)
    calls = []
    result = resume.run_once(manifest, heartbeat_path=heartbeat, command_runner=lambda *a, **k: calls.append(a), at=at)
    assert result["action"] in {"idle", "blocked"}
    state = json.loads(manifest.read_text())
    assert len([job for job in state["jobs"] if job.get("run_id") == target]) == 1
    assert calls == []


def test_precreate_claim_without_receipt_is_sealed_and_not_retried(tmp_path, monkeypatch):
    target = "s1-source-b1"
    claims = {target: {"state": "precreate", "claimed_at_utc": "2026-09-07T02:00:00+00:00"}}
    manifest, heartbeat, at = _setup(tmp_path, monkeypatch, claims=claims)
    state = json.loads(manifest.read_text())
    state["jobs"][0].update({"resume_target_run_id": target, "resume_sequence": 1, "resume_claimed": "now"})
    _write(manifest, state)
    calls = []
    first = resume.run_once(manifest, heartbeat_path=heartbeat, command_runner=lambda *a, **k: calls.append(a), at=at)
    assert first["action"] in {"idle", "blocked"}
    updated = json.loads(manifest.read_text())
    assert updated["submission_claims"][target]["state"] == "needs-reconciliation"
    assert updated["jobs"][0]["resume_state"] == "needs-reconciliation"
    second = resume.run_once(manifest, heartbeat_path=heartbeat, command_runner=lambda *a, **k: calls.append(a), at=at)
    assert second["action"] in {"idle", "blocked"}
    assert calls == []
    log_lines = (manifest.parent / "resume_needs_reconciliation.jsonl").read_text().splitlines()
    assert len([line for line in log_lines if target in line]) == 1


def test_initial_submit_claim_without_blackout_source_is_untouched(tmp_path, monkeypatch):
    manifest, heartbeat, at = _setup(tmp_path, monkeypatch)
    target = "initial-calibration-run"
    claim = {"state": "precreate", "claimed_at_utc": "2026-09-07T02:00:00+00:00"}
    state = json.loads(manifest.read_text())
    state["submission_claims"] = {target: claim}
    _write(manifest, state)
    target_dir = resume.REG / "runs" / target
    target_dir.mkdir(parents=True, exist_ok=True)
    _write(target_dir / "resolved.json", {"run_id": target, "artifact_dir": "/cal/artifact"})
    _write(target_dir / "result.json", {"run_id": target, "job_id": "dlcinitial12345678", "submission_state": "submitted_verified", "returncode": 0})
    assert resume.reconcile_claims(manifest) == []
    updated = json.loads(manifest.read_text())
    assert updated["submission_claims"][target] == claim
    assert not (manifest.parent / "resume_needs_reconciliation.jsonl").exists()


def test_preflight_sealed_target_is_preserved_then_new_sequence_is_submitted(tmp_path, monkeypatch):
    manifest, heartbeat, at = _setup(tmp_path, monkeypatch)
    calls = []
    submit_count = 0

    def runner(args, **kwargs):
        nonlocal submit_count
        calls.append(args)
        target = args[args.index("--run-id") + 1]
        target_dir = resume.REG / "runs" / target
        target_dir.mkdir(parents=True, exist_ok=True)
        if args[1] == "clone":
            _write(target_dir / "resolved.json", {"run_id": target, "artifact_dir": f"/resolved/{target}"})
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        submit_count += 1
        if submit_count == 1:
            _write(target_dir / "submission-state.json", {"state": "preflight_failed_sealed"})
            return SimpleNamespace(returncode=1, stdout="", stderr="blackout admission closed")
        state = json.loads(manifest.read_text())
        state.setdefault("submission_claims", {})[target] = {"state": "precreate", "gpus": 2, "claimed_at_utc": "2026-09-07T02:02:00+00:00"}
        _write(manifest, state)
        _write(target_dir / "result.json", {"run_id": target, "returncode": 0, "submission_state": "submitted_verified", "job_id": "dlcnew12345678"})
        _write(target_dir / "submission-state.json", {"state": "submitted_verified", "job_id": "dlcnew12345678"})
        return SimpleNamespace(returncode=0, stdout="dlcnew12345678\n", stderr="")

    first = resume.run_once(manifest, heartbeat_path=heartbeat, command_runner=runner, at=at)
    assert first["action"] == "submit_failed"
    state = json.loads(manifest.read_text())
    source = state["jobs"][0]
    assert source.get("resume_target_run_id") is None
    assert source["resume_sequence"] == 1
    assert source["resume_failed_targets"][0]["run_id"] == "s1-source-b1"
    assert (resume.REG / "runs" / "s1-source-b1" / "submission-state.json").is_file()

    second = resume.run_once(manifest, heartbeat_path=heartbeat, command_runner=runner, at=at)
    assert second["action"] == "registered"
    submit_ids = [args[args.index("--run-id") + 1] for args in calls if args[1] == "submit-resolved"]
    assert submit_ids == ["s1-source-b1", "s1-source-b2"]
    state = json.loads(manifest.read_text())
    assert any(job.get("run_id") == "s1-source-b2" for job in state["jobs"])
    assert not any(job.get("run_id") == "s1-source-b1" for job in state["jobs"])
