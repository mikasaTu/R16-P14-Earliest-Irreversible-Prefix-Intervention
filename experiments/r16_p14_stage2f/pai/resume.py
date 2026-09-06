"""Fail-closed resume of this lineage's blackout-stopped PAI jobs.

The module has a single bounded ``run_once`` operation so the long-running
controller can be tested without a PAI call. It never retries a run after an
uncertain CreateJob outcome; a sealed submission claim remains for manual
reconciliation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

REG = Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/pai-job-registry")
DEFAULT_HEARTBEAT = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2f/s1-20260907/control/heartbeat.json"
)
DEFAULT_CONFIG = Path("/workspace/leon/.dlc/config")
TERMINAL = {"Succeeded", "Failed", "Stopped", "Deleted"}
CREATED_STATES = {
    "submitted_verified",
    "submitted_pending_readback",
    "submitted_contract_mismatch",
    "submitted_request_contract_mismatch",
}
UNCERTAIN_STATES = {
    "uncertain_timeout",
    "uncertain_submit_nonzero",
    "uncertain_job_id_parse",
    "uncertain_interrupted",
}
JOB_ID_RE = re.compile(r"dlc[a-z0-9]{8,}")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,63}")
BEIJING = ZoneInfo("Asia/Shanghai")


def write(path: Path, value: Any) -> None:
    """Atomically write mutable controller state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".resume.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _clock(value: dt.datetime | None = None) -> dt.datetime:
    value = value or dt.datetime.now(dt.timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _parse_time(value: Any) -> dt.datetime:
    if not isinstance(value, str):
        raise ValueError("heartbeat time must be an ISO-8601 string")
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("heartbeat time must include timezone")
    return parsed.astimezone(dt.timezone.utc)


def blackout(value: dt.datetime | None = None) -> bool:
    local = _clock(value).astimezone(BEIJING)
    minute = local.hour * 60 + local.minute
    return 565 <= minute < 580 or 1165 <= minute < 1180


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _job_id(value: Any) -> str | None:
    value = str(value or "")
    return value if JOB_ID_RE.fullmatch(value) else None


def _run_id(value: Any) -> str:
    value = str(value or "")
    if not RUN_ID_RE.fullmatch(value):
        raise ValueError(f"invalid run id: {value!r}")
    return value


def admission(
    heartbeat: Mapping[str, Any],
    *,
    at: dt.datetime | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Return the local preflight view; pai-job remains the authoritative gate."""
    current = _clock(at)
    if blackout(current):
        return False, "BLACKOUT", {"checked_at_utc": current.isoformat()}
    try:
        age = (current - _parse_time(heartbeat.get("time"))).total_seconds()
    except (TypeError, ValueError) as exc:
        return False, f"HEARTBEAT_INVALID:{exc}", {"checked_at_utc": current.isoformat()}
    if age < 0 or age > 60:
        return False, "HEARTBEAT_STALE", {"checked_at_utc": current.isoformat(), "heartbeat_age_seconds": age}
    if heartbeat.get("resume_allowed") is not True:
        return False, "CONTROLLER_DENIED", {"checked_at_utc": current.isoformat(), "heartbeat_age_seconds": age}
    return True, "OK", {"checked_at_utc": current.isoformat(), "heartbeat_age_seconds": age}


def _active_pending(state: Mapping[str, Any]) -> tuple[int, int, int]:
    jobs = state.get("jobs", [])
    active = sum(1 for job in jobs if job.get("status") not in TERMINAL)
    claims = state.get("submission_claims", {})
    pending = sum(1 for claim in claims.values() if not claim.get("job_id")) if isinstance(claims, Mapping) else 0
    return active, pending, active + pending


def eligible(
    state: Mapping[str, Any],
    heartbeat: Mapping[str, Any],
    *,
    at: dt.datetime | None = None,
) -> list[dict[str, Any]]:
    """List only unclaimed BLACKOUT source jobs when local admission is open."""
    admitted, _, _ = admission(heartbeat, at=at)
    if not admitted:
        return []
    _, _, total = _active_pending(state)
    if total >= 2:
        return []
    return [
        job
        for job in state.get("jobs", [])
        if job.get("status") == "Stopped"
        and job.get("stop_reason") == "BLACKOUT"
        and not job.get("resumed_by")
        and not job.get("resume_claimed")
        and not job.get("resume_target_run_id")
    ]


@contextmanager
def _locked_state(manifest: Path):
    """Yield mutable state under a lock; callers must not sleep in this block."""
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = _load(manifest)
        yield state
        write(manifest, state)


def _append(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def _run_dir(run_id: str) -> Path:
    return REG / "runs" / run_id


def _target_artifacts(run_id: str) -> tuple[Path, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    target = _run_dir(run_id)
    resolved_path = target / "resolved.json"
    resolved = _load(resolved_path) if resolved_path.is_file() else None
    result_path = target / "result.json"
    result = _load(result_path) if result_path.is_file() else None
    submission_path = target / "submission-state.json"
    submission = _load(submission_path) if submission_path.is_file() else None
    return target, resolved, result, submission


def _resolved_artifact_dir(resolved: Mapping[str, Any] | None) -> str:
    if not isinstance(resolved, Mapping):
        raise RuntimeError("resolved manifest is missing")
    artifact = resolved.get("artifact_dir")
    if not isinstance(artifact, str) or not artifact.startswith("/"):
        raise RuntimeError("resolved artifact_dir is missing or not absolute")
    return artifact


def _creation_receipt(
    result: Mapping[str, Any] | None,
    command_output: str = "",
) -> tuple[str | None, str | None]:
    """Return (JobId, state) only for a receipt that proves CreateJob happened."""
    if not isinstance(result, Mapping):
        matches = JOB_ID_RE.findall(command_output)
        return (matches[-1], None) if len(set(matches)) == 1 else (None, None)
    job_id = _job_id(result.get("job_id"))
    state = str(result.get("submission_state") or "")
    if job_id and (state in CREATED_STATES or result.get("returncode") == 0):
        return job_id, state or "created"
    return None, state or None


def _source_for_target(state: Mapping[str, Any], target_run_id: str) -> dict[str, Any] | None:
    for job in state.get("jobs", []):
        if job.get("resume_target_run_id") == target_run_id:
            return job
    return None


def _job_record(
    source: Mapping[str, Any],
    target_run_id: str,
    job_id: str,
    resolved: Mapping[str, Any],
    *,
    claimed_at_utc: Any = None,
) -> dict[str, Any]:
    root_id = str(source.get("root_run_id") or source["run_id"])
    sequence = int(source.get("resume_sequence", 0))
    entry = {
        key: source[key]
        for key in ("phase", "task", "gpus", "source_commit", "source_tree")
        if key in source
    }
    entry.update(
        {
            "run_id": target_run_id,
            "root_run_id": root_id,
            "resume_sequence": sequence,
            "job_id": job_id,
            "status": "Created",
            # The canonical admission claim is the conservative resource
            # start.  Keep its timestamp when registering after readback;
            # using the later reconciliation time would under-account usage.
            "created_at_utc": str(claimed_at_utc) if isinstance(claimed_at_utc, str) else now(),
            "artifact_dir": _resolved_artifact_dir(resolved),
            "resolved_path": str(_run_dir(target_run_id) / "resolved.json"),
            "resumed_from_job_id": source.get("job_id"),
            "resumed_from_run_id": source.get("run_id"),
            "resume_reason": "BLACKOUT",
            "persisted_completion_verified": False,
        }
    )
    return entry


def _register_created(manifest: Path, target_run_id: str, job_id: str, resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Fill a canonical claim and append exactly one job record under lock."""
    with _locked_state(manifest) as state:
        claims = state.setdefault("submission_claims", {})
        claim = claims.setdefault(target_run_id, {})
        existing_jobs = [job for job in state.get("jobs", []) if job.get("run_id") == target_run_id or job.get("job_id") == job_id]
        for existing in existing_jobs:
            if existing.get("job_id") != job_id or existing.get("run_id") != target_run_id:
                raise RuntimeError("created submission conflicts with an existing registration")
        claim_job = _job_id(claim.get("job_id"))
        if claim_job and claim_job != job_id:
            raise RuntimeError("submission claim JobId conflicts with creation receipt")
        source = _source_for_target(state, target_run_id)
        if source is None:
            raise RuntimeError("created submission has no BLACKOUT source registration")
        claim.update({"job_id": job_id, "state": "registered", "resolved_path": str(_run_dir(target_run_id) / "resolved.json")})
        if existing_jobs:
            # A prior process may have appended the exact record but exited
            # before updating the source marker.  Repair only that idempotent
            # bookkeeping under the same lock; never append a second record.
            source["resumed_by"] = job_id
            source["resume_state"] = "registered"
            return existing_jobs[0]
        entry = _job_record(
            source,
            target_run_id,
            job_id,
            resolved,
            claimed_at_utc=claim.get("claimed_at_utc"),
        )
        state.setdefault("jobs", []).append(entry)
        source["resumed_by"] = job_id
        source["resume_state"] = "registered"
        return entry


def _mark_needs_reconciliation(manifest: Path, target_run_id: str, reason: str) -> None:
    with _locked_state(manifest) as state:
        claim = state.setdefault("submission_claims", {}).setdefault(target_run_id, {})
        claim["state"] = "needs-reconciliation"
        claim["needs_reconciliation_reason"] = reason[:500]
        source = _source_for_target(state, target_run_id)
        if source is not None:
            source["resume_state"] = "needs-reconciliation"
    _append(
        manifest.parent / "resume_needs_reconciliation.jsonl",
        {"time": now(), "run_id": target_run_id, "reason": reason[:500]},
    )


def reconcile_claims(manifest: Path) -> list[dict[str, Any]]:
    """Reconcile result.json receipts without retrying any sealed run."""
    state = _load(manifest)
    claims = state.get("submission_claims", {})
    if not isinstance(claims, Mapping):
        return []
    actions: list[dict[str, Any]] = []
    for target_run_id, claim in list(claims.items()):
        target_run_id = str(target_run_id)
        # The manifest is shared with the initial calibration submitter.  Its
        # claims have no resume_target relationship and must remain entirely
        # outside this BLACKOUT-resume reconciler.
        if _source_for_target(state, target_run_id) is None:
            continue
        if not isinstance(claim, Mapping):
            _mark_needs_reconciliation(manifest, target_run_id, "malformed submission claim")
            continue
        # A sealed uncertain claim is a manual-reconciliation boundary.  Keep
        # it immutable on later polling cycles; in particular, do not append a
        # duplicate log record or make it eligible for another CreateJob.
        if claim.get("state") == "needs-reconciliation" and not claim.get("job_id"):
            continue
        try:
            _run_id(target_run_id)
            _, resolved, result, submission = _target_artifacts(target_run_id)
        except (OSError, TypeError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            _mark_needs_reconciliation(manifest, target_run_id, str(exc))
            continue
        claim_job = _job_id(claim.get("job_id"))
        if claim_job:
            if any(job.get("run_id") == target_run_id and job.get("job_id") == claim_job for job in state.get("jobs", [])):
                continue
            if resolved is None:
                _mark_needs_reconciliation(manifest, target_run_id, "claim has JobId but resolved manifest is missing")
                continue
            try:
                entry = _register_created(manifest, target_run_id, claim_job, resolved)
            except RuntimeError as exc:
                _mark_needs_reconciliation(manifest, target_run_id, str(exc))
                continue
            actions.append({"action": "registered_existing_claim", "run_id": target_run_id, "job_id": claim_job, "entry": entry})
            continue
        job_id, receipt_state = _creation_receipt(result)
        if job_id:
            if resolved is None:
                _mark_needs_reconciliation(manifest, target_run_id, "creation receipt has JobId but resolved manifest is missing")
                continue
            try:
                entry = _register_created(manifest, target_run_id, job_id, resolved)
            except RuntimeError as exc:
                _mark_needs_reconciliation(manifest, target_run_id, str(exc))
                continue
            actions.append({"action": "registered_created_receipt", "run_id": target_run_id, "job_id": job_id, "state": receipt_state, "entry": entry})
        elif receipt_state in UNCERTAIN_STATES or (isinstance(submission, Mapping) and submission.get("state") in UNCERTAIN_STATES):
            _mark_needs_reconciliation(manifest, target_run_id, f"sealed uncertain submission state: {receipt_state or submission.get('state')}")
        elif claim.get("state") == "precreate":
            # The canonical pre-create claim is itself evidence that the
            # process may have crossed CreateJob without a durable result.
            # Treat the absence of a receipt as uncertain and seal it rather
            # than trying the same run_id again.
            _mark_needs_reconciliation(manifest, target_run_id, "precreate claim has no durable creation receipt")
    return actions


def _command(args: list[str], *, timeout: int, runner: Callable[..., Any] | None = None) -> Any:
    if runner is not None:
        return runner(args, capture_output=True, text=True, timeout=timeout, check=False)
    # pai-job requires the canonical registry cwd for its controller and
    # source/toolchain pin checks.  Keep injected test runners untouched.
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        cwd=str(REG),
    )


def _output_text(result: Any) -> str:
    return "\n".join(str(getattr(result, key, "") or "") for key in ("stdout", "stderr"))


def _preflight_denied(submission: Mapping[str, Any] | None) -> bool:
    """A refusal before the canonical claim/CreateJob boundary is retryable."""
    return isinstance(submission, Mapping) and submission.get("state") == "preflight_failed_sealed"


def _claim_source(manifest: Path, heartbeat: Mapping[str, Any], at: dt.datetime | None) -> dict[str, Any] | None:
    with _locked_state(manifest) as state:
        admitted, reason, details = admission(heartbeat, at=at)
        if not admitted:
            return None
        _, _, total = _active_pending(state)
        if total >= 2:
            return None
        candidates = eligible(state, heartbeat, at=at)
        if not candidates:
            return None
        source = next(job for job in state["jobs"] if job.get("run_id") == candidates[0].get("run_id"))
        sequence = int(source.get("resume_sequence", 0)) + 1
        root_id = str(source.get("root_run_id") or source["run_id"])
        target_run_id = _run_id(f"{root_id}-b{sequence}")
        source.update(
            {
                "resume_claimed": now(),
                "resume_sequence": sequence,
                "resume_target_run_id": target_run_id,
                "resume_state": "clone_pending",
                "resume_admission": {"reason": reason, **details},
            }
        )
        return dict(source)


def _mark_state(manifest: Path, source_run_id: str, **updates: Any) -> None:
    with _locked_state(manifest) as state:
        for source in state.get("jobs", []):
            if source.get("run_id") == source_run_id:
                source.update(updates)
                return


def _record_preflight_failure(
    manifest: Path,
    source_run_id: str,
    target_run_id: str,
    *,
    submission: Mapping[str, Any] | None,
    reason: str,
) -> None:
    """Retain a sealed preflight target, then release its source for -bN+1."""
    with _locked_state(manifest) as state:
        for source in state.get("jobs", []):
            if source.get("run_id") != source_run_id:
                continue
            failed = source.setdefault("resume_failed_targets", [])
            if not any(item.get("run_id") == target_run_id for item in failed if isinstance(item, Mapping)):
                failed.append(
                    {
                        "run_id": target_run_id,
                        "state": (submission or {}).get("state") if isinstance(submission, Mapping) else None,
                        "failed_at_utc": now(),
                        "reason": reason[-500:],
                    }
                )
            # Keep target_dir/result/submission-state on disk for audit.  The
            # run_id is sealed by pai-job, so the next attempt must get a new
            # deterministic sequence id and a fresh clone.
            source.pop("resume_claimed", None)
            source.pop("resume_target_run_id", None)
            source["resume_state"] = "preflight_failed"
            return


def _pending_target(manifest: Path) -> dict[str, Any] | None:
    state = _load(manifest)
    claims = state.get("submission_claims", {})
    for source in state.get("jobs", []):
        target = source.get("resume_target_run_id")
        if not target or source.get("resumed_by") or source.get("resume_state") in {"needs-reconciliation", "registered", "clone_failed"}:
            continue
        claim_present = isinstance(claims, Mapping) and str(target) in claims
        claim = claims.get(str(target), {}) if claim_present else {}
        # A claim without a JobId is a sealed pre-create boundary until
        # reconcile_claims can prove the result.  Never blindly re-submit it.
        if claim_present and isinstance(claim, Mapping) and not _job_id(claim.get("job_id")):
            continue
        _, _, result, submission = _target_artifacts(str(target))
        if result is not None:
            job_id, receipt_state = _creation_receipt(result)
            if job_id or receipt_state in UNCERTAIN_STATES:
                continue
        return dict(source)
    return None


def run_once(
    manifest: str | Path,
    *,
    heartbeat_path: str | Path = DEFAULT_HEARTBEAT,
    config_path: str | Path = DEFAULT_CONFIG,
    command_runner: Callable[..., Any] | None = None,
    at: dt.datetime | None = None,
) -> dict[str, Any]:
    """Perform one bounded reconciliation/clone/submit cycle."""
    manifest = Path(manifest)
    heartbeat_path = Path(heartbeat_path)
    if not manifest.is_file():
        return {"action": "blocked", "reason": "manifest_missing"}
    if not heartbeat_path.is_file():
        return {"action": "blocked", "reason": "heartbeat_missing"}
    try:
        heartbeat = _load(heartbeat_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"action": "blocked", "reason": f"heartbeat_invalid:{exc}"}
    reconciliation = reconcile_claims(manifest)
    source = _pending_target(manifest)
    if source is None:
        source = _claim_source(manifest, heartbeat, at)
    if source is None:
        admitted, reason, _ = admission(heartbeat, at=at)
        state = _load(manifest)
        active, pending, total = _active_pending(state)
        return {"action": "idle" if admitted and total < 2 else "blocked", "reason": reason if not admitted else "CAP" if total >= 2 else "NO_CANDIDATE", "reconciled": reconciliation, "active": active, "pending_claims": pending}
    source_run_id = str(source["run_id"])
    target_run_id = str(source["resume_target_run_id"])
    target_dir = _run_dir(target_run_id)
    cli = str(REG / "bin" / "pai-job")
    if not target_dir.exists():
        clone = _command([cli, "clone", "--from-run", source_run_id, "--run-id", target_run_id], timeout=300, runner=command_runner)
        if getattr(clone, "returncode", 1) != 0:
            _mark_state(manifest, source_run_id, resume_state="clone_failed")
            return {"action": "clone_failed", "run_id": target_run_id, "reconciled": reconciliation, "output": _output_text(clone)}
        _mark_state(manifest, source_run_id, resume_state="cloned_waiting_admission")
    else:
        _mark_state(manifest, source_run_id, resume_state="cloned_waiting_admission")

    # The clone can take long enough to cross a blackout or heartbeat expiry.
    # Recheck immediately before submit-resolved; pai-job repeats this under
    # the global pre-CreateJob lock.
    try:
        current_heartbeat = _load(heartbeat_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"action": "blocked_after_clone", "reason": f"heartbeat_invalid:{exc}", "run_id": target_run_id}
    admitted, reason, details = admission(current_heartbeat, at=at)
    if not admitted:
        _mark_state(manifest, source_run_id, resume_state="cloned_waiting_admission", resume_admission={"reason": reason, **details})
        return {"action": "blocked_after_clone", "reason": reason, "run_id": target_run_id}
    try:
        submit = _command(
            [cli, "submit-resolved", "--run-id", target_run_id, "--config", str(config_path)],
            timeout=1200,
            runner=command_runner,
        )
    except Exception as exc:
        # The wrapper can disappear or time out after canonical admission.
        # There is no safe way to infer whether CreateJob reached the remote
        # service, so seal the target and require reconciliation.
        reason_text = f"submit command outcome uncertain: {type(exc).__name__}: {exc}"
        _mark_needs_reconciliation(manifest, target_run_id, reason_text)
        return {"action": "needs-reconciliation", "reason": reason_text, "run_id": target_run_id}
    output = _output_text(submit)
    _, resolved, result, submission = _target_artifacts(target_run_id)
    job_id, receipt_state = _creation_receipt(result, output if getattr(submit, "returncode", 1) == 0 else "")
    if job_id and resolved is not None:
        try:
            entry = _register_created(manifest, target_run_id, job_id, resolved)
        except RuntimeError as exc:
            _mark_needs_reconciliation(manifest, target_run_id, str(exc))
            return {"action": "needs-reconciliation", "reason": str(exc), "run_id": target_run_id, "job_id": job_id}
        _append(manifest.parent / "resume_actions.jsonl", {"time": now(), "action": "registered", "job_id": job_id, "run_id": target_run_id, "from_run_id": source_run_id})
        return {"action": "registered", "run_id": target_run_id, "job_id": job_id, "entry": entry, "reconciled": reconciliation}
    claim_state = _load(manifest).get("submission_claims", {}).get(target_run_id, {})
    if claim_state and (claim_state.get("state") == "precreate" or receipt_state in UNCERTAIN_STATES or (isinstance(submission, Mapping) and submission.get("state") in UNCERTAIN_STATES)):
        reason_text = f"submission uncertain: {receipt_state or (submission or {}).get('state') or output[-500:]}"
        _mark_needs_reconciliation(manifest, target_run_id, reason_text)
        return {"action": "needs-reconciliation", "reason": reason_text, "run_id": target_run_id}
    if getattr(submit, "returncode", 1) != 0:
        if _preflight_denied(submission):
            reason_text = output[-500:] or "canonical preflight admission denied"
            _record_preflight_failure(
                manifest,
                source_run_id,
                target_run_id,
                submission=submission,
                reason=reason_text,
            )
            return {
                "action": "submit_failed",
                "reason": reason_text,
                "run_id": target_run_id,
                "retry_run_id": "new-sequence-after-admission",
            }
        reason_text = f"submission outcome uncertain: {output[-500:]}"
        _mark_needs_reconciliation(manifest, target_run_id, reason_text)
        return {"action": "needs-reconciliation", "reason": reason_text, "run_id": target_run_id}
    reason_text = "successful submit lacked a sealed JobId receipt"
    _mark_needs_reconciliation(manifest, target_run_id, reason_text)
    return {"action": "needs-reconciliation", "reason": reason_text, "run_id": target_run_id}


def run(
    manifest: str | Path,
    *,
    heartbeat_path: str | Path = DEFAULT_HEARTBEAT,
    config_path: str | Path = DEFAULT_CONFIG,
    interval: int = 15,
    once: bool = False,
) -> None:
    manifest = Path(manifest)
    # Keep the daemon's current directory aligned with pai-job's canonical
    # registry requirement before invoking any state-changing CLI.
    os.chdir(REG)
    while True:
        try:
            run_once(manifest, heartbeat_path=heartbeat_path, config_path=config_path)
        except BaseException as exc:
            _append(manifest.parent / "resume_errors.jsonl", {"time": now(), "error_type": type(exc).__name__, "message": str(exc)[:500]})
        if once:
            return
        # Never sleep while holding the manifest lock: run_once has returned
        # only after every lock context has exited.
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--heartbeat", default=str(DEFAULT_HEARTBEAT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run(
        args.manifest,
        heartbeat_path=args.heartbeat,
        config_path=args.config,
        interval=args.interval,
        once=args.once,
    )
