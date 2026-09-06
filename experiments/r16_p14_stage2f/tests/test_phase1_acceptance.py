from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from s1.accept_phase1 import (  # noqa: E402
    EXPECTED_EVENTS,
    OPERATORS,
    REFERENCE_PREFIXES,
    SEEDS,
    SOURCE_COMMIT,
    TASKS,
    _check_trace,
    _expected_keys,
    _load_grid_jobs,
    evaluate_base,
    structural_exclusion_is_valid,
    write_report,
)


ALT_COMMIT = "b" * 40


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_source(root: Path) -> None:
    rows = []
    for task_index, task in enumerate(TASKS):
        count = EXPECTED_EVENTS[task]
        for index in range(count):
            actor = SEEDS[index % len(SEEDS)]
            rows.append(
                {
                    "task": task,
                    "event_instance_id": f"{task}__init{index:03d}__actor{actor}",
                    "init_state_id": index + task_index * 100,
                    "split": "calibration",
                    "actor_seed": actor,
                    "generator_actor_seed": actor,
                    "anchor_global_step": 100 + index,
                }
            )
    path = root / "artifacts" / "stage2f" / "phase0b" / "calibration_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_jobs(root: Path, *, status: str, source_commit: str = SOURCE_COMMIT) -> None:
    jobs = []
    for index, task in enumerate(TASKS):
        run_id = f"r16p14-s1-grid-t{index}-fixture"
        artifact = root / "pai_runs" / run_id
        jobs.append(
            {
                "task": task,
                "phase": "grid",
                "run_id": run_id,
                "job_id": f"dlc-fixture-{index}",
                "status": status,
                "gpus": 2,
                "source_commit": source_commit,
                "artifact_dir": str(artifact),
                "readback": {
                    "Status": status,
                    "JobId": f"dlc-fixture-{index}",
                },
            }
        )
    _write(root / "control" / "jobs.json", {"jobs": jobs})


def _completion(root: Path, task: str, *, valid: bool = True) -> None:
    path = root / "artifacts" / "stage2f" / "phase1" / f"completion_{task}.json"
    if not valid:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{bad", encoding="utf-8")
        return
    count = EXPECTED_EVENTS[task]
    _write(
        path,
        {
            "task": task,
            "phase": "grid",
            "events": count,
            "planned_events": 20,
            "planned_sample_complete": False,
            "missing_planned_events": 20 - count,
            "requested_rows": count * 3 * 9 * (4 * 5 + 4),
            "persisted_rows": count * 3 * 9 * (4 * 5 + 4),
        },
    )


def test_running_jobs_are_incomplete_and_do_not_need_evaluation_input(tmp_path):
    _write_source(tmp_path)
    _write_jobs(tmp_path, status="Running")
    sealed = tmp_path / "artifacts" / "stage2f" / "sealed_evaluation" / "must_not_read"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    sealed.write_text("this is deliberately not JSON", encoding="utf-8")
    report = evaluate_base(tmp_path, mode="hash-only")
    assert report["status"] == "INCOMPLETE"
    assert report["evaluation_raw_read"] is False
    assert report["source"]["events_by_task"] == EXPECTED_EVENTS
    assert report["incomplete_reasons"]


def test_terminal_jobs_without_both_completions_are_incomplete(tmp_path):
    _write_source(tmp_path)
    _write_jobs(tmp_path, status="Succeeded")
    report = evaluate_base(tmp_path)
    assert report["status"] == "INCOMPLETE"
    assert "one or both task completion JSON files are absent" in report["incomplete_reasons"]


def test_corrupt_completion_is_blocked_after_terminal_jobs(tmp_path):
    _write_source(tmp_path)
    _write_jobs(tmp_path, status="Succeeded")
    _completion(tmp_path, TASKS[0], valid=False)
    _completion(tmp_path, TASKS[1])
    report = evaluate_base(tmp_path)
    assert report["status"] == "BLOCKED"
    assert any("invalid completion JSON" in reason for reason in report["blocking_reasons"])


def test_explicit_source_allowlist_accepts_repair_commit_and_rejects_default(tmp_path):
    _write_source(tmp_path)
    _write_jobs(tmp_path, status="Running", source_commit=ALT_COMMIT)
    issues = []
    jobs, terminal, history = _load_grid_jobs(tmp_path, issues, {ALT_COMMIT})
    assert set(jobs) == set(TASKS)
    assert terminal is False
    assert set(history) == set(TASKS)
    assert issues == []
    issues = []
    _load_grid_jobs(tmp_path, issues, {SOURCE_COMMIT})
    assert any("source commit is not allowed" in reason for reason in issues)


def test_structural_marker_uses_source_anchor_not_error_string():
    source = {"anchor_global_step": 316}
    valid = {
        "task": TASKS[1],
        "status": "BLOCKED",
        "error_type": "PrefixOutsideTaskHorizon",
        "prefix_k": 8,
        "anchor_global_step": 316,
    }
    assert structural_exclusion_is_valid(valid, source) == (True, "")
    feasible = dict(valid, anchor_global_step=300)
    ok, reason = structural_exclusion_is_valid(feasible, dict(source, anchor_global_step=300))
    assert ok is False
    assert "feasible" in reason
    missing = dict(valid)
    source_missing = {}
    ok, reason = structural_exclusion_is_valid(missing, source_missing)
    assert ok is False
    assert "unavailable" in reason


def test_expected_grid_has_complete_cross_operator_budget_seed_keys():
    event = {
        "task": TASKS[0],
        "event_instance_id": "e",
        "init_state_id": "1",
        "split": "calibration",
        "generator_actor_seed": 7,
    }
    core, reference = _expected_keys([event])
    assert len(core) == 3 * 9 * len(OPERATORS) * 5
    assert len(reference) == 3 * 9 * len(REFERENCE_PREFIXES)
    assert all(key[-2] == 8 for key in core | reference)


def test_trace_full_and_hash_only_are_explicit_scopes(tmp_path):
    record = {
        "event_instance_id": "e",
        "operator": "fresh_h4",
        "prefix_k": 2,
        "actor_seed": 7,
        "normalized_contact_pairs": [["object_g1", "robot_pad"]],
        "contact_stream_available": True,
    }
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path = tmp_path / "e.jsonl.gz"
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(payload)
    raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    content_sha = hashlib.sha256(payload).hexdigest()
    row = {
        "task": TASKS[0],
        "event_instance_id": "e",
        "operator": "fresh_h4",
        "prefix_k": 2,
        "recovery_actor_seed": 7,
        "trace_sha256": raw_sha,
        "trace_content_sha256": content_sha,
        "trace_records": 1,
        "labels": {"record_count": 1, "label_complete": True},
    }
    report = {
        "files": 0,
        "records": 0,
        "bytes": 0,
        "bytes_by_task": {TASKS[0]: 0, TASKS[1]: 0},
        "files_by_task": {TASKS[0]: 0, TASKS[1]: 0},
    }
    issues = []
    _check_trace(row, path, "full", issues, report)
    assert issues == []
    assert report["records"] == 1
    issues = []
    _check_trace(row, path, "hash-only", issues, report)
    assert issues == []
    assert "not parsed" in report["scope_note"]


def test_acceptance_output_must_not_be_inside_gpu_base(tmp_path):
    report = {"status": "INCOMPLETE"}
    with pytest.raises(ValueError):
        write_report(report, tmp_path / "acceptance.json", tmp_path)

