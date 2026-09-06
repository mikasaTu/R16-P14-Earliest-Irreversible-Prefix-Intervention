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
    _check_d4_groups,
    _check_reconstruction,
    _check_trace,
    _expected_behavior_operator,
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


def _trace_fixture(tmp_path: Path, *, physics_count: int = 25, alias_ok: bool = True):
    records = []
    pair = {
        "geom1_name": "object_g1",
        "geom2_name": "robot_pad",
        "normalized_pair": ["object_g1", "robot_pad"],
    }
    for substep in range(1, physics_count + 1):
        records.append(
            {
                "event_instance_id": "e",
                "operator": "fresh_h4",
                "prefix_k": 2,
                "actor_seed": 7,
                "step_kind": "physics_step",
                "step": 0,
                "control_step": 0,
                "substep": substep,
                "physics_step_index": substep,
                "raw_contact_pairs": [pair],
                "normalized_contact_pairs": [["object_g1", "robot_pad"]],
                "contact_pairs": [["object_g1", "robot_pad"]] if alias_ok else [],
                "contact_count": 1,
                "contact_stream_available": True,
            }
        )
    records.append(
        {
            "event_instance_id": "e",
            "operator": "fresh_h4",
            "prefix_k": 2,
            "actor_seed": 7,
            "step_kind": "action_step",
            "step": 0,
            "control_step": 0,
            "substep": 0,
            "physics_step_index": None,
            "raw_contact_pairs": [pair],
            "normalized_contact_pairs": [["object_g1", "robot_pad"]],
            "contact_pairs": [["object_g1", "robot_pad"]] if alias_ok else [],
            "contact_count": 1,
            "contact_stream_available": True,
        }
    )
    payload = b"".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for record in records
    )
    path = tmp_path / "e.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(payload)
    row = {
        "task": TASKS[0],
        "event_instance_id": "e",
        "operator": "fresh_h4",
        "prefix_k": 2,
        "recovery_actor_seed": 7,
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "trace_content_sha256": hashlib.sha256(payload).hexdigest(),
        "trace_records": len(records),
        "labels": {"record_count": len(records), "label_complete": True},
    }
    report = {
        "files": 0,
        "records": 0,
        "bytes": 0,
        "bytes_by_task": {TASKS[0]: 0, TASKS[1]: 0},
        "files_by_task": {TASKS[0]: 0, TASKS[1]: 0},
    }
    return row, path, report


def test_reference_behavior_operator_uses_fresh_h16():
    assert _expected_behavior_operator("immediate_fresh", True) == "fresh_h16"
    assert _expected_behavior_operator("fresh_h4", False) == "fresh_h4"


def test_trace_full_and_hash_only_are_explicit_scopes(tmp_path):
    row, path, report = _trace_fixture(tmp_path)
    issues = []
    _check_trace(row, path, "full", issues, report)
    assert issues == []
    assert report["records"] == 26
    assert report["physics_per_control_counts"] == {"25": 1}
    issues = []
    _check_trace(row, path, "hash-only", issues, report)
    assert issues == []
    assert "not parsed" in report["scope_note"]


def test_trace_requires_raw_pair_alias_and_25_physics_records(tmp_path):
    row, path, report = _trace_fixture(tmp_path, alias_ok=False)
    issues = []
    _check_trace(row, path, "full", issues, report)
    assert any("contact_pairs alias" in reason for reason in issues)
    row, path, report = _trace_fixture(tmp_path / "short", physics_count=24)
    issues = []
    _check_trace(row, path, "full", issues, report)
    assert any("25 physics" in reason for reason in issues)


def test_reconstruction_error_must_be_finite_and_nonnegative():
    row = {
        "reconstruction": {
            "action_history_exact": True,
            "actor_inference_side_effect_free": True,
            "anchor_state_exact": True,
            "event_chunk_exact": True,
            "state_history_exact": True,
            "max_anchor_state_error": float("nan"),
        }
    }
    issues = []
    _check_reconstruction(row, issues, "fixture")
    assert any("finite" in reason for reason in issues)
    row["reconstruction"]["max_anchor_state_error"] = -1e-12
    issues = []
    _check_reconstruction(row, issues, "fixture")
    assert any("[0,1e-9]" in reason for reason in issues)


def test_d4_group_requires_cross_budget_seed_operator_signature_match():
    base = {
        "tail_horizon": 4,
        "action_budget": 8,
        "policy_call_cap": 8,
        "recovery_actor_seed": 7,
        "operator": "fresh_h4",
        "d4_signatures": {
            "detection": {"complete_signature_hash": "a" * 64},
            "pre_tail": {"complete_signature_hash": "b" * 64},
            "final": {"complete_signature_hash": "c" * 64},
        },
    }
    rows = []
    for tail in (4, 8, 16):
        for action in (8, 16, 32):
            for seed in SEEDS:
                for operator in OPERATORS:
                    row = dict(base, tail_horizon=tail, action_budget=action,
                               recovery_actor_seed=seed, operator=operator)
                    row["d4_signatures"] = {
                        part: dict(value)
                        for part, value in base["d4_signatures"].items()
                    }
                    rows.append(row)
    issues = []
    report = _check_d4_groups({(TASKS[0], "e", 2): rows}, issues)
    assert report["complete_groups"] == 1
    assert issues == []
    rows[0]["d4_signatures"]["detection"]["complete_signature_hash"] = "d" * 64
    issues = []
    _check_d4_groups({(TASKS[0], "e", 2): rows}, issues)
    assert any("signature differs" in reason for reason in issues)


def test_acceptance_output_must_not_be_inside_gpu_base(tmp_path):
    report = {"status": "INCOMPLETE"}
    with pytest.raises(ValueError):
        write_report(report, tmp_path / "acceptance.json", tmp_path)
