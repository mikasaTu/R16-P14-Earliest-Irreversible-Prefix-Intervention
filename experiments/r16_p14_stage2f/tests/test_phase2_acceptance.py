from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s1 import accept_phase2 as acceptance  # noqa: E402


TASK_BOWL = "put_the_bowl_on_the_plate"
TASK_CREAM = "put_the_cream_cheese_in_the_bowl"
BUDGET = {"tail_horizon": 4, "action_budget": 8, "policy_call_cap": 8}
SOURCE_KEY = (TASK_BOWL, "event-316", "316", "evaluation", 7)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


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
        "diagnostic_continuation_doc_sha256": acceptance.DIAGNOSTIC_DOC_SHA256,
        "protocol_sha256": acceptance.DIAGNOSTIC_DOC_SHA256,
        "selected_budget": dict(BUDGET),
    }
    path = root / acceptance.DIAGNOSTIC_RECEIPT_REL
    _write(path, value)
    return BUDGET, hashlib.sha256(path.read_bytes()).hexdigest()


def _source(anchor: int = 100) -> dict[tuple[str, str, str, str, int], dict[str, object]]:
    return {
        SOURCE_KEY: {
            "task": TASK_BOWL,
            "event_instance_id": "event-316",
            "init_state_id": "316",
            "split": "evaluation",
            "generator_actor_seed": 7,
            "anchor_global_step": anchor,
        }
    }


def _structural(anchor: int = 316, prefix: int = 8) -> dict[str, object]:
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
        "is_reference": False,
        "status": "BLOCKED",
        "error_type": "PrefixOutsideTaskHorizon",
        "anchor_global_step": anchor,
        "diagnostic_continuation": True,
        "diagnostic_selection_receipt_sha256": "a" * 64,
    }


def _trace_row(path: Path) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for control_step in (0,):
        records.append(
            {
                "event_instance_id": "event-trace",
                "operator": "fresh_h4",
                "prefix_k": 2,
                "actor_seed": 17,
                "contact_stream_available": True,
                "control_step": control_step,
                "step_kind": "action_step",
                "raw_contact_pairs": [
                    {
                        "geom1_name": "b",
                        "geom2_name": "a",
                        "normalized_pair": ["a", "b"],
                    }
                ],
                "normalized_contact_pairs": [["a", "b"]],
                "contact_pairs": [["a", "b"]],
                "contact_count": 1,
            }
        )
        for substep in range(1, 26):
            records.append(
                {
                    "event_instance_id": "event-trace",
                    "operator": "fresh_h4",
                    "prefix_k": 2,
                    "actor_seed": 17,
                    "contact_stream_available": True,
                    "control_step": control_step,
                    "step_kind": "physics_step",
                    "substep": substep,
                    "raw_contact_pairs": [
                        {
                            "geom1_name": "b",
                            "geom2_name": "a",
                            "normalized_pair": ["a", "b"],
                        }
                    ],
                    "normalized_contact_pairs": [["a", "b"]],
                    "contact_pairs": [["a", "b"]],
                    "contact_count": 1,
                }
            )
    raw = b"".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        for record in records
    )
    with gzip.open(path, "wb") as handle:
        handle.write(raw)
    return {
        "task": TASK_BOWL,
        "event_instance_id": "event-trace",
        "init_state_id": "316",
        "split": "evaluation",
        "generator_actor_seed": 7,
        "recovery_actor_seed": 17,
        "operator": "fresh_h4",
        "prefix_k": 2,
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "trace_content_sha256": hashlib.sha256(raw).hexdigest(),
        "trace_records": len(records),
        "labels": {"record_count": len(records), "label_complete": True},
    }


def _d4_rows(prefix: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prefixes = (prefix,) if prefix is not None else acceptance.PHASE2_PREFIXES
    for current_prefix in prefixes:
      for operator_index, operator in enumerate(acceptance.OPERATORS):
        for seed_index, seed in enumerate(acceptance.SEEDS):
            rows.append(
                {
                    "task": TASK_BOWL,
                    "event_instance_id": "event-316",
                    "init_state_id": "316",
                    "split": "evaluation",
                    "generator_actor_seed": 7,
                    "recovery_actor_seed": seed,
                    "operator": operator,
                    "prefix_k": current_prefix,
                    "tail_horizon": 4,
                    "action_budget": 8,
                    "policy_call_cap": 8,
                    "status": "COMPLETE",
                    "is_reference": False,
                    "job_id": f"job-{operator_index}-{seed_index}",
                    "pid": 1000 + operator_index * 10 + seed_index,
                    "d4_signatures": {
                        "detection": {"complete_signature_hash": "a" * 64},
                        "pre_tail": {"complete_signature_hash": "b" * 64},
                        "final": {"complete_signature_hash": "c" * 64},
                    },
                }
            )
    return rows


def test_phase2_expected_key_count_is_96_core_plus_12_reference():
    core, reference = acceptance._expected_keys(_source(), BUDGET)
    assert len(core) == 96
    assert len(reference) == 12
    assert len(core | reference) == 108


def test_admission_is_open_deny_before_evaluation_source_or_shards(tmp_path, monkeypatch):
    def deny(_root: Path):
        raise RuntimeError("proof is absent")

    monkeypatch.setattr(
        acceptance.matrix, "selected_diagnostic_budget", deny, raising=False
    )
    monkeypatch.setattr(
        acceptance.matrix, "diagnostic_selected_budget", None, raising=False
    )
    malformed_eval = (
        tmp_path
        / "phase0b"
        / "sealed_evaluation"
        / TASK_BOWL
        / "must-not-read.json"
    )
    malformed_eval.parent.mkdir(parents=True)
    malformed_eval.write_text("{", encoding="utf-8")
    malformed_shard = tmp_path / "phase2" / "shards" / TASK_BOWL / "must-not-read.json"
    malformed_shard.parent.mkdir(parents=True)
    malformed_shard.write_text("{", encoding="utf-8")

    result = acceptance.evaluate_phase2(tmp_path, trace_workers=1)

    assert result["status"] == "BLOCKED"
    assert result["evaluation_read"] is False
    assert any("OPEN_DENY" in item for item in result["blocking_reasons"])
    assert not any("must-not-read" in item for item in result["blocking_reasons"])


def test_admission_binds_real_receipt_sha_and_protocol(tmp_path, monkeypatch):
    _budget, receipt_sha = _receipt(tmp_path)
    monkeypatch.setattr(
        acceptance.matrix,
        "selected_diagnostic_budget",
        lambda _root: {
            "selected_budget": dict(BUDGET),
            "selection_receipt_sha256": receipt_sha,
        },
        raising=False,
    )
    reasons: list[str] = []

    selected, observed_sha, metadata = acceptance._admit_diagnostic_budget(
        tmp_path, reasons
    )

    assert reasons == []
    assert selected == BUDGET
    assert observed_sha == receipt_sha
    assert metadata["receipt_sha256"] == receipt_sha


def test_structural_marker_requires_source_proven_horizon_and_never_becomes_negative():
    reasons: list[str] = []
    assert acceptance._structural_valid(
        _structural(anchor=316),
        {
            "task": TASK_BOWL,
            "anchor_global_step": 316,
        },
        reasons,
        "outside",
    )
    assert reasons == []

    feasible_reasons: list[str] = []
    assert not acceptance._structural_valid(
        _structural(anchor=300),
        {
            "task": TASK_BOWL,
            "anchor_global_step": 316,
        },
        feasible_reasons,
        "feasible",
    )
    assert any("feasible" in item for item in feasible_reasons)

    unknown_reasons: list[str] = []
    unknown = _structural(anchor=316)
    unknown["error_type"] = "RuntimeError"
    assert not acceptance._structural_valid(
        unknown,
        {"task": TASK_BOWL, "anchor_global_step": 316},
        unknown_reasons,
        "unknown",
    )
    assert any("structural marker" in item for item in unknown_reasons)


def test_reused_calibration_requires_phase1_sha_and_only_allowed_appends(tmp_path):
    prior = {
        "task": TASK_BOWL,
        "event_instance_id": "event",
        "init_state_id": "316",
        "split": "calibration",
        "generator_actor_seed": 7,
        "recovery_actor_seed": 17,
        "operator": "fresh_h4",
        "prefix_k": 2,
        "tail_horizon": 4,
        "action_budget": 8,
        "policy_call_cap": 8,
        "is_reference": False,
        "status": "COMPLETE",
        "diagnostic_only": True,
        "safe_success": False,
        "source_commit": acceptance.PHASE1_SOURCE_COMMIT,
        "runtime_receipt_sha256": "c" * 64,
        "trace_path": "/x/phase1/contact_topology/put_the_bowl_on_the_plate/x.jsonl.gz",
    }
    prior_path = tmp_path / "phase1" / "shards" / TASK_BOWL / "old.json"
    _write(prior_path, prior)
    current = {
        **prior,
        "reused_calibration_shard": str(prior_path),
        "reused_calibration_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        "diagnostic_continuation": True,
        "diagnostic_selection_receipt_sha256": "a" * 64,
    }
    reasons: list[str] = []
    assert acceptance._validate_reused_calibration(
        tmp_path,
        current,
        reasons,
        "row",
        selection_sha="a" * 64,
        phase1_source_commits={acceptance.PHASE1_SOURCE_COMMIT},
    )
    assert reasons == []

    changed = dict(current)
    changed["safe_success"] = True
    changed_reasons: list[str] = []
    assert not acceptance._validate_reused_calibration(
        tmp_path,
        changed,
        changed_reasons,
        "changed",
        selection_sha="a" * 64,
        phase1_source_commits={acceptance.PHASE1_SOURCE_COMMIT},
    )
    assert any("outside allowed" in item for item in changed_reasons)


def test_full_trace_validation_worker_one_and_two_are_identical(tmp_path):
    path = tmp_path / "trace.jsonl.gz"
    row = _trace_row(path)

    first_reasons, first = acceptance._validate_trace_files({path: row}, "full", 1)
    second_reasons, second = acceptance._validate_trace_files({path: row}, "full", 2)

    assert first_reasons == []
    assert second_reasons == []
    assert first["files"] == second["files"] == 1
    assert first["records"] == second["records"] == 26
    assert first["action_records"] == second["action_records"] == 1
    assert first["physics_records"] == second["physics_records"] == 25
    assert first["physics_per_control_counts"] == second["physics_per_control_counts"]


def test_d4_checks_12_process_isolated_rows_and_shared_state_signatures():
    source = _source(anchor=100)
    reasons: list[str] = []
    report = acceptance._check_d4_groups(source, _d4_rows(), reasons)

    assert reasons == []
    assert report["groups"] == len(acceptance.PHASE2_PREFIXES)
    assert report["complete_groups"] == len(acceptance.PHASE2_PREFIXES)
    assert report["structural_groups"] == 0

    broken = _d4_rows()
    broken[1]["d4_signatures"]["pre_tail"]["complete_signature_hash"] = "d" * 64
    broken_reasons: list[str] = []
    broken_report = acceptance._check_d4_groups(source, broken, broken_reasons)
    assert broken_report["mismatch_groups"]
    assert any("state differs" in item for item in broken_reasons)

    duplicate = _d4_rows()
    duplicate[1]["pid"] = duplicate[0]["pid"]
    duplicate_reasons: list[str] = []
    acceptance._check_d4_groups(source, duplicate, duplicate_reasons)
    assert any("process isolation" in item for item in duplicate_reasons)


def test_d4_skips_source_proven_true_horizon_group():
    source = _source(anchor=316)
    rows = _d4_rows()
    for row in rows:
        if row["prefix_k"] >= 6:
            row.update(_structural(anchor=316, prefix=int(row["prefix_k"])))
    reasons: list[str] = []
    report = acceptance._check_d4_groups(source, rows, reasons)

    assert reasons == []
    assert report["structural_groups"] == 6
    assert report["complete_groups"] == 2
