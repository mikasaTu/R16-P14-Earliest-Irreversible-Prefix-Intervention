#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


IMMUTABLE_PARENT = "6eae66d23313cc97231249bfa1c40dc1767ea727"
IMMUTABLE_PARENT_TREE = "ddcedd60f4f4e2878f8a4400d65e9e888f00cdd1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(root: Path) -> int:
    manifest = root / "artifacts/stage2c/SHA256SUMS"
    entries = {}
    for raw in manifest.read_text().splitlines():
        digest, relative = raw.split("  ", 1)
        entries[relative] = digest
    expected = {
        str(path.relative_to(root / "artifacts/stage2c"))
        for path in (root / "artifacts/stage2c").rglob("*")
        if path.is_file() and path != manifest
    }
    assert set(entries) == expected, "checksum manifest does not cover the exact Stage-2C artifact set"
    for relative, digest in entries.items():
        assert sha256(root / "artifacts/stage2c" / relative) == digest, relative
    return len(entries)


def verify(root: Path, *, skip_checksums: bool) -> dict[str, Any]:
    artifact = root / "artifacts/stage2c"
    freeze = read_json(root / "experiments/r16_p14_stage2c/source_freeze/manifest.json")
    assert freeze["immutable_parent"]["head"] == IMMUTABLE_PARENT
    assert freeze["immutable_parent"]["tree"] == IMMUTABLE_PARENT_TREE

    events = read_json(artifact / "actor_events/summary.json")
    assert events["expected_attempts"] == 240
    assert events["completed_attempts"] == 240
    assert events["admitted_events"] == 238
    assert events["unstable_replay_exclusions"] == 0
    assert line_count(artifact / "actor_events/events.jsonl") == 238

    qualification = read_json(artifact / "task_qualification/summary.json")
    assert qualification["completed_attempts"] == 891
    assert qualification["missing_attempts"] == 0
    assert all(cell["error_count"] == 0 for cell in qualification["cells"])
    assert not qualification["target_shift_two_severities"]
    assert not qualification["second_failure_family_pass"]
    assert qualification["continue_all_experiments_after_gate_failure"]

    smoke = read_json(artifact / "integration_smoke/summary.json")
    assert smoke == {
        "all_complete": True,
        "complete_event_files": 2,
        "error_count": 0,
        "event_files": 2,
        "matched_rows": 16,
        "recovery_rows": 48,
        "schema_version": 1,
        "scope": "integration_smoke",
    }

    formal = read_json(artifact / "formal_matrix/summary.json")
    assert formal["all_complete"]
    assert formal["event_files"] == formal["complete_event_files"] == 96
    assert formal["matched_rows"] == 5760
    assert formal["recovery_rows"] == 17280
    assert formal["error_count"] == 0
    assert line_count(artifact / "formal_matrix/matched_prefix_rows.jsonl") == 5760
    assert line_count(artifact / "formal_matrix/recovery_operator_rows.jsonl") == 17280

    aggregate = read_json(artifact / "aggregate_summary.json")
    assert aggregate["atlas_rows"] == 1440
    assert aggregate["boundary_rows"] == 96
    decision = read_json(artifact / "decision.json")
    assert decision["stage1b_universal_hypothesis"] == "KILLED_IMMUTABLE"
    assert decision["stage2b_status"] == "BLOCKED_UPSTREAM"
    assert decision["second_failure_family"] == "BLOCKED"
    assert decision["track_a_operator_relative_prefix_reuse"] == "INCONCLUSIVE"
    assert decision["track_b_crossfit_replanability"] == "INCONCLUSIVE"
    assert decision["local_repair"] == "RETIRED_NO_SIGNAL"
    assert decision["operator_router"] == "RETIRED_NO_SIGNAL"
    assert decision["overall"] == "BLOCKED_BY_SECOND_FAILURE_FAMILY"
    assert decision["accepted"] is False
    assert decision["novelty"] == "N2_ORACLE_PROTOCOL_BOUNDARY_ONLY"
    assert decision["all_experiments_continued_after_failed_gates"]

    complete = read_json(artifact / "STAGE2C_COMPLETE.json")
    assert complete["status"] == "completed_all_planned_attempts"
    assert complete["formal_matrix"] == formal
    assert complete["decision"] == decision
    mechanism = read_json(artifact / "mechanism_reverse_audit.json")
    assert mechanism["new_idea_generated"] is False

    oversized = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts and path.stat().st_size >= 100_000_000
    ]
    assert not oversized, f"GitHub-ineligible files: {oversized}"
    checksummed = None if skip_checksums else verify_checksums(root)
    return {
        "status": "PASS",
        "actor_events": 238,
        "qualification_attempts": 891,
        "formal_event_files": 96,
        "matched_rows": 5760,
        "recovery_rows": 17280,
        "checksum_entries": checksummed,
        "decision": decision["overall"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve(), skip_checksums=args.skip_checksums), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
