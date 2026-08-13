from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from .io_utils import atomic_write_json, atomic_write_text


def as_bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"not a canonical bool: {value!r}")
    return value == "True"


def reproduce(source: Path) -> dict[str, object]:
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "candidate_id",
        "M0_safe_success",
        "M1_safe_success",
        "M2_safe_success",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"source columns do not satisfy contract: {required}")
    parsed = [
        {
            "candidate_id": row["candidate_id"],
            "M0_safe_success": as_bool(row["M0_safe_success"]),
            "M1_safe_success": as_bool(row["M1_safe_success"]),
            "M2_safe_success": as_bool(row["M2_safe_success"]),
        }
        for row in rows
    ]
    m0_safe = {row["candidate_id"] for row in parsed if row["M0_safe_success"]}
    m1_safe = {row["candidate_id"] for row in parsed if row["M1_safe_success"]}
    m2_safe = {row["candidate_id"] for row in parsed if row["M2_safe_success"]}
    all_ids = {row["candidate_id"] for row in parsed}
    m0_fail = all_ids - m0_safe
    m1_fail = all_ids - m1_safe
    m2_only = m2_safe - (m0_safe | m1_safe)
    facts = {
        "row_count": len(parsed),
        "M0_safe_success_count": len(m0_safe),
        "M1_safe_success_count": len(m1_safe),
        "M2_safe_success_count": len(m2_safe),
        "M0_failure_ids": sorted(m0_fail),
        "M1_failure_ids": sorted(m1_fail),
        "M0_M1_failure_samples_differ": m0_fail != m1_fail,
        "M2_unique_safe_when_M0_M1_fail_count": len(m2_only),
        "M0_M1_union_safe_count": len(m0_safe | m1_safe),
    }
    expected = {
        "row_count": 6,
        "M0_safe_success_count": 5,
        "M1_safe_success_count": 5,
        "M2_safe_success_count": 2,
        "M0_M1_failure_samples_differ": True,
        "M2_unique_safe_when_M0_M1_fail_count": 0,
        "M0_M1_union_safe_count": 6,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": facts[key]}
        for key, expected_value in expected.items()
        if facts[key] != expected_value
    }
    return {
        "schema_version": 1,
        "status": "PASS" if not mismatches else "BLOCKED_BY_SOURCE_MISMATCH",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "facts": facts,
        "expected": expected,
        "mismatches": mismatches,
        "rows": parsed,
        "evidence_label": (
            "This is calibration-only n=6 hypothesis-generating evidence, "
            "not algorithm performance."
        ),
    }


def render(result: dict[str, object]) -> str:
    facts = result["facts"]
    return "\n".join(
        [
            "# Current clue reproduction",
            "",
            f"Status: **{result['status']}**.",
            "",
            "The six source rows were parsed mechanically, without a new interpretation:",
            "",
            f"- M0 safe success: `{facts['M0_safe_success_count']}/6`.",
            f"- M1 safe success: `{facts['M1_safe_success_count']}/6`.",
            f"- M2 safe success: `{facts['M2_safe_success_count']}/6`.",
            f"- M0 failure IDs: `{facts['M0_failure_ids']}`.",
            f"- M1 failure IDs: `{facts['M1_failure_ids']}`.",
            f"- M0 and M1 failure samples differ: `{facts['M0_M1_failure_samples_differ']}`.",
            "- M2-only safe successes when both M0 and M1 fail: "
            f"`{facts['M2_unique_safe_when_M0_M1_fail_count']}`.",
            f"- M0/M1 per-sample safe union: `{facts['M0_M1_union_safe_count']}/6`.",
            "",
            "**This is calibration-only n=6 hypothesis-generating evidence, not algorithm performance.**",
            "",
            f"Source SHA256: `{result['source_sha256']}`.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = reproduce(args.source.resolve())
    atomic_write_json(args.output_dir / "current_clue_reproduction.json", result)
    atomic_write_text(args.output_dir / "current_clue_reproduction.md", render(result))
    if result["status"] != "PASS":
        raise SystemExit("BLOCKED_BY_SOURCE_MISMATCH")
    print("CURRENT_CLUE_REPRODUCTION_OK n=6 M0=5 M1=5 M2=2 union=6")


if __name__ == "__main__":
    main()
