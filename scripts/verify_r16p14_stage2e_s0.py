#!/usr/bin/env python3
"""One-command CPU verifier for the Stage-2E/S0 evidence package.

The verifier never constructs an environment or imports a model.  It checks
the frozen input receipt, machine summaries, required fail-closed decisions,
checksum coverage, and that immutable predecessor directories were untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


EXPECTED_A1 = {
    "action_disagreement": (10, 144, 48),
    "fixed_delay_1": (8, 144, 48),
    "fixed_delay_2": (13, 144, 48),
    "fixed_delay_4": (15, 144, 48),
    "fixed_delay_8": (16, 144, 48),
    "immediate_fresh_h16": (23, 144, 48),
    "k_last_observed_safe": (3, 72, 24),
    "k_last_recoverable": (20, 54, 18),
    "velocity_phase": (16, 144, 48),
}
REQUIRED = (
    "experiments/r16_p14_stage2e/PREREG_S0.md",
    "experiments/r16_p14_stage2e/PREREG_S0.md",
    "experiments/r16_p14_stage2e/metric_contract.md",
    "experiments/r16_p14_stage2e/commands.sh",
    "experiments/r16_p14_stage2e/INSTRUMENTATION_S1.md",
    "experiments/r16_p14_stage2e/decision.json",
    "experiments/r16_p14_stage2e/reports/REPORT_S0.md",
    "artifacts/stage2e/source_freeze/input_receipt.json",
    "artifacts/stage2e/source_freeze/analysis_manifest.json",
    "artifacts/stage2e/s0/common_support/table.csv",
    "artifacts/stage2e/s0/common_support/a1_published_evaluation.csv",
    "artifacts/stage2e/s0/common_support/a5_event_rows.csv",
    "artifacts/stage2e/s0/common_support/a5_table.csv",
    "artifacts/stage2e/s0/common_support/summary.json",
    "artifacts/stage2e/s0/headroom/table.csv",
    "artifacts/stage2e/s0/headroom/selection_receipt.json",
    "artifacts/stage2e/s0/headroom/summary.json",
    "artifacts/stage2e/s0/operator_ladder/per_event.jsonl",
    "artifacts/stage2e/s0/operator_ladder/summary.json",
    "artifacts/stage2e/s0/summary.json",
    "artifacts/stage2e/SHA256SUMS",
    "scripts/run_r16p14_stage2e_s0.py",
    "scripts/verify_r16p14_stage2e_s0.py",
)
OLD_DIRS = (
    "artifacts/stage2c", "artifacts/stage2d", "experiments/r16_p14_stage2a",
    "experiments/r16_p14_stage2b", "experiments/r16_p14_stage2c",
    "experiments/r16_p14_stage2d", "docs/feishu",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value, path="root"):
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"non-finite {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite(item, f"{path}[{index}]")


def verify_checksums(root: Path) -> None:
    path = root / "artifacts/stage2e/SHA256SUMS"
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        if rel in seen:
            raise AssertionError(f"duplicate checksum path: {rel}")
        seen.add(rel)
        target = root / rel
        if not target.is_file():
            raise AssertionError(f"checksum target missing: {rel}")
        if target == path:
            raise AssertionError("SHA256SUMS must document its own exclusion")
        actual = sha256(target)
        if actual != digest:
            raise AssertionError(f"checksum mismatch {rel}: {actual} != {digest}")
    expected = set()
    for base in (root / "experiments/r16_p14_stage2e", root / "artifacts/stage2e"):
        if base.exists():
            for item in base.rglob("*"):
                if item.is_file() and item != path and item.suffix not in {".pyc"} and "__pycache__" not in item.parts:
                    expected.add(item.relative_to(root).as_posix())
    for rel in ("scripts/run_r16p14_stage2e_s0.py", "scripts/verify_r16p14_stage2e_s0.py"):
        expected.add(rel)
    missing = sorted(expected - seen)
    if missing:
        raise AssertionError(f"checksum coverage missing: {missing[:10]}")


def verify_old_dirs(root: Path) -> None:
    parent = "edebdfc64576129d994535dacb76de930f493c8d"
    for old in OLD_DIRS:
        result = subprocess.run(["git", "diff", "--quiet", parent, "--", old], cwd=root)
        if result.returncode != 0:
            raise AssertionError(f"immutable predecessor changed: {old}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        missing = [rel for rel in REQUIRED if not (root / rel).is_file()]
        if missing:
            raise AssertionError(f"required files missing: {missing}")
        prereg = (root / "experiments/r16_p14_stage2e/PREREG_S0.md").read_text(encoding="utf-8")
        if "diagnostic_only" not in prereg or "PAI planned/submitted jobs" not in prereg:
            raise AssertionError("frozen prereg metadata missing")
        a = load_json(root / "artifacts/stage2e/s0/common_support/summary.json")
        b = load_json(root / "artifacts/stage2e/s0/headroom/summary.json")
        c = load_json(root / "artifacts/stage2e/s0/operator_ladder/summary.json")
        decision = load_json(root / "experiments/r16_p14_stage2e/decision.json")
        receipt = load_json(root / "artifacts/stage2e/source_freeze/input_receipt.json")
        selection = load_json(root / "artifacts/stage2e/s0/headroom/selection_receipt.json")
        for obj in (a, b, c, decision, receipt, selection):
            finite(obj)
        for obj in (a, b, c, decision):
            if obj.get("diagnostic_only") is not True or obj.get("formal_positive_evidence_allowed") is not False or obj.get("new_idea_generated") is not False:
                raise AssertionError("diagnostic/formal/new-idea flags are not fail-closed")
        if decision.get("accepted") is not False or decision.get("novelty") != "N2_ORACLE_PROTOCOL_BOUNDARY_ONLY":
            raise AssertionError("decision boundary changed")
        if decision.get("s1_started") is not False or decision.get("planned_pai_jobs") != 0 or decision.get("submitted_pai_jobs") != 0:
            raise AssertionError("S1/PAI boundary changed")
        # A1 exact machine-table reproduction.
        a1 = {row["baseline"]: row for row in a.get("A1_rows", [])}
        if set(a1) != set(EXPECTED_A1):
            raise AssertionError("A1 method set mismatch")
        for method, target in EXPECTED_A1.items():
            got = (int(a1[method]["numerator"]), int(a1[method]["denominator"]), int(a1[method]["support_events"]))
            if got != target or a1[method].get("exact_match") is not True:
                raise AssertionError(f"A1 mismatch {method}: {got} != {target}")
        if a.get("G0_1") != "FULL_SUPPORT_ADVANTAGE" or a.get("A1_exact_reproduction") is not True:
            raise AssertionError("A gate unexpectedly positive/blocked")
        # Expected preregistered cluster estimates and CI direction.
        deltas = a["cohorts"]
        expected_delta = {"all_contaminated": 11.0 / 144.0, "replay_valid_subset": 4.0 / 63.0}
        for cohort, target in expected_delta.items():
            got = float(deltas[cohort]["delta"]["full_delta"])
            if abs(got - target) > 1e-12:
                raise AssertionError(f"A full delta mismatch {cohort}: {got} != {target}")
        if b.get("G0_2") != "BUDGET_SCAN_FIRST" or b.get("b3", {}).get("S1_FIRST_ACTION") != "BUDGET_SCAN_FIRST":
            raise AssertionError("B did not fail closed on absent configured budget grid")
        if selection.get("evaluation_rows_opened_before_receipt") is not False or selection.get("stage2d_evaluation_read_for_selection") is not False or selection.get("reserve_ids_read") is not False:
            raise AssertionError("selection receipt violates open-deny contract")
        for cohort, target in (("all_contaminated", (21, 29, 38, 39, 21, 11 / 21)), ("replay_valid_subset", (8, 12, 12, 12, 8, 5 / 8))):
            item = c["cohorts"][cohort]
            got = (item["u1_defined_events"], item["u2_defined_events"], item["u3_defined_events"], item["u4_defined_events"], item["both_defined_events"], item["absolute_shift_ge_1_fraction"])
            if got[:5] != target[:5] or abs(float(got[5]) - target[5]) > 1e-12:
                raise AssertionError(f"C support mismatch {cohort}: {got}")
            if item["monotonicity_violation_count"] != 0 or item["u4_exact_crosscheck_mismatch_count"] != 0:
                raise AssertionError(f"C integrity mismatch {cohort}")
        if c.get("G0_3") != "EXPAND_OPERATOR_FAMILY_FIRST":
            raise AssertionError("C did not fail closed on valid U1 support")
        report = (root / "experiments/r16_p14_stage2e/reports/REPORT_S0.md").read_text(encoding="utf-8").lower()
        for forbidden in ("physical irreversibility point", "universally irreversible prefix", "validated vla method", "accepted idea", "n3", "n4"):
            if forbidden in report:
                raise AssertionError(f"forbidden wording in report: {forbidden}")
        # The deferred input receipt must document the order, and evaluation
        # records may only be descriptive after the selection receipt.
        if receipt.get("evaluation_rows_opened_before_receipt") is True:
            raise AssertionError("evaluation opened before selection receipt")
        verify_old_dirs(root)
        verify_checksums(root)
    except Exception as exc:
        print(f"VERIFY_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("VERIFY_PASS: Stage-2E/S0 CPU offline contract, numerical outputs, checksums, and old-tree immutability")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
