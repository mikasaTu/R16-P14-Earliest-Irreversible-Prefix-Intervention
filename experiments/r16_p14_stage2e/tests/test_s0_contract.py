"""Small CPU-only contract tests for the Stage-2E/S0 analysis helpers."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("stage2e_s0_runner", ROOT / "scripts/run_r16p14_stage2e_s0.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class Stage2ES0ContractTests(unittest.TestCase):
    def test_selection_guard_rejects_evaluation_and_reserve(self):
        with self.assertRaises(MODULE.ContractError):
            MODULE.selection_guard([{"split": "evaluation", "init_state_id": 40}])
        with self.assertRaises(MODULE.ContractError):
            MODULE.selection_guard([{"split": "calibration", "init_state_id": 80}])

    def test_selection_guard_accepts_calibration_only(self):
        MODULE.selection_guard([{"split": "calibration", "init_state_id": 10}, {"split": "calibration", "init_state_id": 39}])

    def test_cluster_bootstrap_is_deterministic_and_cluster_first(self):
        rows = [
            {"task": "a", "init_state_id": 1, "value": 0.0},
            {"task": "a", "init_state_id": 1, "value": 1.0},
            {"task": "b", "init_state_id": 2, "value": 1.0},
        ]
        left = MODULE.cluster_bootstrap(rows, replicates=250, seed=216214)
        right = MODULE.cluster_bootstrap(rows, replicates=250, seed=216214)
        self.assertEqual(left, right)
        self.assertEqual(left["clusters"], 2)
        self.assertAlmostEqual(left["estimate"], 0.75)

    def test_nested_operator_families_have_no_false_violation(self):
        rows = []
        for k in MODULE.PREFIXES:
            for operator in MODULE.OPERATORS:
                for actor in MODULE.ACTOR_SEEDS:
                    # One common success cell at d, then all-zero cells.  The
                    # family nesting must never report a reverse movement.
                    success = k == MODULE.DETECTION_PREFIX and operator == "fresh_h4"
                    rows.append({
                        "event_instance_id": "synthetic", "prefix_k": k, "operator": operator,
                        "recovery_actor_seed": actor, "safe_success": success,
                    })
        result = MODULE._ladder_for_event(rows)
        self.assertTrue(result["complete"])
        self.assertEqual(result["monotonicity_violations"], [])
        self.assertEqual(result["k_last_recoverable"]["U1"], 2)
        self.assertEqual(result["k_last_recoverable"]["U4"], 2)

    def test_a5_contract_names_raw_prefix_field(self):
        source = (ROOT / "scripts/run_r16p14_stage2e_s0.py").read_text(encoding="utf-8")
        self.assertIn('"any": "raw prefix_cause_violation"', source)
        self.assertIn('"fraction": "raw prefix_cause_violation"', source)
        self.assertIn('"type_set": "raw cause_violation_type"', source)

    def test_no_positive_decision_after_upstream_or_budget_failure(self):
        decision = json.loads((ROOT / "experiments/r16_p14_stage2e/decision.json").read_text(encoding="utf-8"))
        headroom = json.loads((ROOT / "artifacts/stage2e/s0/headroom/summary.json").read_text(encoding="utf-8"))
        self.assertFalse(decision["accepted"])
        self.assertEqual(headroom["G0_2"], "BUDGET_SCAN_FIRST")
        self.assertFalse(headroom["formal_positive_evidence_allowed"])

    def test_report_uses_only_allowed_stage2e_language(self):
        report = (ROOT / "experiments/r16_p14_stage2e/reports/REPORT_S0.md").read_text(encoding="utf-8").lower()
        for forbidden in ("physical irreversibility point", "universally irreversible prefix", "validated vla method", "accepted idea", "n3", "n4"):
            self.assertNotIn(forbidden, report)


if __name__ == "__main__":
    unittest.main()
