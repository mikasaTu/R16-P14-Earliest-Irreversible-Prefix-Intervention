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

    def test_four_preregistered_cluster_bootstrap_intervals_are_exact(self):
        summary = json.loads((ROOT / "artifacts/stage2e/s0/common_support/summary.json").read_text(encoding="utf-8"))
        expected = {
            "all_contaminated": {
                "restricted_bootstrap": (-0.069444444444444, 0.236111111111111),
                "full_bootstrap": (0.006944444444444, 0.145833333333333),
            },
            "replay_valid_subset": {
                "restricted_bootstrap": (-0.194444444444444, 0.361111111111111),
                "full_bootstrap": (-0.007936507936508, 0.142857142857143),
            },
        }
        for cohort, intervals in expected.items():
            delta = summary["cohorts"][cohort]["delta"]
            for key, bounds in intervals.items():
                self.assertEqual(len(delta[key]["ci95"]), 2)
                for actual, target in zip(delta[key]["ci95"], bounds):
                    self.assertAlmostEqual(actual, target, delta=1e-12)
                self.assertEqual(delta[key]["replicates"], 10000)
                self.assertEqual(delta[key]["seed"], 216214)
                self.assertEqual(delta[key]["cluster_aggregation"], "event_actor_mean_then_equal_cluster_mean")

    def test_decision_is_stage2e_s0_only_and_has_no_combined_overall(self):
        decision = json.loads((ROOT / "experiments/r16_p14_stage2e/decision.json").read_text(encoding="utf-8"))
        expected = {
            "stage1b_universal_hypothesis", "stage2c_status", "stage2d_status",
            "A_status", "A_G0_1", "B_status", "B_G0_2", "C_status", "C_G0_3",
            "diagnostic_only", "formal_positive_evidence_allowed", "new_idea_generated",
            "planned_pai_jobs", "submitted_pai_jobs", "s1_started", "accepted", "novelty",
        }
        self.assertEqual(set(decision), expected)
        self.assertNotIn("overall", decision)
        self.assertNotIn("combined_stage2e_s0_decision", decision)
        for key in ("h1_observed_safe_window", "h2_cached_prefix_content", "h3_event_aligned_handoff", "h4_nontrivial_selection", "cached_prefix_claim"):
            self.assertNotIn(key, decision)

    def test_b1_causal_contribution_is_not_identifiable(self):
        headroom = json.loads((ROOT / "artifacts/stage2e/s0/headroom/summary.json").read_text(encoding="utf-8"))
        axis = json.loads((ROOT / "artifacts/stage2e/s0/headroom/axis_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(headroom["budget_severity_runtime_causal_contribution"], "NOT_IDENTIFIABLE/BLOCKED_BY_NO_COMMON_SUPPORT")
        self.assertEqual(axis["causal_contribution"], "NOT_IDENTIFIABLE_BUDGET_SEVERITY_RUNTIME_CO_VARY")
        self.assertEqual(axis["only_quantifiable_sensitivity"], "stage2c_all_to_valid_invalid_event_exclusion_subset_sensitivity")

    def test_report_uses_only_allowed_stage2e_language(self):
        report = (ROOT / "experiments/r16_p14_stage2e/reports/REPORT_S0.md").read_text(encoding="utf-8").lower()
        for forbidden in ("physical irreversibility point", "universally irreversible prefix", "validated vla method", "accepted idea", "n3", "n4"):
            self.assertNotIn(forbidden, report)


if __name__ == "__main__":
    unittest.main()
