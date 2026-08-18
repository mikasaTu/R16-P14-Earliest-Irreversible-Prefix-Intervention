from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from r16_p14_stage2d import (
    calibration,
    checksums,
    confirmatory,
    events,
    oracle_appendix,
    runtime,
    statistics,
)
from r16_p14_stage2d.contracts import (
    has_zero_to_one,
    maximum_budgets,
    outcome_label_allowed,
    prefix_length,
    required_detection_calls,
    splits_disjoint,
)
from r16_p14_stage2d.freeze_rule import rule_k
from r16_p14_stage2d.io_utils import load_jsonl, sha256_file
from r16_p14_stage2d.qualification import cell_metrics
from r16_p14_stage2d.settings import (
    ARTIFACT_ROOT,
    CALIBRATION_IDS,
    EVALUATION_IDS,
    EXPERIMENT_ROOT,
    INFRASTRUCTURE_IDS,
    RESERVE_IDS,
    TARGET_SHIFT_TASK,
    TASKS,
)


def require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"phase artifact not generated yet: {path}")
    return path


def test_01_every_isolation_branch_has_unique_spawned_process():
    summary = json.loads(require(ARTIFACT_ROOT / "branch_isolation/summary.json").read_text())
    assert summary["checks"]["unique_spawned_process_per_branch"]


def test_02_no_shared_runtime_class_in_stage2d():
    assert "class EventRuntime" not in inspect.getsource(runtime)
    assert 'mp.get_context("spawn")' in (EXPERIMENT_ROOT / "r16_p14_stage2d/fresh_process.py").read_text()
    launcher = (EXPERIMENT_ROOT / "pai/launcher.sh").read_text()
    assert 'cd -- "$SOURCE_ROOT"' in launcher
    assert 'test "$(pwd -P)" = "$SOURCE_ROOT"' in launcher


def test_03_same_action_cached_signatures_are_exact():
    summary = json.loads(require(ARTIFACT_ROOT / "branch_isolation/summary.json").read_text())
    assert summary["checks"]["same_action_signature_100_percent"]


def test_04_actor_inference_has_no_side_effect():
    smoke = json.loads(require(ARTIFACT_ROOT / "test_results/integration_smoke.json").read_text())
    assert all(row["actor_inference_side_effect_free"] for row in smoke["rows"])


def test_05_branch_order_permutation_is_invariant():
    summary = json.loads(require(ARTIFACT_ROOT / "branch_isolation/summary.json").read_text())
    assert summary["checks"]["branch_order_invariant"]


def test_06_three_reconstructions_are_exact():
    summary = json.loads(require(ARTIFACT_ROOT / "branch_isolation/summary.json").read_text())
    assert summary["checks"]["reconstruction_100_percent"]
    assert summary["maximum_state_error"] <= 1e-9


def test_07_replay_error_blocks_event_admission():
    assert "replay_admitted_3_of_3" in inspect.getsource(confirmatory.consolidate)
    assert "and not any(row.get(\"error\")" in inspect.getsource(confirmatory.consolidate)


def test_08_zero_to_one_is_detected():
    assert has_zero_to_one([True, False, True])
    assert not has_zero_to_one([True, True, False, False])


def test_09_global_step_fallback_is_forbidden():
    assert "global_step >=" not in inspect.getsource(events.eligible)
    for event in load_jsonl(require(ARTIFACT_ROOT / "actor_events/events.jsonl")):
        assert not event["global_step_fallback_used"]


def test_10_target_event_requires_future_release():
    for event in load_jsonl(require(ARTIFACT_ROOT / "actor_events/events.jsonl")):
        if event["task"] == TARGET_SHIFT_TASK:
            assert 6 <= event["release_index"] <= 13


def test_11_path_event_requires_future_swept_point():
    for event in load_jsonl(require(ARTIFACT_ROOT / "actor_events/events.jsonl")):
        if event["task"] != TARGET_SHIFT_TASK:
            assert event["predicted_path_indices"]
            assert all(index in (6, 9, 12) for index in event["predicted_path_indices"])


def test_12_manipulated_object_teleport_is_exactly_zero():
    smoke = json.loads(require(ARTIFACT_ROOT / "test_results/integration_smoke.json").read_text())
    assert all(row["perturbation"]["manipulated_qpos_max_error"] == 0 for row in smoke["rows"])


def test_13_injection_contact_blocks_qualification_cell():
    base = {
        "error": None,
        "cause_violation": False,
        "immediate_violation": False,
        "first_violation_offset": None,
        "actor_seed": 7,
        "injection_contact": True,
        "reconstruction": {
            "anchor_state_exact": True,
            "state_history_exact": True,
            "action_history_exact": True,
            "event_chunk_exact": True,
            "actor_inference_side_effect_free": True,
            "max_anchor_state_error": 0.0,
        },
    }
    metrics = cell_metrics([base] * 20, {"parameter_id": "synthetic"})
    assert not metrics["checks"]["injection_contact_rate_zero"]
    assert not metrics["qualified"]


def test_14_qualification_selection_reads_no_method_outcome():
    source = inspect.getsource(cell_metrics)
    assert "safe_success" not in source
    assert "task_success" not in source


def test_15_all_data_splits_are_disjoint():
    assert splits_disjoint(INFRASTRUCTURE_IDS, CALIBRATION_IDS, EVALUATION_IDS, RESERVE_IDS)


def test_16_evaluation_requires_committed_rule_manifest():
    source = inspect.getsource(confirmatory.verify_frozen_rule)
    assert "git" in source and "HEAD:" in source and "checksum mismatch" in source


def test_17_cached_and_fresh_prefix_lengths_match():
    assert prefix_length(2) == 0
    assert prefix_length(6) == 4
    assert prefix_length(16) == 14


def test_18_cached_and_fresh_have_equal_required_pre_k_calls():
    assert required_detection_calls("CACHED_MATCHED") == 1
    assert required_detection_calls("FRESH_MATCHED") == 1


def test_19_primary_tail_horizon_is_four():
    assert maximum_budgets()["tail_horizon"] == 4


def test_20_all_primary_arms_share_maximum_budgets():
    assert maximum_budgets() == {
        "tail_horizon": 4,
        "maximum_action_budget": 18,
        "maximum_policy_call_budget": 2,
    }


def test_21_actual_actor_calls_are_not_padded():
    source = inspect.getsource(runtime.execute_branch)
    assert "while actor_calls" not in source
    assert "dummy" not in source


def test_22_cached_count_is_not_efficiency_metric():
    assert "cached_actions_retained" not in statistics.EFFICIENCY_FIELDS


def test_23_oracle_appendix_requires_primary_lock():
    source = inspect.getsource(oracle_appendix.verify_primary_lock)
    assert "primary manifest must be committed" in source


def test_24_bootstrap_resamples_event_rows():
    source = inspect.getsource(statistics.bootstrap_mean_difference)
    assert "size=(BOOTSTRAP_REPLICATES, n)" in source
    assert "prefix" not in source


def test_25_positive_label_forbidden_after_upstream_failure():
    assert not outcome_label_allowed(False, True)
    assert outcome_label_allowed(True, False)


def test_26_reports_do_not_emit_forbidden_positive_claims():
    report = require(ARTIFACT_ROOT / "REPORT.md").read_text().lower()
    for phrase in ("accepted idea", "validated vla method", "universally irreversible prefix"):
        assert phrase not in report


def test_27_no_demonstration_chunk_enters_formal_evidence():
    for event in load_jsonl(require(ARTIFACT_ROOT / "actor_events/events.jsonl")):
        assert event["source_is_actor_generated_chunk"]
        assert not event["source_is_demonstration_chunk"]


def test_28_shard_resume_does_not_overwrite_completed_evidence():
    source = inspect.getsource(events.run_shard)
    assert "if path.is_file()" in source and "EVENT_RESUME" in source


def test_29_checksum_manifest_covers_every_artifact():
    checksum_path = require(ARTIFACT_ROOT / "SHA256SUMS")
    listed = {line.split("  ", 1)[1] for line in checksum_path.read_text().splitlines() if line}
    expected = {
        str(Path("artifacts/stage2d") / path.relative_to(ARTIFACT_ROOT))
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_file() and path != checksum_path
    }
    assert listed == expected


def test_30_two_task_real_libero_smoke_passes():
    smoke = json.loads(require(ARTIFACT_ROOT / "test_results/integration_smoke.json").read_text())
    assert smoke["status"] == "PASS"
    assert set(smoke["real_libero_tasks"]) == set(TASKS)


def test_31_rule_is_outcome_blind_and_clipped():
    target = {"task": TARGET_SHIFT_TASK, "release_index": 13}
    path = {"task": TASKS[1]}
    assert rule_k(target, {"magnitude_m": 0.04}) == 11
    assert rule_k(path, {"future_index": 6, "clearance_delta_m": 0}) == 4


def test_32_fresh_pool_has_100_unique_valid_states_per_task():
    manifest = json.loads(require(ARTIFACT_ROOT / "init_pool/manifest.json").read_text())
    for task in TASKS:
        assert manifest["tasks"][task]["count"] == 100
        assert manifest["tasks"][task]["unique_state_hashes"] == 100
        assert manifest["tasks"][task]["all_valid"]


def test_33_reserve_ids_are_rejected_by_event_loader():
    with pytest.raises(PermissionError):
        events.load_init_state(TASKS[0], 80)


def test_34_primary_decision_is_immutable_during_oracle_appendix():
    summary = json.loads(require(ARTIFACT_ROOT / "oracle_appendix/summary.json").read_text())
    assert summary["primary_decision_unchanged"]
    assert summary["primary_decision_effect"] == "NONE"


def test_35_immutable_negative_conclusions_remain_exact():
    decision = json.loads(require(ARTIFACT_ROOT / "decision.json").read_text())
    assert decision["stage1b_universal_hypothesis"] == "KILLED_IMMUTABLE"
    assert decision["stage2c_status"] == "BLOCKED_UPSTREAM_IMMUTABLE"
    assert decision["accepted"] is False
    assert decision["novelty"] == "N2_ORACLE_PROTOCOL_BOUNDARY_ONLY"


def test_36_pai_frozen_rule_verification_uses_committed_repository_path():
    source = inspect.getsource(confirmatory.verify_frozen_rule)
    assert 'relative = "artifacts/stage2d/frozen_rule/manifest.json"' in source
    assert "manifest_path.relative_to(PROJECT_ROOT)" not in source


def test_37_pai_primary_lock_verification_uses_committed_repository_path():
    source = inspect.getsource(oracle_appendix.verify_primary_lock)
    assert 'relative = "artifacts/stage2d/statistics/primary_manifest.json"' in source
    assert "path.relative_to(PROJECT_ROOT)" not in source


def test_38_checksum_paths_are_stable_for_external_pai_artifact_root():
    external_file = ARTIFACT_ROOT / "statistics/statistics.json"
    assert checksums.canonical_artifact_path(external_file) == Path(
        "artifacts/stage2d/statistics/statistics.json"
    )
    assert "relative_to(PROJECT_ROOT)" not in inspect.getsource(checksums.main)


def test_39_bootstrap_clusters_collapse_actor_seed_repeats_but_keep_task_and_severity():
    rows = []
    for actor_seed in (7, 17, 29):
        rows.append(
            {
                "task": "task_a",
                "parameter_id": "sev_a",
                "init_state_id": 1,
                "actor_seed": actor_seed,
                "methods": {"left": {"x": 1.0}, "right": {"x": 0.0}},
            }
        )
    rows.append(
        {
            "task": "task_a",
            "parameter_id": "sev_b",
            "init_state_id": 1,
            "actor_seed": 7,
            "methods": {"left": {"x": 2.0}, "right": {"x": 0.0}},
        }
    )
    rows.append(
        {
            "task": "task_b",
            "parameter_id": "sev_a",
            "init_state_id": 1,
            "actor_seed": 7,
            "methods": {"left": {"x": 3.0}, "right": {"x": 0.0}},
        }
    )
    overall_left, _ = statistics.cluster_paired_values(rows, "left", "right", "x")
    severity_left, _ = statistics.cluster_paired_values(
        rows,
        "left",
        "right",
        "x",
        cluster_fields=("task", "parameter_id", "source_rollout"),
    )
    seed_left, _ = statistics.cluster_paired_values(
        [row for row in rows if row["actor_seed"] == 7],
        "left",
        "right",
        "x",
        cluster_fields=("task", "parameter_id", "source_rollout", "actor_seed"),
    )
    assert len(overall_left) == 2  # task_a/init1 and task_b/init1
    assert len(severity_left) == 3  # task + severity + init1
    assert len(seed_left) == 3  # seed strata remain separate, but still clustered


def test_40_binary_tables_retain_event_rows_while_bootstrap_is_clustered():
    rows = [
        {
            "task": "task_a",
            "parameter_id": "sev_a",
            "init_state_id": 1,
            "actor_seed": seed,
            "methods": {
                "left": {"safe_success": True, "cause_violation": False, "task_success": True},
                "right": {"safe_success": False, "cause_violation": False, "task_success": False},
            },
        }
        for seed in (7, 17, 29)
    ]
    comparison = statistics.comparison(rows, "left", "right")
    assert comparison["differences"]["safe_success"]["n"] == 1
    assert comparison["binary_tables"]["safe_success"] == {"method_1__baseline_0": 3}
    assert comparison["bootstrap_cluster_fields"] == ["task", "source_rollout"]


def test_41_report_contains_all_primary_comparison_tables():
    report = require(ARTIFACT_ROOT / "REPORT.md").read_text()
    for label in ("- h2:", "- h3:", "- h4:"):
        assert label in report
    assert "bootstrap=source rollout cluster; prefix rows averaged within cluster" in report
