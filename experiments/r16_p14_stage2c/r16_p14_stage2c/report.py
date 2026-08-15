from __future__ import annotations

import json
from pathlib import Path

from r16_p14_stage2b.io_utils import atomic_write_json, atomic_write_text

from .settings import ARTIFACT_ROOT, EXPERIMENT_ROOT


def fmt(value, digits=3):
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    decision = json.loads((ARTIFACT_ROOT / "decision.json").read_text())
    aggregate = json.loads((ARTIFACT_ROOT / "aggregate_summary.json").read_text())
    stats = json.loads((ARTIFACT_ROOT / "statistics.json").read_text())
    qualification = json.loads((ARTIFACT_ROOT / "task_qualification/summary.json").read_text())
    events = json.loads((ARTIFACT_ROOT / "actor_events/summary.json").read_text())
    pool = json.loads((ARTIFACT_ROOT / "actor_events/formal_event_pool_summary.json").read_text())
    formal = json.loads((ARTIFACT_ROOT / "formal_matrix/summary.json").read_text())
    mechanism = json.loads((ARTIFACT_ROOT / "mechanism_audit.json").read_text())
    root = json.loads((ARTIFACT_ROOT / "contract_repair/replay_root_cause.json").read_text())
    a_stats = stats["track_a_safe_success"]
    b_stats = stats["track_b_safe_success"]
    lines = [
        "# R16-P14 Stage-2C — Recoverability-Defined and Compute-Matched Validation",
        "",
        f"Overall: **{decision['overall']}**. Track A: **{decision['track_a_operator_relative_prefix_reuse']}**; Track B: **{decision['track_b_crossfit_replanability']}**. `accepted=false`; novelty remains `N2_ORACLE_PROTOCOL_BOUNDARY_ONLY`.",
        "",
        "## Immutable negative evidence",
        "",
        "Stage-1b's universal hypothesis remains `KILLED_IMMUTABLE`. Stage2B remains blocked on actor-conditioned perturbation and actor-history replay; its Track A/B results remain inconclusive. Nothing in this stage overwrites or reinterprets those records.",
        "",
        "## Correctness repairs",
        "",
        f"The sole historical replay failure first becomes observable at global step {root['observation_feature']['first_observable_divergent_global_step']}: a {root['four_state_history']['max_abs_difference']:.3g} history delta is amplified into a {root['original_actor_chunk']['max_abs_difference']:.3g} ACT chunk delta. Three new reconstructions are mutually byte-exact. Stage2C therefore uses outcome-blind 3/3 admission and excludes unstable events; it does not relax tolerances.",
        "",
        "Replay rates now divide by all attempts and require zero errors. Prefix safety is absorbing, and goal progress comes from the live BDDL predicate/site/object geometry rather than demo-0.",
        "",
        "## Qualified perturbation families",
        "",
        f"Qualification status: **{qualification['status']}**. Target-shift two-severity gate: `{qualification['target_shift_two_severities']}`; second family: `{qualification['second_failure_family']}` with gate `{qualification['second_failure_family_pass']}`. All later experiments ran even when a gate failed.",
        "",
        f"Actor events: {events['admitted_events']} admitted from {events['completed_attempts']} completed attempts; unstable exclusion rate {events['unstable_event_exclusion_rate']:.3f}. Formal minimum-data gate: `{pool['minimum_data_pass']}`.",
        "",
        "## Compute-matched cached-prefix value",
        "",
        f"Track A evaluation rows: {aggregate['track_a_rows']}. Safe-success difference (cached minus fresh): {fmt(a_stats['estimate'] if a_stats else None)} with cluster-bootstrap 95% CI {a_stats['ci95'] if a_stats else 'n/a'}. Mean new-action reduction: {fmt(aggregate['track_a_new_action_reduction_fraction'])}.",
        "",
        "Both primary branches execute exactly `k-d` prefix actions, make one detection-time call and one call at k, then use h=16 with identical action budgets. `CACHED_NOQUERY` remains secondary deployment evidence.",
        "",
        "## Operator-relative recoverability",
        "",
        f"Atlas rows: {aggregate['atlas_rows']}; event boundaries: {aggregate['boundary_rows']}; invalid/nonmonotonic prefix events: {aggregate['invalid_events']}. Median recoverable windows by task: `{aggregate['median_recoverable_windows']}`. `k_irrev_U` is an operator-family persistent crossing, not a physical irreversibility point.",
        "",
        "## Cross-fitted replanability",
        "",
        f"Leave-one-recovery-actor-out rows: {aggregate['crossfit_rows']}. Frozen calibration-only baseline: `{decision['frozen_strongest_baseline']}`. Held-out safe-success difference: {fmt(b_stats['estimate'] if b_stats else None)} with 95% CI {b_stats['ci95'] if b_stats else 'n/a'}; mean new-action reduction: {fmt(aggregate['track_b_new_action_reduction_fraction'])}.",
        "",
        "Held-out outcomes never enter prefix selection. Actor seed is a repeated measurement within `(task, init_state_id)`; all reported intervals use the preregistered 10,000-draw cluster bootstrap.",
        "",
        "## Why mechanisms raise or lower performance",
        "",
        f"Across {mechanism['paired_prefix_count']} matched prefix cells, cached reuse improves safe outcome in {mechanism['outcome_groups']['cached_improves']['count']} and worsens it in {mechanism['outcome_groups']['cached_worsens']['count']}. The reverse audit attributes gains only to retained nominal progress/fewer new actions under equal compute; losses are checked against stale-prefix cause violations and larger cached/fresh action displacement. This is an explanation of frozen code behavior, not a new idea.",
        "",
        "## Retired mechanisms and evidence boundary",
        "",
        "Local repair remains `RETIRED_NO_SIGNAL`; the operator router remains `RETIRED_NO_SIGNAL`. No learned predictor, RGB/VLA, world model, π0.5, or deployable policy was trained. Learned/deployable evidence remains **NONE**.",
        "",
        "## Execution completeness and decision",
        "",
        f"Formal matrix: {formal['matched_rows']} matched rows, {formal['recovery_rows']} recovery rows, {formal['error_count']} errors, `all_complete={formal['all_complete']}`. Upstream gates: `{decision['upstream_gates']}`.",
        "",
        f"Final decision: **{decision['overall']}**. Raw Track-A/Track-B criteria were `{decision['track_a_raw_signal']}` / `{decision['track_b_raw_signal']}`, but positive labels are impossible whenever an upstream gate is false.",
        "",
    ]
    report = "\n".join(lines)
    atomic_write_text(ARTIFACT_ROOT / "REPORT.md", report)
    destination = EXPERIMENT_ROOT / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination / "REPORT.md", report)
    atomic_write_json(EXPERIMENT_ROOT / "decision.json", decision)
    atomic_write_json(destination / "statistics.json", stats)
    atomic_write_json(destination / "mechanism_audit.json", mechanism)
    print(json.dumps({"status": "complete", "report": str(ARTIFACT_ROOT / "REPORT.md")}, sort_keys=True))


if __name__ == "__main__":
    main()

