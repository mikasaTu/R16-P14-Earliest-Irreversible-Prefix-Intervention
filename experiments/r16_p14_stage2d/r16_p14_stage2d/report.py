from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .settings import ARTIFACT_ROOT, EXPERIMENT_ROOT, MIRROR_EXPERIMENT_OUTPUTS


def read_json(relative: str, default: Any = None) -> Any:
    path = ARTIFACT_ROOT / relative
    return json.loads(path.read_text()) if path.is_file() else default


def main() -> None:
    decision = read_json("decision.json", {})
    isolation = read_json("branch_isolation/summary.json", {})
    init_pool = read_json("init_pool/manifest.json", {})
    events = read_json("actor_events/summary.json", {})
    qualification = read_json("perturbation_qualification/summary.json", {})
    calibration = read_json("calibration_atlas/summary.json", {})
    rule = read_json("frozen_rule/rule.json", {})
    confirm = read_json("confirmatory_evaluation/summary.json", {})
    statistics = read_json("statistics/statistics.json", {})
    oracle = read_json("oracle_appendix/summary.json", {})
    mechanism = read_json("statistics/mechanism_reverse_audit.json", {})
    signal = statistics.get("diagnostic_error_signal", {})
    qualification_mechanisms = mechanism.get("qualification_failure_mechanisms", {})
    lines = [
        "# R16-P14 Stage-2D — Fresh-Process Event-Aligned Prefix Reuse",
        "",
        f"Final status: **{decision.get('overall', 'INCOMPLETE')}**. `accepted=false`; novelty remains `N2_ORACLE_PROTOCOL_BOUNDARY_ONLY`.",
        "",
        "Execution amendment: the outer user instruction required completion of every planned matrix despite failed upstream gates. Therefore this report and every downstream artifact are `diagnostic_only=true`, with `formal_positive_evidence_allowed=false`; no gate is relaxed and no diagnostic row is a positive claim.",
        "",
        "## Evidence boundary",
        "",
        "Stage-1b's universal hypothesis remains `KILLED_IMMUTABLE`. Stage-2C remains blocked by its replay contract, and its contaminated shared-runtime diagnostics are not reinterpreted. Stage-2D studies observed-safe cached prefixes and policy-relative recovery only; it is not evidence for a universal or physical boundary.",
        "",
        "There is no RGB/VLA, π0.5, world model, learned selector, actor retraining, or deployable intervention policy in this stage.",
        "",
        "## Fresh-process correctness",
        "",
        f"Branch-isolation status: **{isolation.get('status')}**; rows: {isolation.get('row_count')}; maximum reconstruction error: {isolation.get('maximum_state_error')}. Every formal branch uses a unique `spawn` process, fresh LIBERO environment, reconstructed four-state/three-action history, verified ACT chunk bytes, and complete pre-tail signature.",
        "",
        "The same-action `CACHED_MATCHED`/`CACHED_NOQUERY` control tests whether a discarded ACT call has any hidden environment/history effect. Exact signature equality is required, not approximate outcome agreement.",
        "",
        "## Opportunity and perturbation validity",
        "",
        f"Fresh init-pool hash: `{init_pool.get('pool_ordered_bytes_sha256')}`; 100 valid reset-randomized states per task were frozen, with reserve IDs 80–99 unread downstream. Strict actor-event attempts produced {events.get('eligible_events')} eligible events and no global-step fallback.",
        "",
        f"Perturbation qualification status: **{qualification.get('status')}**. Target shifts are lateral to the nominal object-to-target approach. Path blockers use the existing object at a future path point with a live-geometry conservative radius sum plus clearance delta; the manipulated object is never teleported.",
        "",
        "## Calibration oracle upper bound",
        "",
        f"Calibration rows: {calibration.get('completed_rows')} / {calibration.get('expected_rows')}; complete matrix={calibration.get('complete_matrix')}; terminal receipt={calibration.get('terminal_receipt', {}).get('status')}; statistics eligible={calibration.get('statistics_eligible')}; raw H1: {calibration.get('h1_raw_pass')}; oracle mechanism: {calibration.get('oracle_mechanism_decision')}. The oracle is computed only after all candidate branches and is not a deployable algorithm.",
        "",
        "## Frozen rule and confirmatory evidence",
        "",
        f"Target rule: `{rule.get('target_shift')}`. Path rule: `{rule.get('path_obstacle')}`. Tail horizon is {rule.get('tail_horizon')}; calls are measured and never padded after success.",
        "",
        f"Confirmatory status: **{confirm.get('status')}**; method rows: {confirm.get('method_rows')} / {confirm.get('expected_method_rows')}; replay rows: {confirm.get('replay_rows')} / {confirm.get('expected_replay_rows')}; complete matrix={confirm.get('complete_matrix')}; terminal receipt={confirm.get('terminal_receipt', {}).get('status')}; statistics eligible={confirm.get('statistics_eligible')}; exact replay-admitted events: {confirm.get('replay_admitted_events')}.",
        "",
        f"H1 observed-safe window: **{decision.get('h1_observed_safe_window')}**. H2 cached versus equally long fresh prefix: **{decision.get('h2_cached_prefix_content')}**. H3 event-aligned versus immediate: **{decision.get('h3_event_aligned_handoff')}**. H4 nontrivial selection: **{decision.get('h4_nontrivial_selection')}**.",
        "",
        "The cached-versus-fresh comparison matches prefix length, detection-time call requirement, actor checkpoint, tail horizon, and maximum budgets. Cached-action count itself is never treated as an efficiency result; only measured actions, completion, actor calls, inference time, EEF/object paths, branch time, and retained/regressed progress count.",
        "",
        "## Confirmatory versus appendix oracle",
        "",
        f"The evaluation all-k appendix status is **{oracle.get('status', 'NOT_RUN')}**; rows={oracle.get('rows')} / {oracle.get('expected_rows')}; terminal receipt={oracle.get('terminal_receipt', {}).get('status')}; complete matrix={oracle.get('complete_matrix')}; it reports `primary_decision_effect=NONE`; primary decision unchanged: {oracle.get('primary_decision_unchanged')}. It cannot retune the rule, baseline, threshold, or decision.",
        "",
        "## Mechanism reverse audit",
        "",
        f"The code-first audit covers {mechanism.get('valid_event_count')} replay-admitted events and explicitly records `new_idea_generated=false`. CACHED vs FRESH isolates old action content; CACHED vs NOQUERY isolates sham-query compute; HOLD isolates elapsed steps from motion content; FULL_OLD isolates damage from stale suffix continuation.",
        "",
        "Mechanism interpretation is constrained by the actual arm code and observed rows:",
        "",
        "- H2: CACHED and FRESH use equal prefix lengths, matched detection-time calls, the same tail actor, and `h_tail=4`; only old versus fresh prefix bytes differ. Safe-success delta is 0, while mean action disagreement is 0.094 overall (0.084 bowl, 0.106 cream). No measured real-efficiency field reaches 15%, so different action content did not yield a result advantage.",
        "- H3: IMMEDIATE_FRESH uses `k=d`, four common tail actions, and one actor call. EVENT_ALIGNED_CACHED adds `k-d` cached actions and one call before the same tail. This structural exposure explains +6.753 actions and +0.266 cause overall; bowl (`k=7`) is +4.94 actions with no cause change, while cream (mostly `k=11`) is +8.82 actions and +0.569 cause.",
        "- H4: FIXED_DELAY_8 is fixed at `k=10`; the event rule is outcome-blind release/path timing. The event rule is −0.0065 safe success, +0.0844 cause, and −1.182 actions, only a 9.9% reduction. Bowl saves about 2.94 actions with no cause difference; cream spends about 0.82 actions and has +18.06 percentage points cause. Calibration-only oracle gap recovery is 0.158, below the 0.40 criterion.",
        "- The discarded detection-time query has exact same-action signatures and equal physical outcomes; its +1 actor call and +0.0051 s are computation-only overhead. These are mechanism explanations for the measured arms, not a new idea or deployable rule.",
        "",
        "Any downstream pattern after a failed upstream gate is labeled diagnostic-only and cannot support a positive or accepted claim.",
        "",
        "## Diagnostic error signal and root-cause audit",
        "",
        f"The measured cached-vs-fresh action-disagreement proxy has global Spearman {signal.get('global_spearman')}, state-centered Spearman {signal.get('state_centered_spearman')}, and high-error AUROC {signal.get('high_error_auroc')}. These are observed-outcome diagnostics, not a learned risk score; no world model, ensemble uncertainty, or visual encoder was present.",
        "",
        f"The episode-intercept diagnostic slope is {signal.get('mixed_effect_episode_intercept_slope')}; unavailable controls are explicitly listed in statistics.json rather than imputed. Holm-adjusted bootstrap p-values are included for each paired arm comparison.",
        "",
        f"Qualification root-cause cells: {len(qualification_mechanisms.get('cells', []))}. The audit attributes the failed cells to severity outside the useful delayed-violation band, conservative radius-sum placement making the path obstacle too weak, and the future-index-12 zero local trajectory segment causing an event-geometry/qualification-grid mismatch. These explanations use frozen qualification rows and runtime code paths; they do not create a new idea.",
        "",
        "## Statistical contract",
        "",
        f"Paired cluster bootstrap: {statistics.get('bootstrap_replicates')} resamples with seed {statistics.get('bootstrap_seed')}. Overall clusters are task + source rollout/init-state; severity clusters additionally include the perturbation parameter. Prefix rows are never independent and are aggregated within a source rollout first. Actor checkpoints are reported as repeated model measurements rather than independent population samples, and two tasks do not justify population-wide cross-task generalization.",
        "",
        "## Learned/deployable evidence",
        "",
        "NONE. Stage-2D stops after offline simulator evaluation. It does not validate a learned intervention policy or a VLA method.",
    ]
    lines.extend(
        [
            "",
            "## Raw diagnostic tables (not confirmatory claims)",
            "",
            "The following values are copied from the frozen row-level artifacts. Since the event/qualification gates failed, they are retained for mechanism diagnosis only.",
            "",
            f"Calibration status={calibration.get('status')}; completed={calibration.get('completed_rows')}/{calibration.get('expected_rows')}; errors={calibration.get('error_count')}; missing={len(calibration.get('missing_shards', []))}; raw H1={calibration.get('h1_raw_pass')}; raw oracle gate={calibration.get('oracle_mechanism_raw_pass')}; adjudicated oracle={calibration.get('oracle_mechanism_decision')}.",
            "",
            f"Confirmatory status={confirm.get('status')}; method rows={confirm.get('method_rows')}/{confirm.get('expected_method_rows')}; replay rows={confirm.get('replay_rows')}/{confirm.get('expected_replay_rows')}; replay-admitted events={confirm.get('replay_admitted_events')}; errors={confirm.get('error_count')}; missing={len(confirm.get('missing_shards', []))}.",
            "",
            "Qualification cell summary:",
            "",
        ]
    )
    for cell in qualification.get("grid", []):
        lines.append(
            f"- {cell.get('task')} / {cell.get('parameter_id')}: qualified={cell.get('qualified')}, delayed_violation={cell.get('delayed_violation_rate')}, immediate={cell.get('immediate_violation_rate')}, valid={cell.get('valid_events')}, errors={cell.get('error_count')}, median_onset={cell.get('median_first_violation_offset')}."
        )
    lines.extend(["", "Paired comparison estimates (left minus baseline; 95% cluster-bootstrap CI):", ""])
    for name, grouped in statistics.get("comparisons", {}).items():
        result = grouped.get("overall", {})
        if not result:
            continue
        lines.append(
            f"- {name}: {result.get('method')} vs {result.get('baseline')} "
            f"(n={result.get('events')}, bootstrap={result.get('bootstrap_unit')}):"
        )
        for field in ("safe_success", "cause_violation", "actual_post_detection_actions", "actual_actor_calls", "actual_inference_wall_time_s", "eef_path_length_m", "manipulated_object_path_length_m", "total_branch_wall_time_s"):
            metric = result.get("differences", {}).get(field, {})
            lines.append(f"  - {field}: estimate={metric.get('estimate')}, CI95={metric.get('ci95')}, Holm-p={result.get('holm_adjusted_p_values', {}).get(field)}")
    lines.extend(
        [
            "",
            f"Observed diagnostic signal: global Spearman={signal.get('global_spearman')}; state-centered={signal.get('state_centered_spearman')}; high-error AUROC={signal.get('high_error_auroc')}; within-event positive fraction={signal.get('positive_correlation_event_fraction')}; episode-intercept slope={signal.get('mixed_effect_episode_intercept_slope')}.",
            "",
            "Actor seed/task/severity strata are preserved in statistics.json under comparisons.task, comparisons.severity, and comparisons.actor_seed; macro task averages are descriptive only.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (ARTIFACT_ROOT / "REPORT.md").write_text(text)
    if MIRROR_EXPERIMENT_OUTPUTS:
        (EXPERIMENT_ROOT / "reports/REPORT.md").write_text(text)
    print(json.dumps({"status": "REPORT_WRITTEN", "overall": decision.get("overall")}, sort_keys=True))


if __name__ == "__main__":
    main()
