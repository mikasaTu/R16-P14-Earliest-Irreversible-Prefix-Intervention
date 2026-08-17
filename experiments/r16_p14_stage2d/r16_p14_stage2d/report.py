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
    lines = [
        "# R16-P14 Stage-2D — Fresh-Process Event-Aligned Prefix Reuse",
        "",
        f"Final status: **{decision.get('overall', 'INCOMPLETE')}**. `accepted=false`; novelty remains `N2_ORACLE_PROTOCOL_BOUNDARY_ONLY`.",
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
        f"Calibration rows: {calibration.get('completed_rows')} / {calibration.get('expected_rows')}; raw H1: {calibration.get('h1_raw_pass')}; oracle mechanism: {calibration.get('oracle_mechanism_decision')}. The oracle is computed only after all candidate branches and is not a deployable algorithm.",
        "",
        "## Frozen rule and confirmatory evidence",
        "",
        f"Target rule: `{rule.get('target_shift')}`. Path rule: `{rule.get('path_obstacle')}`. Tail horizon is {rule.get('tail_horizon')}; calls are measured and never padded after success.",
        "",
        f"Confirmatory status: **{confirm.get('status')}**; method rows: {confirm.get('method_rows')} / {confirm.get('expected_method_rows')}; exact replay-admitted events: {confirm.get('replay_admitted_events')}.",
        "",
        f"H1 observed-safe window: **{decision.get('h1_observed_safe_window')}**. H2 cached versus equally long fresh prefix: **{decision.get('h2_cached_prefix_content')}**. H3 event-aligned versus immediate: **{decision.get('h3_event_aligned_handoff')}**. H4 nontrivial selection: **{decision.get('h4_nontrivial_selection')}**.",
        "",
        "The cached-versus-fresh comparison matches prefix length, detection-time call requirement, actor checkpoint, tail horizon, and maximum budgets. Cached-action count itself is never treated as an efficiency result; only measured actions, completion, actor calls, inference time, EEF/object paths, branch time, and retained/regressed progress count.",
        "",
        "## Confirmatory versus appendix oracle",
        "",
        f"The evaluation all-k appendix status is **{oracle.get('status', 'NOT_RUN')}** and reports `primary_decision_effect=NONE`; primary decision unchanged: {oracle.get('primary_decision_unchanged')}. It cannot retune the rule, baseline, threshold, or decision.",
        "",
        "## Mechanism reverse audit",
        "",
        f"The code-first audit covers {mechanism.get('valid_event_count')} replay-admitted events and explicitly records `new_idea_generated=false`. CACHED vs FRESH isolates old action content; CACHED vs NOQUERY isolates sham-query compute; HOLD isolates elapsed steps from motion content; FULL_OLD isolates damage from stale suffix continuation.",
        "",
        "Any downstream pattern after a failed upstream gate is labeled diagnostic-only and cannot support a positive or accepted claim.",
        "",
        "## Statistical contract",
        "",
        f"Paired event bootstrap: {statistics.get('bootstrap_replicates')} resamples with seed {statistics.get('bootstrap_seed')}. The event—not the prefix row—is resampled. Actor checkpoints are reported as repeated model measurements, and two tasks do not justify population-wide cross-task generalization.",
        "",
        "## Learned/deployable evidence",
        "",
        "NONE. Stage-2D stops after offline simulator evaluation. It does not validate a learned intervention policy or a VLA method.",
    ]
    text = "\n".join(lines) + "\n"
    (ARTIFACT_ROOT / "REPORT.md").write_text(text)
    if MIRROR_EXPERIMENT_OUTPUTS:
        (EXPERIMENT_ROOT / "reports/REPORT.md").write_text(text)
    print(json.dumps({"status": "REPORT_WRITTEN", "overall": decision.get("overall")}, sort_keys=True))


if __name__ == "__main__":
    main()
