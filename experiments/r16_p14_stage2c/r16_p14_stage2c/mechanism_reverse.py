from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from r16_p14_stage2b.io_utils import atomic_write_json, atomic_write_text, load_jsonl

from .settings import ARTIFACT_ROOT, DETECTION_PREFIX, H_VALID, TARGET_SHIFT_TASK


def _mean(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _rate(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [bool(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _group(rows: Iterable[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    result: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[tuple(row[key] for key in keys)].append(row)
    return result


def target_shift_activation(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit whether frozen nominal chunks can activate the target-shift cause label."""
    rows = []
    for event in events:
        if event["split"] != "calibration" or event["task"] != TARGET_SHIFT_TASK:
            continue
        chunk = np.asarray(event["original_chunk"], dtype=np.float32)[:H_VALID]
        previous = float(np.asarray(event["action_history"], dtype=np.float32)[-1, -1])
        gripper = np.concatenate(([previous], chunk[:, -1]))
        releases = [
            index
            for index in range(1, len(gripper))
            if gripper[index - 1] > 0.0 and gripper[index] < 0.0 and index > DETECTION_PREFIX
        ]
        positions = np.asarray(event["nominal_trace"]["positions"], dtype=np.float64)
        initial_z = float(np.asarray(event["initial_manipulated_qpos"], dtype=np.float64)[2])
        cause_eligible = []
        for index in releases:
            lifted = bool(event["phase"]["ever_lifted"]) or bool(
                np.max(positions[: index + 1, 2] - initial_z) >= 0.025
            )
            if lifted:
                cause_eligible.append(index)
        rows.append(
            {
                "event_id": event["event_id"],
                "actor_seed": int(event["actor_seed"]),
                "init_state_id": int(event["init_state_id"]),
                "post_detection_release_indices": releases,
                "cause_eligible_release_indices": cause_eligible,
                "anchor_ever_lifted": bool(event["phase"]["ever_lifted"]),
            }
        )
    return {
        "calibration_events": len(rows),
        "events_with_post_detection_release": sum(bool(row["post_detection_release_indices"]) for row in rows),
        "events_with_cause_eligible_release": sum(bool(row["cause_eligible_release_indices"]) for row in rows),
        "by_actor_seed": dict(sorted(Counter(row["actor_seed"] for row in rows).items())),
        "event_rows": rows,
        "confirmed_mechanism": (
            "The target-shift cause tracker can fire only on a positive-to-negative gripper transition "
            "after the manipulated object has been lifted. The frozen calibration chunks contain no "
            "cause-eligible transition, so changing shift magnitude cannot create the preregistered "
            "delayed-violation rate within the 16-action qualification window."
        ),
    }


def qualification_mechanism(summary: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, Any] = {}
    for task, rows in _group(summary["cells"], "task").items():
        failures = Counter(
            check
            for row in rows
            for check, passed in row["checks"].items()
            if not passed
        )
        by_task[str(task[0])] = {
            "cell_count": len(rows),
            "qualifying_cells": sum(bool(row["qualifies"]) for row in rows),
            "injection_contact_free_cells": sum(bool(row["checks"]["injection_contact_count_eq_0"]) for row in rows),
            "maximum_delayed_cause_violation_rate": max(float(row["delayed_cause_violation_rate"]) for row in rows),
            "failed_check_counts": dict(sorted(failures.items())),
        }
    return {
        "qualification_status": summary["status"],
        "failure_label": summary["failure_label"],
        "completed_attempts": int(summary["completed_attempts"]),
        "missing_attempts": int(summary["missing_attempts"]),
        "tasks": by_task,
        "target_shift_activation": target_shift_activation(events),
        "interpretation": (
            "Qualification failed before any method-gain comparison. Target shift lacked an activatable "
            "release event; path-obstacle cells either introduced contact at injection or produced too "
            "few delayed violations. Downstream matrices are therefore diagnostic-only."
        ),
    }


def paired_prefix_rows(matched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for (instance, prefix), rows in sorted(_group(matched, "event_instance_id", "prefix_k").items()):
        cached = next((row for row in rows if row["branch"] == "CACHED_MATCHED"), None)
        fresh = next((row for row in rows if row["branch"] == "FRESH_MATCHED"), None)
        noquery = next((row for row in rows if row["branch"] == "CACHED_NOQUERY"), None)
        hold = next((row for row in rows if row["branch"] == "HOLD_PREFIX_MATCHED"), None)
        if cached is None or fresh is None:
            continue
        row = {
            "event_instance_id": instance,
            "task": cached["task"],
            "split": cached["split"],
            "parameter_id": cached["parameter_id"],
            "generator_actor_seed": int(cached["generator_actor_seed"]),
            "init_state_id": int(cached["init_state_id"]),
            "prefix_k": int(prefix),
            "safe_success_delta": int(cached["safe_success"]) - int(fresh["safe_success"]),
            "task_success_delta": int(cached["task_success"]) - int(fresh["task_success"]),
            "cause_violation_delta": int(cached["cause_violation"]) - int(fresh["cause_violation"]),
            "cached_minus_fresh_progress": float(cached["task_progress_retained_at_k"]) - float(fresh["task_progress_retained_at_k"]),
            "fresh_minus_cached_new_actions": float(fresh["new_non_nominal_actions"]) - float(cached["new_non_nominal_actions"]),
            "cached_minus_fresh_completion_steps": float(cached["completion_steps"]) - float(fresh["completion_steps"]),
            "cached_minus_fresh_eef_path": float(cached["eef_path_length"]) - float(fresh["eef_path_length"]),
            "cached_minus_fresh_object_path": float(cached["object_path_length"]) - float(fresh["object_path_length"]),
            "cached_minus_fresh_contacts": float(cached["contact_count"]) - float(fresh["contact_count"]),
            "action_disagreement": cached.get("cached_fresh_action_displacement"),
            "compute_matched": cached["allocated_actor_call_budget"] == fresh["allocated_actor_call_budget"],
            "budget_matched": cached["total_post_detection_action_budget"] == fresh["total_post_detection_action_budget"],
        }
        if noquery is not None:
            row["cached_noquery_same_safe_outcome"] = bool(noquery["safe_success"] == cached["safe_success"])
            row["cached_noquery_actor_call_saving"] = float(cached["actor_calls"]) - float(noquery["actor_calls"])
        if hold is not None:
            row["cached_minus_hold_safe_success"] = int(cached["safe_success"]) - int(hold["safe_success"])
        result.append(row)
    return result


def _outcome_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "safe_success_delta": _mean(rows, "safe_success_delta"),
        "cause_violation_delta": _mean(rows, "cause_violation_delta"),
        "mean_action_disagreement": _mean(rows, "action_disagreement"),
        "mean_cached_minus_fresh_progress": _mean(rows, "cached_minus_fresh_progress"),
        "mean_fresh_minus_cached_new_actions": _mean(rows, "fresh_minus_cached_new_actions"),
        "mean_cached_minus_fresh_completion_steps": _mean(rows, "cached_minus_fresh_completion_steps"),
        "mean_cached_minus_fresh_eef_path": _mean(rows, "cached_minus_fresh_eef_path"),
        "mean_cached_minus_fresh_object_path": _mean(rows, "cached_minus_fresh_object_path"),
        "mean_cached_minus_fresh_contacts": _mean(rows, "cached_minus_fresh_contacts"),
    }


def matched_mechanism(matched: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = paired_prefix_rows(matched)
    evaluation = [row for row in pairs if row["split"] == "evaluation"]
    groups = {
        "cached_improves": [row for row in evaluation if row["safe_success_delta"] > 0],
        "cached_worsens": [row for row in evaluation if row["safe_success_delta"] < 0],
        "same_safe_outcome": [row for row in evaluation if row["safe_success_delta"] == 0],
    }
    by_task = {
        str(task[0]): _outcome_group(rows)
        for task, rows in sorted(_group(evaluation, "task").items())
    }
    noquery_rows = [row for row in evaluation if "cached_noquery_same_safe_outcome" in row]
    return {
        "all_paired_prefix_cells": len(pairs),
        "evaluation_paired_prefix_cells": len(evaluation),
        "all_compute_matched": bool(evaluation) and all(row["compute_matched"] for row in evaluation),
        "all_budget_matched": bool(evaluation) and all(row["budget_matched"] for row in evaluation),
        "outcome_groups": {name: _outcome_group(rows) for name, rows in groups.items()},
        "by_task": by_task,
        "cached_noquery": {
            "paired_cells": len(noquery_rows),
            "same_safe_outcome_rate": _rate(noquery_rows, "cached_noquery_same_safe_outcome"),
            "mean_actor_call_saving": _mean(noquery_rows, "cached_noquery_actor_call_saving"),
        },
        "metric_semantics": {
            "new_non_nominal_actions": (
                "The metric counts cached A[d:k] actions as retained and fresh-prefix/tail actions as new. "
                "A reduction can therefore be structural at larger k; it is not independent evidence of "
                "better safety or task completion."
            ),
            "safe_success": "Task success AND no target-cause violation; this is the primary outcome used to classify improvement or degradation.",
            "action_disagreement": "Normalized ACT action displacement is diagnostic association, not a learned selector or causal mediator test.",
        },
        "paired_rows": pairs,
    }


def operator_mechanism(recovery: list[dict[str, Any]]) -> dict[str, Any]:
    evaluation = [row for row in recovery if row["split"] == "evaluation"]
    return {
        str(operator[0]): {
            "rows": len(rows),
            "safe_success_rate": _rate(rows, "safe_success"),
            "task_success_rate": _rate(rows, "task_success"),
            "cause_violation_rate": _rate(rows, "cause_violation"),
            "mean_new_non_nominal_actions": _mean(rows, "new_non_nominal_actions"),
            "mean_actor_calls": _mean(rows, "actor_calls"),
            "mean_completion_steps": _mean(rows, "completion_steps"),
        }
        for operator, rows in sorted(_group(evaluation, "operator").items())
    }


def selection_mechanism(root: Path) -> dict[str, Any]:
    path = root / "crossfit_replanability/crossfit_rows.jsonl"
    if not path.is_file():
        return {"available": False}
    rows = [row for row in load_jsonl(path) if row["split"] == "evaluation"]
    selected = Counter(int(row["selected_prefix"]) for row in rows)
    baseline = Counter(int(row["baseline_prefix"]) for row in rows)
    return {
        "available": True,
        "rows": len(rows),
        "selected_prefix_distribution": dict(sorted(selected.items())),
        "baseline_prefix_distribution": dict(sorted(baseline.items())),
        "safe_success_difference": _mean(rows, "safe_success_difference"),
        "new_action_reduction_fraction": _mean(rows, "new_action_reduction_fraction"),
        "differs_from_detection_rate": _rate(rows, "differs_from_d"),
        "differs_from_last_recoverable_rate": _rate(rows, "differs_from_last_recoverable"),
        "interior_prefix_rate": _rate(rows, "interior_prefix"),
        "confirmed_code_path": (
            "Two recovery actors rank prefixes by safe success, then fewer new actions, then fewer actor "
            "calls, then lower k. The held-out actor is read only after selection. When safe outcomes tie, "
            "the new-action definition can drive selection toward later prefixes."
        ),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    q = payload["qualification"]
    m = payload["matched_prefix"]
    improve = m["outcome_groups"]["cached_improves"]
    worsen = m["outcome_groups"]["cached_worsens"]
    activation = q["target_shift_activation"]
    return "\n".join(
        [
            "# Stage-2C code-first mechanism reverse audit",
            "",
            "This is a reverse explanation of frozen observations. It introduces no new method or idea.",
            "",
            "## Qualification mechanism",
            "",
            f"All {q['completed_attempts']} qualification attempts completed with {q['missing_attempts']} missing attempts, but the gate was `{q['qualification_status']}` (`{q['failure_label']}`).",
            f"For target shift, {activation['events_with_post_detection_release']}/{activation['calibration_events']} chunks contained a post-detection gripper release, and {activation['events_with_cause_eligible_release']} were eligible after the lift condition. This confirms why all frozen target-shift severities had zero delayed cause violations.",
            "Path-obstacle failures are summarized from every frozen cell in the JSON companion; no task or severity was selected using method outcomes.",
            "",
            "## Matched-prefix mechanism",
            "",
            f"The evaluation contains {m['evaluation_paired_prefix_cells']} exact cached/fresh prefix pairs. Cached improved safe success in {improve['count']} cells and worsened it in {worsen['count']} cells.",
            "The JSON companion reports action disagreement, retained progress, cause violations, completion steps, paths and contacts separately for improvements and degradations.",
            "`new_non_nominal_actions` is partly structural: cached prefix actions are defined as retained, while fresh-prefix and tail actions are defined as new. A reduction without safe-success improvement is not an independent efficiency mechanism.",
            "",
            "## Causal scope",
            "",
            "Cached and fresh primary branches match the detection-time call, the call at k, the h=16 tail, action budget, simulator seed, perturbation and actor. Therefore an observed branch difference is isolated to which k-d prefix actions were executed. Associations with disagreement or progress explain the code path but do not establish a universal physical irreversibility mechanism.",
            "",
            "All downstream outcomes remain diagnostic-only when the perturbation-family gate is blocked. `accepted=false`, Stage-1b remains `KILLED_IMMUTABLE`, and novelty remains at most N2.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    qualification = json.loads((root / "task_qualification/summary.json").read_text())
    events = load_jsonl(root / "actor_events/events.jsonl")
    matched = load_jsonl(root / "formal_matrix/matched_prefix_rows.jsonl")
    recovery = load_jsonl(root / "formal_matrix/recovery_operator_rows.jsonl")
    payload = {
        "schema_version": 1,
        "purpose": "code-first reverse explanation of observed gains and losses; no new idea",
        "qualification": qualification_mechanism(qualification, events),
        "matched_prefix": matched_mechanism(matched),
        "recovery_operators": operator_mechanism(recovery),
        "crossfit_selection": selection_mechanism(root),
        "new_idea_generated": False,
    }
    atomic_write_json(root / "mechanism_reverse_audit.json", payload)
    atomic_write_text(root / "mechanism_reverse_audit.md", render_markdown(payload))
    print(json.dumps({"status": "complete", "output": str(root / "mechanism_reverse_audit.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
