from __future__ import annotations

import inspect
import json
from collections import defaultdict
from typing import Any

import numpy as np

from . import runtime
from .io_utils import atomic_write_json, load_jsonl
from .settings import ARTIFACT_ROOT, EXPERIMENT_ROOT, MIRROR_EXPERIMENT_OUTPUTS


FIELDS = (
    "safe_success",
    "cause_violation",
    "actual_post_detection_actions",
    "actual_actor_calls",
    "actual_inference_wall_time_s",
    "eef_path_length_m",
    "manipulated_object_path_length_m",
    "progress_regression_m",
)


def contrast(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    result = {"left": left, "right": right, "n": 0, "effects_left_minus_right": {}}
    pairs = [row for row in rows if left in row["methods"] and right in row["methods"]]
    result["n"] = len(pairs)
    for field in FIELDS:
        values = []
        for row in pairs:
            lval, rval = row["methods"][left].get(field), row["methods"][right].get(field)
            if lval is not None and rval is not None:
                values.append(float(lval) - float(rval))
        estimate = float(np.mean(values)) if values else None
        result["effects_left_minus_right"][field] = {
            "estimate": estimate,
            "direction": (
                "equal"
                if estimate == 0
                else "higher_in_left"
                if estimate is not None and estimate > 0
                else "lower_in_left"
                if estimate is not None
                else "unavailable"
            ),
        }
    return result


def source_location(function) -> dict[str, Any]:
    lines, start = inspect.getsourcelines(function)
    return {
        "file": inspect.getsourcefile(function),
        "function": function.__name__,
        "start_line": start,
        "end_line": start + len(lines) - 1,
    }


def main() -> None:
    rows = [
        row
        for row in load_jsonl(ARTIFACT_ROOT / "confirmatory_evaluation/paired_rows.jsonl")
        if row["replay_admitted_3_of_3"]
    ]
    contrasts = {
        "cached_content_at_matched_k": contrast(
            rows, "EVENT_ALIGNED_CACHED", "FRESH_MATCHED_AT_RULE_K"
        ),
        "event_timing_vs_immediate": contrast(
            rows, "EVENT_ALIGNED_CACHED", "IMMEDIATE_FRESH"
        ),
        "discarded_detection_query": contrast(
            rows, "EVENT_ALIGNED_CACHED", "CACHED_NOQUERY_AT_RULE_K"
        ),
        "cached_motion_vs_hold": contrast(
            rows, "EVENT_ALIGNED_CACHED", "HOLD_MATCHED_AT_RULE_K"
        ),
        "bounded_cached_prefix_vs_full_old": contrast(
            rows, "EVENT_ALIGNED_CACHED", "FULL_OLD_CHUNK"
        ),
    }
    mechanisms = []
    content = contrasts["cached_content_at_matched_k"]["effects_left_minus_right"]
    mechanisms.append(
        {
            "mechanism": "cached action content under matched handoff time",
            "code_path": "execute_branch selects old[d:k] for CACHED_MATCHED and fresh_d[:k-d] for FRESH_MATCHED, then both call the same actor at k and execute h_tail=4",
            "observed_effect": content,
            "interpretation": (
                "A safe-success increase with no cause increase isolates useful cached motion content. "
                "A decrease isolates stale pre-perturbation content because call count, prefix length, tail actor, and tail horizon are matched."
            ),
        }
    )
    query = contrasts["discarded_detection_query"]["effects_left_minus_right"]
    mechanisms.append(
        {
            "mechanism": "detection-time sham query overhead",
            "code_path": "CACHED_MATCHED calls ACT at d and discards the tensor; CACHED_NOQUERY executes identical old[d:k] bytes",
            "observed_effect": query,
            "interpretation": (
                "Exact pre-tail signatures show the discarded query has no physical or history effect. "
                "Any actor-call, inference-time, or branch-time increase is therefore bookkeeping compute overhead, not action content value."
            ),
        }
    )
    timing = contrasts["event_timing_vs_immediate"]["effects_left_minus_right"]
    mechanisms.append(
        {
            "mechanism": "event-aligned delay versus immediate replanning",
            "code_path": "arm_plan maps IMMEDIATE_FRESH to k=d and EVENT_ALIGNED_CACHED to the frozen event rule",
            "observed_effect": timing,
            "interpretation": (
                "Safety changes come from actions executed between d and k plus the changed state/history seen by the common tail. "
                "Execution-cost changes are reported only for measured actions, calls, paths, and wall time; retained cached count is not treated as efficiency."
            ),
        }
    )
    mechanisms.append(
        {
            "mechanism": "motion content versus elapsed-time control",
            "code_path": "HOLD_MATCHED preserves gripper state but zeros six motion dimensions for exactly k-d actions",
            "observed_effect": contrasts["cached_motion_vs_hold"]["effects_left_minus_right"],
            "interpretation": "The contrast separates old motion content from merely waiting the same number of simulator steps.",
        }
    )
    mechanisms.append(
        {
            "mechanism": "bounded reuse versus stale full-chunk execution",
            "code_path": "FULL_OLD_CHUNK executes old[d:H] with no recovery tail; the event rule truncates reuse and invokes the actor at k",
            "observed_effect": contrasts["bounded_cached_prefix_vs_full_old"]["effects_left_minus_right"],
            "interpretation": "This locates harm caused by continuing stale actions beyond the event-aligned boundary.",
        }
    )
    audit = {
        "schema_version": 1,
        "method": "code-first mechanism reverse audit",
        "new_idea_generated": False,
        "scope": "explain observed improvements and decreases in the preregistered arms only",
        "valid_event_count": len(rows),
        "source_locations": {
            "execute_branch": source_location(runtime.execute_branch),
            "arm_plan": source_location(runtime.arm_plan),
            "actor_call": source_location(runtime.actor_call),
            "cause_tracker": source_location(runtime.CauseTracker.observe_action),
            "perturbation": source_location(runtime.apply_perturbation),
        },
        "contrasts": contrasts,
        "mechanisms": mechanisms,
        "causal_limits": [
            "The operator uses a frozen state ACT, not RGB or a VLA.",
            "Only two LIBERO task families are observed.",
            "The oracle is an upper bound and is not deployable.",
            "A failed upstream gate makes all downstream patterns diagnostic-only.",
        ],
    }
    output = ARTIFACT_ROOT / "statistics"
    atomic_write_json(output / "mechanism_reverse_audit.json", audit)
    report = [
        "# Code-first mechanism reverse audit",
        "",
        "This audit explains the preregistered increases and decreases; it does not generate a new idea.",
        "",
    ]
    for mechanism in mechanisms:
        report.extend(
            [
                f"## {mechanism['mechanism']}",
                "",
                mechanism["code_path"],
                "",
                mechanism["interpretation"],
                "",
                "Observed mean effects (left minus right):",
                "",
            ]
        )
        for field, value in mechanism["observed_effect"].items():
            report.append(f"- {field}: {value['estimate']} ({value['direction']})")
        report.append("")
    (output / "mechanism_reverse_audit.md").write_text("\n".join(report))
    if MIRROR_EXPERIMENT_OUTPUTS:
        mirror = EXPERIMENT_ROOT / "statistics"
        mirror.mkdir(parents=True, exist_ok=True)
        for name in ("mechanism_reverse_audit.json", "mechanism_reverse_audit.md"):
            (mirror / name).write_bytes((output / name).read_bytes())
    print(json.dumps({"status": "COMPLETE", "events": len(rows), "new_idea_generated": False}, sort_keys=True))


if __name__ == "__main__":
    main()
