
"""Diagnostic-only Phase-2 budget selection from a Phase-1 calibration summary."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TASKS = (
    "put_the_cream_cheese_in_the_bowl",
    "put_the_bowl_on_the_plate",
)
OPERATORS = (
    "fresh_h4",
    "fresh_h16",
    "hold_1+fresh_h4",
    "rollback_1+fresh_h16",
)
PHASE1_PREFIXES = (2, 4, 8, 12, 16)
BUDGETS = tuple((tail, action, 8) for tail in (4, 8, 16) for action in (8, 16, 32))
BUDGET_KEYS = frozenset(BUDGETS)
DEFAULT_DIAGNOSTIC_CONTINUATION_DOC_SHA256 = (
    "2d14427c2391422dd372b0528d6d784d1066d8d059e1405597cfb1954f260c6a"
)
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_ERROR_STATUSES = {
    "error", "failed", "failure", "exception", "timeout", "timed_out",
    "blocked", "invalid", "crashed",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _map(value: Any) -> bool:
    return isinstance(value, Mapping)


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer")
    return result


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{field} must be finite and in [0,1]")
    return result


def _budget(value: Any) -> dict[str, int]:
    if not _map(value):
        raise ValueError("budget is not an object")
    return {
        field: _int(value[field], field)
        for field in ("tail_horizon", "action_budget", "policy_call_cap")
    }


def _key(value: Mapping[str, Any]) -> tuple[int, int, int]:
    return tuple(int(value[field]) for field in ("tail_horizon", "action_budget", "policy_call_cap"))


def _label(value: Mapping[str, Any]) -> str:
    return "/".join(
        f"{field}={int(value[field])}"
        for field in ("tail_horizon", "action_budget", "policy_call_cap")
    )


def _nonempty(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (str, bytes)):
        return bool(value.strip()) and value.strip().lower() not in {b"none", b"null", b"false", b"0", "none", "null", "false", "0"}
    if isinstance(value, (Mapping, Sequence)):
        return len(value) > 0
    return bool(value)


def _error_status(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in _ERROR_STATUSES:
        return True
    return text.startswith((
        "error", "failed", "failure", "blocked", "exception",
        "timeout", "timed_out", "invalid", "crash",
    ))


def _records(summary: Mapping[str, Any]) -> list[Any]:
    value = summary.get("grid", summary.get("budgets"))
    if isinstance(value, Mapping):
        return list(value.values())
    return list(value) if isinstance(value, list) else []


def _summary_failures(summary: Mapping[str, Any], *, expected_phase: str | None = None) -> list[str]:
    failures: list[str] = []
    status = summary.get("status")
    if summary.get("blocked") is True:
        failures.append("summary.blocked=true")
    if status not in {"COMPLETE", "COMPLETE_AVAILABLE_REQUESTS_SAMPLE_SHORTFALL"}:
        failures.append(f"summary.status is not final numerical status: {status!r}")
    phase = summary.get("phase")
    # The committed Phase-1 consolidator historically omitted this redundant
    # field while binding the same identity through the phase1/summary.json
    # path. Accept that one path-bound omission; an explicit wrong phase, an
    # unbound summary, or any other missing identity remains a hard failure.
    if phase is None:
        if expected_phase != "phase1":
            failures.append(f"summary.phase must be phase1: {phase!r}")
    elif phase != "phase1":
        failures.append(f"summary.phase must be phase1: {phase!r}")
    if summary.get("analysis") not in (None, "grid"):
        failures.append(f"summary.analysis must be grid: {summary.get('analysis')!r}")
    if summary.get("split") != "calibration":
        failures.append(f"summary.split must be calibration: {summary.get('split')!r}")
    if summary.get("selection_source") != "calibration_only":
        failures.append("summary.selection_source must be exactly calibration_only")
    tasks = summary.get("tasks")
    if tasks is not None and (set(tasks) != set(TASKS) or len(tasks) != 2):
        failures.append("summary.tasks do not match the frozen two-task schema")
    operators = summary.get("operators")
    if operators is not None and set(operators) != set(OPERATORS):
        failures.append("summary.operators do not match the frozen four-arm schema")
    prefixes = summary.get("prefixes")
    if prefixes is not None:
        try:
            if tuple(int(item) for item in prefixes) != PHASE1_PREFIXES:
                failures.append("summary.prefixes do not match Phase-1 prefixes")
        except (TypeError, ValueError):
            failures.append("summary.prefixes are invalid")
    for name in (
        "unknown_errors", "errors", "missing_shards", "missing_fragments",
        "missing_grid_rows", "missing_reference_rows", "incomplete_shards",
    ):
        if _nonempty(summary.get(name)):
            failures.append(f"summary.{name} is non-empty")
    try:
        if _int(summary.get("excluded_error_row_count", 0), "excluded_error_row_count"):
            failures.append(
                f"summary.excluded_error_row_count={summary.get('excluded_error_row_count')}"
            )
    except (TypeError, ValueError) as exc:
        failures.append(f"invalid excluded_error_row_count: {exc}")
    if _nonempty(summary.get("blocking_reasons")):
        failures.append("summary.blocking_reasons is non-empty")
    for field in ("grid_row_count", "support_grid_row_count"):
        if field in summary:
            try:
                if _int(summary[field], field) <= 0:
                    failures.append(f"summary.{field} must be positive")
            except (TypeError, ValueError) as exc:
                failures.append(f"invalid {field}: {exc}")
    if "reference_row_count" in summary:
        try:
            if _int(summary["reference_row_count"], "reference_row_count") < 0:
                failures.append("summary.reference_row_count is negative")
        except (TypeError, ValueError) as exc:
            failures.append(f"invalid reference_row_count: {exc}")
    support = summary.get("support_event_count_by_task")
    if support is not None:
        if not _map(support):
            failures.append("support_event_count_by_task is not an object")
        else:
            for task in TASKS:
                try:
                    if _int(support.get(task), f"support event count for {task}") <= 0:
                        failures.append(f"no actual support events for {task}")
                except (TypeError, ValueError) as exc:
                    failures.append(str(exc))
    formal = summary.get("selection")
    if _map(formal):
        if formal.get("status") not in {"SELECTED", "NO_QUALIFYING_BUDGET", "BLOCKED"}:
            failures.append(f"unknown formal selection status: {formal.get('status')!r}")
        if _nonempty(formal.get("unknown_errors")):
            failures.append("formal selection has unknown errors")
    return _dedupe(failures)


def _diagnostic_prerequisites(summary: Mapping[str, Any]) -> list[str]:
    sample = summary.get("sample")
    sample = sample if _map(sample) else {}
    complete = summary.get("sample_complete", summary.get("planned_sample_complete"))
    if complete is None:
        complete = sample.get("sample_complete", sample.get("planned_sample_complete"))
    failures: list[str] = []
    if complete is False:
        failures.append(
            "planned calibration sample incomplete: "
            f"observed={summary.get('observed_events_by_task', sample.get('observed_events_by_task'))!r}, "
            f"planned={summary.get('planned_events', sample.get('planned_events', 20))!r}, "
            f"shortfall={summary.get('shortfall', sample.get('shortfall'))!r}"
        )
    excluded = summary.get(
        "structurally_excluded_event_count",
        sample.get("structurally_excluded_event_count", 0),
    )
    try:
        excluded = _int(excluded or 0, "structurally_excluded_event_count")
        if excluded < 0:
            raise ValueError("negative structural exclusion count")
    except (TypeError, ValueError) as exc:
        failures.append(f"invalid structurally_excluded_event_count: {exc}")
    else:
        if excluded:
            failures.append(
                "structurally excluded source events are excluded from support: "
                f"count={excluded}"
            )
    return _dedupe(failures)


def _metric(
    record: Mapping[str, Any],
    task: str,
    support: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    by_task = record.get("by_task", record.get("task_metrics"))
    if not _map(by_task) or not _map(by_task.get(task)):
        return None, [f"{task} metrics are missing"]
    raw = by_task[task]
    failures: list[str] = []
    for owner, value in (("record", record), (f"{task} metrics", raw)):
        if value.get("blocked") is True:
            failures.append(f"{owner}.blocked=true")
        for field in ("unknown_errors", "errors", "missing_shards", "missing_fragments", "blocking_reasons"):
            if _nonempty(value.get(field)):
                failures.append(f"{owner}.{field} is non-empty")
        if "status" in value:
            raw_status = value.get("status")
            if str(raw_status).strip() != "COMPLETE":
                failures.append(f"{owner}.status must be explicit COMPLETE: {raw_status!r}")
            elif _error_status(raw_status):
                failures.append(f"{owner}.status is an error")
    completeness = raw.get("completeness")
    if not _map(completeness) or completeness.get("complete") is not True:
        failures.append(f"{task} completeness is not COMPLETE")
    elif "status" in completeness and completeness.get("status") != "COMPLETE":
        failures.append(
            f"{task}.completeness.status must be explicit COMPLETE: "
            f"{completeness.get('status')!r}"
        )
    try:
        event_count = _int(raw.get("event_count"), f"{task}.event_count")
        if event_count <= 0:
            failures.append(f"{task}.event_count must be positive")
    except (TypeError, ValueError) as exc:
        event_count = None
        failures.append(str(exc))
    support_count = event_count
    if support is not None:
        try:
            support_count = _int(support.get(task), f"support event count for {task}")
            if support_count <= 0:
                failures.append(f"{task} support event count must be positive")
            if event_count is not None and event_count != support_count:
                failures.append(
                    f"{task} event_count={event_count} disagrees with support={support_count}"
                )
        except (TypeError, ValueError) as exc:
            failures.append(str(exc))
    values: dict[str, float] = {}
    for name, aliases in (
        ("oracle_best", ("oracle_best", "oracle")),
        ("weakest_arm", ("weakest_arm",)),
        ("gap", ("oracle_minus_weakest", "gap")),
    ):
        value = next((raw.get(alias) for alias in aliases if alias in raw), None)
        try:
            values[name] = _number(value, f"{task}.{name}")
        except (TypeError, ValueError) as exc:
            failures.append(str(exc))
    if len(values) == 3:
        implied = values["oracle_best"] - values["weakest_arm"]
        if abs(values["gap"] - implied) > 1e-6:
            failures.append(f"{task} gap disagrees with oracle minus weakest")
    if failures:
        return None, _dedupe(failures)
    compact = {
        "oracle_best": values["oracle_best"],
        "weakest_arm": values["weakest_arm"],
        "oracle_minus_weakest": values["gap"],
        "gap": values["gap"],
        "event_count": event_count,
        "support_event_count": support_count,
        "completeness": {"status": completeness.get("status"), "complete": True},
    }
    return compact, []


def _k2_failures(by_task: Mapping[str, Mapping[str, Any]]) -> list[str]:
    failures: list[str] = []
    for task in TASKS:
        oracle = by_task[task]["oracle_best"]
        gap = by_task[task]["oracle_minus_weakest"]
        if not 0.25 <= oracle <= 0.85:
            failures.append(f"{task}: oracle_best={oracle} outside [0.25,0.85]")
        if gap < 0.15:
            failures.append(f"{task}: oracle_minus_weakest={gap} below 0.15")
    return failures


def _candidate_key(item: Mapping[str, Any]) -> tuple[float, float, int, int, int]:
    budget = item["budget"]
    return (
        -float(item["min_oracle"]),
        -float(item["min_gap"]),
        int(budget["action_budget"]),
        int(budget["tail_horizon"]),
        int(budget["policy_call_cap"]),
    )


def _candidates(summary: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    raw_records = _records(summary)
    failures: list[str] = []
    if len(raw_records) != 9:
        failures.append(f"expected exactly 9 Phase-1 budget records; observed {len(raw_records)}")
    support = summary.get("support_event_count_by_task")
    support = support if _map(support) else None
    seen: set[tuple[int, int, int]] = set()
    candidates: list[dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records):
        if not _map(raw_record):
            failures.append(f"grid record {index} is not an object")
            continue
        try:
            budget = _budget(raw_record.get("budget"))
        except (TypeError, ValueError, KeyError) as exc:
            failures.append(f"grid record {index} has invalid budget: {exc}")
            continue
        key = _key(budget)
        if key in seen:
            failures.append(f"duplicate budget record: {_label(budget)}")
            continue
        seen.add(key)
        if key not in BUDGET_KEYS:
            failures.append(f"budget outside frozen nine-cell grid: {budget}")
            continue
        by_task: dict[str, dict[str, Any]] = {}
        record_failures: list[str] = []
        for task in TASKS:
            value, reasons = _metric(raw_record, task, support)
            record_failures.extend(reasons)
            if value is not None:
                by_task[task] = value
        if record_failures:
            failures.extend(f"{_label(budget)}: {reason}" for reason in record_failures)
            continue
        min_oracle = min(item["oracle_best"] for item in by_task.values())
        min_gap = min(item["oracle_minus_weakest"] for item in by_task.values())
        k2 = _k2_failures(by_task)
        candidates.append({
            "budget": budget,
            "by_task": by_task,
            "min_oracle": float(min_oracle),
            "min_gap": float(min_gap),
            "k2_qualifies": not k2,
            "qualifies": not k2,
            "k2_failures": k2,
        })
    missing = BUDGET_KEYS - seen
    extra = seen - BUDGET_KEYS
    if missing:
        failures.append(
            "missing frozen budget records: " + ", ".join(
                _label(dict(zip(("tail_horizon", "action_budget", "policy_call_cap"), key)))
                for key in sorted(missing)
            )
        )
    if extra:
        failures.append(f"unexpected budget keys: {sorted(extra)!r}")
    candidates.sort(key=_candidate_key)
    for rank, candidate in enumerate(candidates, 1):
        candidate["rank"] = rank
        candidate["original_rank"] = rank
    return candidates, _dedupe(failures)


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX64.fullmatch(value))


def _base_receipt(
    summary_path: str | Path | None,
    summary_sha256: str,
    continuation_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "receipt_type": "DIAGNOSTIC_SELECTION_RECEIPT",
        "phase": "phase1",
        "analysis": "grid",
        "split": "calibration",
        "status": "BLOCKED",
        "blocked": True,
        "calibration_only": True,
        "confirmatory": False,
        "evaluation_read": False,
        "evaluation_authorized": False,
        "proof_written": False,
        "selection_source": "calibration_only",
        "input_summary_path": str(summary_path) if summary_path is not None else None,
        "input_summary_sha256": summary_sha256,
        "input_sha256": summary_sha256,
        "diagnostic_continuation_doc_sha256": continuation_sha256,
        "protocol_sha256": continuation_sha256,
        "diagnostic_only": True,
        "selection_authorized": False,
        "gitproof_path": None,
        "authorization_path": None,
        "selection_proof_path": None,
        "selection_authorization_path": None,
        "original_rank": None,
        "selection_rule": {
            "original_k2": {
                "requires_both_tasks": True,
                "oracle_best_interval": [0.25, 0.85],
                "minimum_oracle_minus_weakest": 0.15,
            },
            "ranking": [
                "min(task oracle_best) descending",
                "min(task oracle_minus_weakest) descending",
                "action_budget ascending",
                "tail_horizon ascending",
                "policy_call_cap ascending",
            ],
            "prefer_k2_qualifying_candidate": True,
            "fallback": "rank all valid candidates if no candidate passes original K2 numerical thresholds",
        },
        "prerequisite_failures": [],
        "all_candidates": [],
        "selected_budget": None,
        "selected_candidate": None,
        "k2_numeric_status": "NOT_EVALUATED",
        "k2_numeric_gate": {
            "status": "NOT_EVALUATED",
            "qualifying_candidate_count": 0,
        },
        "fallback_used": False,
    }


def build_diagnostic_selection_receipt(
    summary: Mapping[str, Any],
    *,
    input_summary_sha256: str,
    input_summary_path: str | Path | None = None,
    diagnostic_continuation_doc_sha256: str = DEFAULT_DIAGNOSTIC_CONTINUATION_DOC_SHA256,
) -> dict[str, Any]:
    receipt = _base_receipt(
        input_summary_path,
        input_summary_sha256,
        diagnostic_continuation_doc_sha256,
    )
    failures: list[str] = []
    if not _valid_sha(input_summary_sha256):
        failures.append("input_summary_sha256 must be a 64-character hexadecimal SHA-256")
    if not _valid_sha(diagnostic_continuation_doc_sha256):
        failures.append("diagnostic_continuation_doc_sha256 must be a 64-character hexadecimal SHA-256")
    if not _map(summary):
        receipt["prerequisite_failures"] = _dedupe(failures + ["Phase-1 summary must be a JSON object"])
        return receipt
    expected_phase = None
    if input_summary_path is not None:
        candidate_path = Path(input_summary_path)
        if candidate_path.name == "summary.json" and candidate_path.parent.name == "phase1":
            expected_phase = "phase1"
    failures.extend(_summary_failures(summary, expected_phase=expected_phase))
    candidates, candidate_failures = _candidates(summary)
    failures.extend(candidate_failures)
    receipt["all_candidates"] = candidates
    receipt["prerequisite_failures"] = _dedupe(
        failures + _diagnostic_prerequisites(summary)
    )
    hard_failures = [
        item for item in receipt["prerequisite_failures"]
        if not (
            item.startswith("planned calibration sample incomplete:")
            or item.startswith("structurally excluded source events")
        )
    ]
    if hard_failures or len(candidates) != 9:
        if len(candidates) != 9:
            receipt["prerequisite_failures"] = _dedupe(
                receipt["prerequisite_failures"]
                + ["all nine budgets do not have two-task valid actual support"]
            )
        return receipt
    qualifying = [item for item in candidates if item["k2_qualifies"]]
    if qualifying:
        selected = qualifying[0]
        receipt["k2_numeric_status"] = "PASS"
        receipt["k2_numeric_gate"] = {
            "status": "PASS",
            "qualifying_candidate_count": len(qualifying),
            "selected_candidate_rank": selected["rank"],
        }
    else:
        selected = candidates[0]
        receipt["k2_numeric_status"] = "FAIL"
        receipt["k2_numeric_gate"] = {
            "status": "FAIL",
            "qualifying_candidate_count": 0,
            "failure": "no valid budget passes both-task original K2 numerical thresholds",
        }
        receipt["fallback_used"] = True
        receipt["prerequisite_failures"] = _dedupe(
            receipt["prerequisite_failures"]
            + ["original K2 numerical thresholds fail for every valid budget; diagnostic fallback used"]
        )
    receipt["status"] = "DIAGNOSTIC_SELECTED"
    receipt["blocked"] = False
    receipt["selected_budget"] = dict(selected["budget"])
    receipt["original_rank"] = int(selected["original_rank"])
    receipt["selected_candidate"] = selected
    sample = summary.get("sample")
    sample = sample if _map(sample) else {}
    receipt["observed_events_by_task"] = summary.get(
        "observed_events_by_task", sample.get("observed_events_by_task")
    )
    receipt["support_event_count_by_task"] = summary.get("support_event_count_by_task")
    receipt["shortfall"] = summary.get("shortfall", sample.get("shortfall"))
    receipt["planned_events"] = summary.get("planned_events", sample.get("planned_events"))
    receipt["planned_sample_complete"] = summary.get("planned_sample_complete")
    receipt["sample_complete"] = summary.get("sample_complete")
    receipt["structurally_excluded_event_count"] = summary.get(
        "structurally_excluded_event_count",
        sample.get("structurally_excluded_event_count", 0),
    )
    receipt["formal_selection_status"] = (
        summary.get("selection", {}).get("status")
        if _map(summary.get("selection"))
        else None
    )
    return receipt


def select_diagnostic_budget(
    summary: Mapping[str, Any],
    *,
    input_summary_sha256: str,
    input_summary_path: str | Path | None = None,
    diagnostic_continuation_doc_sha256: str = DEFAULT_DIAGNOSTIC_CONTINUATION_DOC_SHA256,
) -> dict[str, Any]:
    """Compatibility API returning the diagnostic selection receipt."""
    return build_diagnostic_selection_receipt(
        summary,
        input_summary_sha256=input_summary_sha256,
        input_summary_path=input_summary_path,
        diagnostic_continuation_doc_sha256=diagnostic_continuation_doc_sha256,
    )


def write_diagnostic_selection_receipt(
    input_summary: str | Path,
    output: str | Path | None = None,
    *,
    diagnostic_continuation_doc_sha256: str = DEFAULT_DIAGNOSTIC_CONTINUATION_DOC_SHA256,
) -> dict[str, Any]:
    path = Path(input_summary)
    raw = path.read_bytes()
    try:
        summary = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Phase-1 summary JSON: {path}: {exc}") from exc
    receipt = build_diagnostic_selection_receipt(
        summary,
        input_summary_sha256=hashlib.sha256(raw).hexdigest(),
        input_summary_path=path,
        diagnostic_continuation_doc_sha256=diagnostic_continuation_doc_sha256,
    )
    destination = Path(output) if output is not None else path.parent / "diagnostic_selection_receipt.json"
    text = json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"immutable diagnostic receipt differs: {destination}")
    else:
        destination.write_text(text, encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-summary", "--phase1-summary", "--summary", "--input", dest="input_summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--diagnostic-continuation-doc-sha256",
        "--continuation-doc-sha256",
        dest="continuation_sha256",
        default=DEFAULT_DIAGNOSTIC_CONTINUATION_DOC_SHA256,
    )
    args = parser.parse_args(argv)
    try:
        receipt = write_diagnostic_selection_receipt(
            args.input_summary,
            args.output,
            diagnostic_continuation_doc_sha256=args.continuation_sha256,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0 if receipt["status"] == "DIAGNOSTIC_SELECTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
