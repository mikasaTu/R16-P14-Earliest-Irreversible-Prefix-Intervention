"""Diagnostic-only Phase-2 atlas consolidation.

This reader has a deliberately separate admission path from the formal
selection gate. It calls matrix.selected_diagnostic_budget before opening any
sealed evaluation source or Phase-2 shard. The resulting atlas is descriptive
(confirmatory=False); it never changes formal K1/K2/K3 receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import matrix
    from .consolidate import (
        BUDGETS, OPERATORS, PHASE2_PREFIXES, REFERENCE_OPERATORS,
        REFERENCE_PREFIXES, SEEDS, TASKS, TASK_HORIZONS, _branch_key,
        _canonical_row, _event_key, _read_json, _read_rows, _sha256,
        _write_json, _write_jsonl,
    )
    from .statistics import analyze_crossing
except ImportError:  # direct script invocation from the s1 directory
    import matrix  # type: ignore
    from consolidate import (  # type: ignore
        BUDGETS, OPERATORS, PHASE2_PREFIXES, REFERENCE_OPERATORS,
        REFERENCE_PREFIXES, SEEDS, TASKS, TASK_HORIZONS, _branch_key,
        _canonical_row, _event_key, _read_json, _read_rows, _sha256,
        _write_json, _write_jsonl,
    )
    from statistics import analyze_crossing  # type: ignore

DIAGNOSTIC_DOC_SHA256 = (
    "2d14427c2391422dd372b0528d6d784d1066d8d059e1405597cfb1954f260c6a"
)
EXPECTED_SOURCE_EVENTS = {
    TASKS[0]: {"calibration": 19, "evaluation": 23},
    TASKS[1]: {"calibration": 11, "evaluation": 13},
}
HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
DIAGNOSTIC_SHA_FIELDS = (
    "diagnostic_selection_receipt_sha256",
    "selection_receipt_sha256",
    "diagnostic_selection_sha256",
)


def _issue(reasons: list[str], message: str) -> None:
    if message not in reasons:
        reasons.append(message)


def _text(value: Any) -> str:
    return str(value)


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer")
    return result


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json_bytes(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), raw


def _budget(value: Mapping[str, Any]) -> dict[str, int]:
    return {
        name: _int(value[name], name)
        for name in ("tail_horizon", "action_budget", "policy_call_cap")
    }


def _budget_key(value: Mapping[str, Any]) -> tuple[int, int, int]:
    return tuple(value[name] for name in ("tail_horizon", "action_budget", "policy_call_cap"))


def _valid_sha(value: Any, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and bool(pattern.fullmatch(value))


def _source_key(value: Mapping[str, Any]) -> tuple[str, str, str, str, int]:
    seed = value.get("generator_actor_seed", value.get("actor_seed"))
    return (
        _text(value["task"]), _text(value["event_instance_id"]),
        _text(value["init_state_id"]), _text(value["split"]),
        _int(seed, "generator_actor_seed"),
    )


def _source_sort(key: tuple[str, str, str, str, int]) -> tuple[str, str, int, str, int]:
    return (key[0], key[3], int(key[2]), key[1], key[4])


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return _branch_key(row)


def _is_structural(row: Mapping[str, Any]) -> bool:
    return (
        _text(row.get("status")).strip().upper().startswith("BLOCKED")
        and _text(row.get("error_type")).strip() == "PrefixOutsideTaskHorizon"
    )


def _diagnostic_marker(
    row: Mapping[str, Any], selection_sha: str, label: str, reasons: list[str]
) -> None:
    if row.get("diagnostic_continuation") is not True:
        _issue(reasons, f"{label}: diagnostic_continuation marker is not true")
    actual = next(
        (row.get(name) for name in DIAGNOSTIC_SHA_FIELDS if row.get(name) is not None),
        None,
    )
    if actual != selection_sha:
        _issue(
            reasons,
            f"{label}: diagnostic selection SHA mismatch "
            f"(observed={actual!r}, expected={selection_sha})",
        )


def _admit_diagnostic_budget(
    input_root: Path, reasons: list[str]
) -> tuple[dict[str, int] | None, str | None, dict[str, Any]]:
    """Call the canonical runtime admission API before any evaluation read."""
    # Runtime briefly exposed the implementation spelling before the public
    # alias landed. Prefer the alias and retain the implementation fallback so
    # this diagnostic remains compatible with either checked-out runtime.
    admit = getattr(matrix, "selected_diagnostic_budget", None)
    if admit is None:
        admit = getattr(matrix, "diagnostic_selected_budget", None)
    if admit is None:
        _issue(reasons, "evaluation OPEN_DENY: diagnostic budget API is unavailable")
        return None, None, {}
    try:
        result = admit(input_root)
    except Exception as exc:  # canonical API fail-closed boundary
        _issue(reasons, f"evaluation OPEN_DENY: {exc}")
        return None, None, {}
    if not isinstance(result, Mapping):
        _issue(reasons, "evaluation OPEN_DENY: diagnostic admission returned no mapping")
        return None, None, {}
    raw_budget = result.get("selected_budget", result)
    if not isinstance(raw_budget, Mapping):
        _issue(reasons, "evaluation OPEN_DENY: diagnostic admission has no selected budget")
        return None, None, {}
    try:
        budget = _budget(raw_budget)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        _issue(reasons, f"evaluation OPEN_DENY: invalid admitted budget: {exc}")
        return None, None, {}
    if _budget_key(budget) not in BUDGETS:
        _issue(reasons, f"evaluation OPEN_DENY: budget outside frozen grid: {budget!r}")
        return None, None, {}

    receipt_path = input_root / "phase1" / "diagnostic_selection_receipt.json"
    if not receipt_path.is_file():
        _issue(reasons, f"evaluation OPEN_DENY: missing diagnostic selection receipt: {receipt_path}")
        return None, None, {}
    try:
        receipt, raw = _read_json_bytes(receipt_path)
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _issue(reasons, f"evaluation OPEN_DENY: invalid diagnostic selection receipt: {exc}")
        return None, None, {}
    if not isinstance(receipt, Mapping):
        _issue(reasons, "evaluation OPEN_DENY: diagnostic selection receipt is not an object")
        return None, None, {}
    selection_sha = _sha256_bytes(raw)
    if receipt.get("status") != "DIAGNOSTIC_SELECTED":
        _issue(reasons, f"evaluation OPEN_DENY: diagnostic receipt status={receipt.get('status')!r}")
    if receipt.get("selection_source") != "calibration_only":
        _issue(reasons, "evaluation OPEN_DENY: diagnostic receipt is not calibration-only")
    if receipt.get("calibration_only") is not True:
        _issue(reasons, "evaluation OPEN_DENY: diagnostic receipt lacks calibration_only marker")
    if receipt.get("confirmatory") is not False or receipt.get("diagnostic_only") is not True:
        _issue(reasons, "evaluation OPEN_DENY: diagnostic receipt is not marked non-confirmatory")
    if receipt.get("evaluation_read") is not False:
        _issue(reasons, "evaluation OPEN_DENY: diagnostic receipt already reports evaluation read")
    if receipt.get("diagnostic_continuation_doc_sha256") != DIAGNOSTIC_DOC_SHA256:
        _issue(reasons, "evaluation OPEN_DENY: diagnostic continuation protocol SHA mismatch")
    if receipt.get("protocol_sha256") != DIAGNOSTIC_DOC_SHA256:
        _issue(reasons, "evaluation OPEN_DENY: diagnostic protocol SHA mismatch")
    declared = receipt.get("selected_budget")
    try:
        if not isinstance(declared, Mapping) or _budget(declared) != budget:
            _issue(reasons, "evaluation OPEN_DENY: admitted budget disagrees with diagnostic receipt")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        _issue(reasons, f"evaluation OPEN_DENY: invalid receipt budget: {exc}")
    for name in ("selection_receipt_sha256", "diagnostic_selection_receipt_sha256"):
        value = result.get(name)
        if value is not None and value != selection_sha:
            _issue(reasons, f"evaluation OPEN_DENY: admission {name} does not match receipt")
    if reasons:
        return None, None, {}
    return budget, selection_sha, {
        "receipt_path": str(receipt_path),
        "receipt_sha256": selection_sha,
        "receipt": dict(receipt),
        "api_result": dict(result),
    }


def _qualification_rows(
    input_root: Path, reasons: list[str]
) -> dict[tuple[str, str, str, str, int], dict[str, Any]]:
    """Read qualification metadata after admission; it is source identity only."""
    qualified: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    for task in TASKS:
        directory = input_root / "phase0b" / "qualification" / task
        if not directory.is_dir():
            _issue(reasons, f"missing qualification directory: {directory}")
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                payload = _read_json(path)
                if not isinstance(payload, Mapping):
                    raise ValueError("row is not an object")
                if _text(payload.get("task")) != task:
                    raise ValueError("task/path mismatch")
                split = _text(payload.get("split"))
                if split not in ("calibration", "evaluation"):
                    continue
                if payload.get("status") != "COMPLETE":
                    raise ValueError("qualification row is not COMPLETE")
                if payload.get("qualified_natural_failure") is not True:
                    continue
                key = _source_key(payload)
                if key in qualified:
                    _issue(reasons, f"duplicate qualified source identity: {key!r}")
                    continue
                item = dict(payload)
                item["source_path"] = str(path)
                item["source_file_sha256"] = _sha256(path)
                qualified[key] = item
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                _issue(reasons, f"invalid qualification row {path}: {exc}")
    return qualified


def _event_from_payload(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = payload.get("event")
    return value if isinstance(value, Mapping) else payload


def _source_events(
    input_root: Path,
    qualified: Mapping[tuple[str, str, str, str, int], Mapping[str, Any]],
    reasons: list[str],
) -> dict[tuple[str, str, str, str, int], dict[str, Any]]:
    """Read calibration source plus sealed evaluation source only after proof."""
    source: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    calibration_path = input_root / "phase0b" / "calibration_events.jsonl"
    if not calibration_path.is_file():
        _issue(reasons, f"missing calibration source: {calibration_path}")
    else:
        try:
            for line_number, line in enumerate(
                calibration_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"line {line_number} is not an object")
                event = dict(_event_from_payload(payload))
                if _text(event.get("split")) != "calibration":
                    raise ValueError(f"line {line_number} is not calibration")
                key = _source_key(event)
                if key in source:
                    _issue(reasons, f"duplicate calibration source event: {key!r}")
                    continue
                event["source_path"] = str(calibration_path)
                event["source_file_sha256"] = _sha256(calibration_path)
                source[key] = event
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            _issue(reasons, f"invalid calibration source: {exc}")

    for task in TASKS:
        directory = input_root / "phase0b" / "sealed_evaluation" / task
        if not directory.is_dir():
            _issue(reasons, f"missing sealed evaluation source directory: {directory}")
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                payload = _read_json(path)
                if not isinstance(payload, Mapping):
                    raise ValueError("episode is not an object")
                event = dict(_event_from_payload(payload))
                if _text(event.get("split")) != "evaluation":
                    continue
                if _text(event.get("task")) != task:
                    raise ValueError("sealed evaluation task/path mismatch")
                key = _source_key(event)
                if key in source:
                    _issue(reasons, f"duplicate evaluation source event: {key!r}")
                    continue
                event["source_path"] = str(path)
                event["source_file_sha256"] = _sha256(path)
                source[key] = event
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                _issue(reasons, f"invalid sealed evaluation source {path}: {exc}")

    expected_keys = set(qualified)
    observed_keys = set(source)
    for key in sorted(expected_keys - observed_keys, key=_source_sort):
        _issue(reasons, f"qualified source event missing raw source: {key!r}")
    for key in sorted(observed_keys - expected_keys, key=_source_sort):
        _issue(reasons, f"raw source event is not a qualified event: {key!r}")
    for key, event in source.items():
        if key not in qualified:
            continue
        try:
            anchor = _int(event.get("anchor_global_step"), "anchor_global_step")
            if anchor < 0:
                raise ValueError("anchor_global_step must be non-negative")
            event["anchor_global_step"] = anchor
        except (TypeError, ValueError, OverflowError) as exc:
            _issue(reasons, f"source event {key!r} has invalid anchor_global_step: {exc}")
        fields = ("task", "event_instance_id", "init_state_id", "split")
        for pos, field in enumerate(fields):
            if _text(event.get(field)) != key[pos]:
                _issue(reasons, f"source event identity mismatch for {key!r}")
    counts = {
        task: {
            split: sum(key[0] == task and key[3] == split for key in source)
            for split in ("calibration", "evaluation")
        }
        for task in TASKS
    }
    if counts != EXPECTED_SOURCE_EVENTS:
        _issue(reasons, f"source event counts {counts!r} != {EXPECTED_SOURCE_EVENTS!r}")
    return source


def _expected_keys(
    source: Mapping[tuple[str, str, str, str, int], Mapping[str, Any]],
    budget: Mapping[str, Any],
) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]]]:
    core: set[tuple[Any, ...]] = set()
    reference: set[tuple[Any, ...]] = set()
    for key in source:
        for recovery_seed in SEEDS:
            for operator in OPERATORS:
                for prefix in PHASE2_PREFIXES:
                    core.add(
                        key + (
                            recovery_seed, operator, prefix,
                            budget["tail_horizon"], budget["action_budget"],
                            budget["policy_call_cap"],
                        )
                    )
            for operator, prefix in REFERENCE_PREFIXES.items():
                reference.add(
                    key + (
                        recovery_seed, operator, prefix,
                        budget["tail_horizon"], budget["action_budget"],
                        budget["policy_call_cap"],
                    )
                )
    return core, reference


def _structural_valid(
    row: Mapping[str, Any],
    source_event: Mapping[str, Any],
    reasons: list[str],
    label: str,
) -> bool:
    if not _is_structural(row):
        _issue(reasons, f"{label}: unknown/error row cannot count as failure")
        return False
    try:
        prefix = _int(row["prefix_k"], "prefix_k")
        anchor = _int(source_event["anchor_global_step"], "anchor_global_step")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        _issue(reasons, f"{label}: structural source anchor/prefix unavailable: {exc}")
        return False
    horizon = TASK_HORIZONS[str(source_event["task"])]
    if anchor < 0 or anchor >= horizon:
        _issue(
            reasons,
            f"{label}: source anchor_global_step={anchor} outside [0, {horizon})",
        )
        return False
    row_anchor = row.get("anchor_global_step")
    if row_anchor is not None:
        try:
            if _int(row_anchor, "row anchor_global_step") != anchor:
                raise ValueError("row anchor_global_step disagrees with source")
        except (TypeError, ValueError, OverflowError) as exc:
            _issue(reasons, f"{label}: {exc}")
            return False
    if anchor + prefix <= horizon:
        _issue(
            reasons,
            f"{label}: structural row is feasible: anchor={anchor}+prefix={prefix}<=horizon={horizon}",
        )
        return False
    return True


def _structural_row(
    raw: Mapping[str, Any],
    source_event: Mapping[str, Any],
    selection_sha: str,
    reasons: list[str],
    label: str,
) -> dict[str, Any] | None:
    _diagnostic_marker(raw, selection_sha, label, reasons)
    if not _is_structural(raw):
        _issue(reasons, f"{label}: non-structural row is not COMPLETE")
        return None
    result = {key: value for key, value in raw.items() if not str(key).startswith("__")}
    for field in (
        "task", "event_instance_id", "init_state_id", "split",
        "generator_actor_seed", "recovery_actor_seed", "operator",
        "prefix_k", "tail_horizon", "action_budget", "policy_call_cap",
        "status", "error_type",
    ):
        if field not in result:
            _issue(reasons, f"{label}: structural row missing {field}")
            return None
    try:
        result["task"] = _text(result["task"])
        result["event_instance_id"] = _text(result["event_instance_id"])
        result["init_state_id"] = _text(result["init_state_id"])
        result["split"] = _text(result["split"])
        result["generator_actor_seed"] = _int(result["generator_actor_seed"], "generator_actor_seed")
        result["recovery_actor_seed"] = _int(result["recovery_actor_seed"], "recovery_actor_seed")
        result["prefix_k"] = _int(result["prefix_k"], "prefix_k")
        for name in ("tail_horizon", "action_budget", "policy_call_cap"):
            result[name] = _int(result[name], name)
    except (TypeError, ValueError, OverflowError) as exc:
        _issue(reasons, f"{label}: invalid structural branch key: {exc}")
        return None
    operator = result["operator"] = _text(result["operator"])
    expected_reference = operator in REFERENCE_OPERATORS
    if not isinstance(result.get("is_reference"), bool):
        result["is_reference"] = expected_reference
    if bool(result["is_reference"]) != expected_reference:
        _issue(reasons, f"{label}: reference/operator mismatch")
    if result["split"] not in ("calibration", "evaluation"):
        _issue(reasons, f"{label}: invalid split")
    if not _structural_valid(result, source_event, reasons, label):
        return None
    result["structurally_excluded"] = True
    result["structural_exclusion_reason"] = "PrefixOutsideTaskHorizon"
    return result


def _runtime_provenance(
    row: Mapping[str, Any], structural: bool, selection_sha: str,
    reasons: list[str], label: str,
) -> None:
    _diagnostic_marker(row, selection_sha, label, reasons)
    if row.get("source_commit") is not None and not _valid_sha(row.get("source_commit"), HEX40):
        _issue(reasons, f"{label}: invalid source_commit")
    if row.get("runtime_receipt_sha256") is not None and not _valid_sha(
        row.get("runtime_receipt_sha256"), HEX64
    ):
        _issue(reasons, f"{label}: invalid runtime_receipt_sha256")
    if structural:
        return
    for name in ("source_commit", "runtime_receipt_sha256", "job_id", "pai_run_id"):
        if row.get(name) in (None, ""):
            _issue(reasons, f"{label}: missing {name}")
    if not _valid_sha(row.get("runtime_receipt_sha256"), HEX64):
        _issue(reasons, f"{label}: runtime receipt is not a SHA-256")
    if not _valid_sha(row.get("source_commit"), HEX40):
        _issue(reasons, f"{label}: source_commit is not a commit SHA")
    if not isinstance(row.get("pid"), int) or isinstance(row.get("pid"), bool) or row.get("pid") <= 0:
        _issue(reasons, f"{label}: invalid pid")
    for name in ("env_hash", "chunk_hash"):
        if row.get(name) in (None, ""):
            _issue(reasons, f"{label}: missing {name}")


def _normalize_phase2_rows(
    input_root: Path,
    source: Mapping[tuple[str, str, str, str, int], Mapping[str, Any]],
    budget: Mapping[str, Any],
    selection_sha: str,
    reasons: list[str],
) -> tuple[list[dict[str, Any]], set[tuple[str, str, str, str, int]]]:
    raw: list[dict[str, Any]] = []
    for task in TASKS:
        material, read_reasons = _read_rows(input_root / "phase2" / "shards" / task)
        reasons.extend(read_reasons)
        raw.extend(material)
    normalized: list[dict[str, Any]] = []
    structural_events: set[tuple[str, str, str, str, int]] = set()
    seen: set[tuple[Any, ...]] = set()
    for index, item in enumerate(raw):
        label = _text(item.get("__source_path", f"row-{index}"))
        source_path = item.get("__source_path")
        try:
            if source_path is not None and Path(source_path).parent.name not in TASKS:
                _issue(reasons, f"{label}: shard path is not task-scoped")
                continue
            if source_path is not None and Path(source_path).parent.name != _text(item.get("task")):
                _issue(reasons, f"{label}: shard task/path mismatch")
                continue
            if _is_structural(item):
                source_key = (
                    _text(item.get("task")), _text(item.get("event_instance_id")),
                    _text(item.get("init_state_id")), _text(item.get("split")),
                    _int(item.get("generator_actor_seed"), "generator_actor_seed"),
                )
                source_event = source.get(source_key)
                if source_event is None:
                    _issue(reasons, f"{label}: structural row has unknown source event")
                    continue
                row = _structural_row(item, source_event, selection_sha, reasons, label)
            else:
                row, row_reasons = _canonical_row(item, "phase2")
                if row is None:
                    for reason in row_reasons:
                        _issue(reasons, f"{label}: {reason}")
                    continue
                row = dict(row)
                row["is_reference"] = bool(
                    item.get("is_reference", row["operator"] in REFERENCE_OPERATORS)
                )
                _runtime_provenance(row, False, selection_sha, reasons, label)
                source_key = (
                    _text(row.get("task")), _text(row.get("event_instance_id")),
                    _text(row.get("init_state_id")), _text(row.get("split")),
                    _int(row.get("generator_actor_seed"), "generator_actor_seed"),
                )
                source_event = source.get(source_key)
                if source_event is None:
                    _issue(reasons, f"{label}: COMPLETE row has unknown source event")
                    continue
            if row is None:
                continue
            key = _row_key(row)
            if key in seen:
                _issue(reasons, f"exact duplicate branch key rejected: {key!r}")
                continue
            seen.add(key)
            if row["task"] not in TASKS:
                _issue(reasons, f"{label}: unknown task")
            if row["split"] not in ("calibration", "evaluation"):
                _issue(reasons, f"{label}: invalid diagnostic split")
            if row["generator_actor_seed"] != source_key[4]:
                _issue(reasons, f"{label}: generator actor seed mismatch")
            if _budget_key(row) != _budget_key(budget):
                _issue(reasons, f"{label}: selected budget mismatch")
            configured = row.get("configured_budget")
            if configured is not None and isinstance(configured, Mapping):
                try:
                    if _budget(configured) != budget:
                        _issue(reasons, f"{label}: configured_budget mismatch")
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    _issue(reasons, f"{label}: invalid configured_budget: {exc}")
            operator = _text(row["operator"])
            expected_reference = operator in REFERENCE_OPERATORS
            if operator not in OPERATORS + REFERENCE_OPERATORS:
                _issue(reasons, f"{label}: unknown operator {operator!r}")
            if bool(row.get("is_reference")) != expected_reference:
                _issue(reasons, f"{label}: reference/operator mismatch")
            if expected_reference and row["prefix_k"] != REFERENCE_PREFIXES[operator]:
                _issue(reasons, f"{label}: reference prefix mismatch")
            if not expected_reference and row["prefix_k"] not in PHASE2_PREFIXES:
                _issue(reasons, f"{label}: core prefix outside Phase-2 atlas")
            if source_event.get("anchor_global_step") is not None and row.get("anchor_global_step") is not None:
                if _int(row["anchor_global_step"], "anchor_global_step") != int(source_event["anchor_global_step"]):
                    _issue(reasons, f"{label}: anchor_global_step mismatch")
            if _is_structural(row):
                structural_events.add(source_key)
            normalized.append(row)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            _issue(reasons, f"{label}: invalid Phase-2 row: {exc}")
    return normalized, structural_events


def _check_grid(
    rows: Sequence[Mapping[str, Any]],
    source: Mapping[tuple[str, str, str, str, int], Mapping[str, Any]],
    budget: Mapping[str, Any],
    structural_events: set[tuple[str, str, str, str, int]],
    reasons: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    expected_core, expected_reference = _expected_keys(source, budget)
    expected = expected_core | expected_reference
    observed = {_row_key(row) for row in rows}
    missing = expected - observed
    extra = observed - expected
    if missing:
        _issue(reasons, f"missing expected Phase-2 branch keys: {len(missing)}")
    if extra:
        _issue(reasons, f"unexpected Phase-2 branch keys: {len(extra)}")
    for row in rows:
        source_key = (
            _text(row["task"]), _text(row["event_instance_id"]),
            _text(row["init_state_id"]), _text(row["split"]),
            _int(row["generator_actor_seed"], "generator_actor_seed")
        )
        event = source.get(source_key)
        if event is None:
            continue
        outside = int(event["anchor_global_step"]) + int(row["prefix_k"]) > TASK_HORIZONS[row["task"]]
        structural = bool(row.get("structurally_excluded"))
        if outside and not structural:
            _issue(reasons, f"branch should be structural outside horizon: {_row_key(row)!r}")
        if not outside and structural:
            _issue(reasons, f"feasible branch marked structural: {_row_key(row)!r}")
    # _event_key is the five-field source identity, including the generator
    # actor seed.  Keep that exact key so one structural branch excludes the
    # entire event cluster from support without accidentally retaining siblings.
    excluded_keys = set(structural_events)
    support = [
        dict(row) for row in rows
        if not row.get("structurally_excluded") and _event_key(row) not in excluded_keys
    ]
    core = [dict(row) for row in rows if not bool(row.get("is_reference"))]
    reference = [dict(row) for row in rows if bool(row.get("is_reference"))]
    excluded_info = [
        {
            "task": key[0], "event_instance_id": key[1], "init_state_id": key[2],
            "split": key[3], "generator_actor_seed": key[4],
            "reason": "PrefixOutsideTaskHorizon",
        }
        for key in sorted(excluded_keys, key=_source_sort)
    ]
    return core, reference, support, excluded_info


def _completion_metadata(
    input_root: Path,
    source: Mapping[tuple[str, str, str, str, int], Mapping[str, Any]],
    selection_sha: str,
    reasons: list[str],
) -> list[dict[str, Any]]:
    """Require both atlas task completion receipts before computing statistics."""
    result: list[dict[str, Any]] = []
    expected_events = {
        task: sum(key[0] == task for key in source)
        for task in TASKS
    }
    expected_rows = {task: count * (len(SEEDS) * (len(OPERATORS) * len(PHASE2_PREFIXES) + len(REFERENCE_OPERATORS)))
                     for task, count in expected_events.items()}
    allowed_statuses = {"COMPLETE", "COMPLETE_WITH_BLOCKED_CONTRACT_ROWS"}
    for task in TASKS:
        path = input_root / "phase2" / f"completion_{task}.json"
        if not path.is_file():
            _issue(reasons, f"missing diagnostic completion receipt: {path}")
            continue
        try:
            payload = _read_json(path)
            if not isinstance(payload, Mapping):
                raise ValueError("completion receipt is not an object")
            if _text(payload.get("task")) != task or _text(payload.get("phase")) != "atlas":
                raise ValueError("completion task/phase binding mismatch")
            if _text(payload.get("status")) not in allowed_statuses:
                raise ValueError(f"completion status is not terminal-success: {payload.get('status')!r}")
            if payload.get("diagnostic_continuation") is not True:
                raise ValueError("completion lacks diagnostic_continuation marker")
            if payload.get("diagnostic_selection_receipt_sha256") != selection_sha:
                raise ValueError("completion diagnostic selection SHA mismatch")
            if _int(payload.get("events"), "events") != expected_events[task]:
                raise ValueError(
                    f"completion event count {_int(payload.get('events'), 'events')} != {expected_events[task]}"
                )
            if _int(payload.get("requested_rows"), "requested_rows") != expected_rows[task]:
                raise ValueError(
                    f"completion requested row count {_int(payload.get('requested_rows'), 'requested_rows')} != {expected_rows[task]}"
                )
            if _int(payload.get("persisted_rows"), "persisted_rows") != expected_rows[task]:
                raise ValueError(
                    f"completion persisted row count {_int(payload.get('persisted_rows'), 'persisted_rows')} != {expected_rows[task]}"
                )
            result.append({key: value for key, value in payload.items()})
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            _issue(reasons, f"invalid diagnostic completion receipt {path}: {exc}")
    return result


def _source_reference_rows(
    source: Mapping[tuple[str, str, str, str, int], Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result = []
    for key in sorted(source, key=_source_sort):
        event = source[key]
        result.append({
            "task": key[0], "event_instance_id": key[1], "init_state_id": key[2],
            "split": key[3], "generator_actor_seed": key[4],
            "anchor_global_step": event.get("anchor_global_step"),
            "source_path": event.get("source_path"),
            "source_file_sha256": event.get("source_file_sha256"),
        })
    return result


def _upstream_k_failures(input_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "K1": {"status": "UNKNOWN"},
        "K2": {"status": "UNKNOWN"},
        "K3": {"status": "NOT_EVALUATED_FORMAL"},
    }
    for name, key in (("K1", "phase0b/summary.json"), ("K2", "phase1/summary.json")):
        path = input_root / key
        if not path.is_file():
            result[name] = {"status": "MISSING", "path": str(path)}
            continue
        try:
            payload = _read_json(path)
            if not isinstance(payload, Mapping):
                raise ValueError("summary is not an object")
            if name == "K1":
                gate = payload.get("gate_status")
                if gate is None:
                    gate = "PASS" if (payload.get("k1") or {}).get("pass") is True else payload.get("status")
                result[name] = {"status": gate, "path": str(path)}
            else:
                selection = payload.get("selection")
                if isinstance(selection, Mapping):
                    result[name] = {
                        "status": selection.get("status"),
                        "selected_budget": selection.get("selected_budget"),
                        "path": str(path),
                    }
                else:
                    result[name] = {"status": payload.get("status"), "path": str(path)}
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            result[name] = {"status": "INVALID", "path": str(path), "error": str(exc)}
    return result


def _run_statistics(
    rows: Sequence[Mapping[str, Any]], split: str, reasons: list[str]
) -> dict[str, Any]:
    selected = [row for row in rows if _text(row.get("split")) == split]
    try:
        result = analyze_crossing(selected, split=split, replicates=10000, seed=216214)
    except (TypeError, ValueError, KeyError) as exc:
        _issue(reasons, f"{split} crossing statistics failed closed: {exc}")
        return {}
    if result.get("status") != "COMPLETE":
        _issue(
            reasons,
            f"{split} crossing statistics blocked: "
            + "; ".join(str(x) for x in result.get("blocking_reasons", ())),
        )
    return result


def _diagnostic_k3(result: Mapping[str, Any]) -> dict[str, Any]:
    raw = result.get("k3")
    if not isinstance(raw, Mapping):
        return {"status": "NOT_EVALUATED", "decision": "STATISTICS_UNAVAILABLE"}
    by_task = raw.get("by_task")
    insufficient = False
    if isinstance(by_task, Mapping):
        for metrics in by_task.values():
            checks = metrics.get("checks") if isinstance(metrics, Mapping) else None
            if not isinstance(checks, Mapping) or checks.get("both_defined_at_least_30") is not True:
                insufficient = True
                break
    if insufficient:
        return {
            "status": "INSUFFICIENT_SUPPORT",
            "decision": "NOT_ESTABLISHED_LOW_SUPPORT",
            "raw_k3": dict(raw),
        }
    return dict(raw)


def _write_blocked(
    output_root: Path,
    reasons: Sequence[str],
    *,
    evaluation_read: bool,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": 1, "phase": "phase2", "status": "BLOCKED",
        "blocked": True, "confirmatory": False, "diagnostic_continuation": True,
        "evaluation_read": bool(evaluation_read),
        "blocking_reasons": list(dict.fromkeys(str(x) for x in reasons))[:200]
        or ["unspecified blocker"],
    }
    if extra:
        summary.update(dict(extra))
    _write_json(output_root / "phase2" / "summary.json", summary)
    return summary


def consolidate_diagnostic_phase2(
    input_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    input_path, output_path = Path(input_root).resolve(), Path(output_root).resolve()
    admission_reasons: list[str] = []
    budget, selection_sha, admission = _admit_diagnostic_budget(input_path, admission_reasons)
    if admission_reasons or budget is None or selection_sha is None:
        return _write_blocked(output_path, admission_reasons, evaluation_read=False)
    reasons: list[str] = []
    qualified = _qualification_rows(input_path, reasons)
    source = _source_events(input_path, qualified, reasons)
    normalized, structural_events = _normalize_phase2_rows(
        input_path, source, budget, selection_sha, reasons
    )
    core, reference, support, excluded = _check_grid(
        normalized, source, budget, structural_events, reasons
    )
    completions = _completion_metadata(input_path, source, selection_sha, reasons)
    extra = {
        "selection_receipt_sha256": selection_sha, "selected_budget": budget,
        "source_event_references": _source_reference_rows(source),
        "source_event_counts": {
            task: {
                split: sum(key[0] == task and key[3] == split for key in source)
                for split in ("calibration", "evaluation")
            }
            for task in TASKS
        },
        "row_counts": {
            "atlas_rows": len(core), "reference_rows": len(reference),
            "support_rows": len(support),
            "structural_rows": sum(bool(row.get("structurally_excluded")) for row in normalized),
        },
        "excluded_events": excluded, "evaluation_read": True,
        "admission": admission,
        "completion_receipts": completions,
        "upstream_k_failures": _upstream_k_failures(input_path),
    }
    if reasons:
        return _write_blocked(output_path, reasons, evaluation_read=True, extra=extra)
    _write_jsonl(output_path / "phase2" / "atlas_rows.jsonl", core)
    _write_jsonl(output_path / "phase2" / "reference_rows.jsonl", reference)
    stats_reasons: list[str] = []
    evaluation_stats = _run_statistics(support, "evaluation", stats_reasons)
    calibration_stats = _run_statistics(support, "calibration", stats_reasons)
    if stats_reasons or not evaluation_stats or not calibration_stats:
        return _write_blocked(
            output_path, stats_reasons or ["statistics unavailable"],
            evaluation_read=True, extra=extra,
        )
    boundary_rows: list[dict[str, Any]] = []
    for split, result in (("evaluation", evaluation_stats), ("calibration", calibration_stats)):
        for row in result.get("boundaries", ()):
            boundary_rows.append({"split": split, **dict(row)})
    _write_jsonl(output_path / "phase2" / "boundaries.jsonl", boundary_rows)
    _write_json(
        output_path / "phase2" / "crossing.json",
        {
            "schema_version": 1, "primary_split": "evaluation",
            "evaluation": evaluation_stats.get("crossing", {}),
            "calibration_descriptive": calibration_stats.get("crossing", {}),
            "confirmatory": False, "diagnostic_continuation": True,
        },
    )
    _write_json(
        output_path / "phase2" / "null_distribution.json",
        {
            "schema_version": 1, "primary_split": "evaluation",
            "evaluation": evaluation_stats.get("split_half_null", {}),
            "calibration_descriptive": calibration_stats.get("split_half_null", {}),
            "bootstrap_replicates": 10000, "bootstrap_seed": 216214,
            "confirmatory": False, "diagnostic_continuation": True,
        },
    )
    upstream = _upstream_k_failures(input_path)
    summary = {
        "schema_version": 1, "phase": "phase2", "status": "COMPLETE", "blocked": False,
        "confirmatory": False, "diagnostic_continuation": True,
        "selection_receipt_sha256": selection_sha, "selected_budget": budget,
        "evaluation_read": True, "source_event_counts": extra["source_event_counts"],
        "source_event_references": extra["source_event_references"],
        "row_counts": extra["row_counts"], "excluded_events": excluded,
        "excluded_event_count_by_task_split": {
            task: {
                split: sum(item.get("task") == task and item.get("split") == split for item in excluded)
                for split in ("calibration", "evaluation")
            }
            for task in TASKS
        },
        "support_event_count_by_task_split": {
            task: {
                split: len({
                    _event_key(row) for row in support
                    if row.get("task") == task and row.get("split") == split
                })
                for split in ("calibration", "evaluation")
            }
            for task in TASKS
        },
        "upstream_k_failures": upstream,
        "statistics": {
            "evaluation": {
                "status": evaluation_stats.get("status"),
                "row_count": evaluation_stats.get("row_count"),
                "both_defined_by_task": {
                    task: evaluation_stats.get("crossing", {}).get("by_task", {}).get(task, {}).get("both_defined")
                    for task in TASKS
                },
                "k3": _diagnostic_k3(evaluation_stats),
            },
            "calibration_descriptive": {
                "status": calibration_stats.get("status"),
                "row_count": calibration_stats.get("row_count"),
                "both_defined_by_task": {
                    task: calibration_stats.get("crossing", {}).get("by_task", {}).get(task, {}).get("both_defined")
                    for task in TASKS
                },
                "k3": _diagnostic_k3(calibration_stats),
            },
        },
        "primary_split": "evaluation", "calibration_split_is_descriptive_only": True,
        "bootstrap_replicates": 10000, "bootstrap_seed": 216214,
    }
    _write_json(output_path / "phase2" / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", "--input", dest="input_root", type=Path, required=True)
    parser.add_argument("--output-root", "--output", dest="output_root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = consolidate_diagnostic_phase2(args.input_root, args.output_root)
    print(json.dumps({"phase": "phase2", "status": result.get("status"), "output_root": str(args.output_root)}, sort_keys=True))
    return 0 if result.get("status") != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
