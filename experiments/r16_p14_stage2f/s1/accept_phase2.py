"""Fail-closed acceptance for the diagnostic Stage-2F Phase-2 atlas.

The checker is intentionally separate from the Phase-2 consolidator.  It
validates the complete source/terminal/shard/trace chain after the diagnostic
selection proof has admitted the run.  After admission it reads sealed raw
evaluation records and Phase-2 traces only for provenance and integrity checks;
it does not compute statistics or make a selection, and it does not write any
artifact below the input root.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # package invocation
    from . import matrix
    from .accept_phase1 import (
        TASKS,
        TASK_HORIZONS,
        SEEDS,
        OPERATORS,
        REFERENCE_PREFIXES,
        _validate_trace_files,
        _validate_trace_workers,
    )
except ImportError:  # direct invocation from the s1 directory
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from s1 import matrix  # type: ignore
    from s1.accept_phase1 import (  # type: ignore
        TASKS,
        TASK_HORIZONS,
        SEEDS,
        OPERATORS,
        REFERENCE_PREFIXES,
        _validate_trace_files,
        _validate_trace_workers,
    )

PHASE2_PREFIXES = (2, 4, 6, 8, 10, 12, 14, 16)
TAILS = (4, 8, 16)
ACTION_BUDGETS = (8, 16, 32)
POLICY_CALL_CAP = 8
BUDGETS = tuple(
    (tail, action, POLICY_CALL_CAP)
    for tail in TAILS
    for action in ACTION_BUDGETS
)
REFERENCE_OPERATORS = tuple(REFERENCE_PREFIXES)
PHASE1_SOURCE_COMMIT = "a0888d751117cbf7c5a73080d1ee1f421689e8bf"
PHASE1_SOURCE_COMMITS = frozenset({
    PHASE1_SOURCE_COMMIT,
    "93872b41ad48d24e1c6cb46d8c46690dae359b62",
})
DEFAULT_PHASE2_SOURCE_COMMITS = frozenset(
    {"4bc5f5a840d1fcafa76b4a6db8343230ab8ff184"}
)
DIAGNOSTIC_DOC_SHA256 = (
    "2d14427c2391422dd372b0528d6d784d1066d8d059e1405597cfb1954f260c6a"
)
DIAGNOSTIC_RECEIPT_REL = "phase1/diagnostic_selection_receipt.json"
UID_GID = 2254
HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
SOURCE_COUNTS = {
    TASKS[0]: {"calibration": 19, "evaluation": 23},
    TASKS[1]: {"calibration": 11, "evaluation": 13},
}
ALLOWED_REUSE_FIELDS = frozenset(
    {
        "reused_calibration_shard",
        "reused_calibration_sha256",
        "diagnostic_continuation",
        "diagnostic_selection_receipt_sha256",
        "diagnostic_selection_receipt_path",
        "diagnostic_selection_binding",
    }
)
DIAGNOSTIC_SHA_FIELDS = (
    "diagnostic_selection_receipt_sha256",
    "selection_receipt_sha256",
    "diagnostic_selection_sha256",
)
TERMINAL_COMPLETION_STATUSES = frozenset(
    {"COMPLETE", "COMPLETE_WITH_BLOCKED_CONTRACT_ROWS"}
)


def _issue(reasons: list[str], message: str) -> None:
    message = str(message)
    if message not in reasons:
        reasons.append(message)


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer")
    return result


def _strict_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _read_json(path: Path) -> Any:
    return json.loads(
        path.read_bytes().decode("utf-8"),
        parse_constant=_strict_constant,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _valid_sha(value: Any, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and bool(pattern.fullmatch(value))


def _text(value: Any) -> str:
    return str(value)


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _check_owned(path: Path, reasons: list[str], label: str) -> None:
    try:
        stat = path.stat()
    except OSError as exc:
        _issue(reasons, f"{label}: cannot stat: {exc}")
        return
    if stat.st_uid != UID_GID or stat.st_gid != UID_GID:
        _issue(
            reasons,
            f"{label}: ownership {stat.st_uid}:{stat.st_gid} != {UID_GID}:{UID_GID}",
        )


def _source_key(value: Mapping[str, Any]) -> tuple[str, str, str, str, int]:
    seed = value.get("generator_actor_seed", value.get("actor_seed"))
    return (
        _text(value["task"]),
        _text(value["event_instance_id"]),
        _text(value["init_state_id"]),
        _text(value["split"]),
        _int(seed, "generator_actor_seed"),
    )


def _source_sort(key: tuple[str, str, str, str, int]) -> tuple[str, str, int, str, int]:
    try:
        init = int(key[2])
    except (TypeError, ValueError):
        init = 0
    return (key[0], key[3], init, key[1], key[4])


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _text(row["task"]),
        _text(row["event_instance_id"]),
        _text(row["init_state_id"]),
        _text(row["split"]),
        _int(row["generator_actor_seed"], "generator_actor_seed"),
        _int(row["recovery_actor_seed"], "recovery_actor_seed"),
        _text(row["operator"]),
        _int(row["prefix_k"], "prefix_k"),
        _int(row["tail_horizon"], "tail_horizon"),
        _int(row["action_budget"], "action_budget"),
        _int(row["policy_call_cap"], "policy_call_cap"),
    )


def _source_key_from_row(row: Mapping[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        _text(row["task"]),
        _text(row["event_instance_id"]),
        _text(row["init_state_id"]),
        _text(row["split"]),
        _int(row["generator_actor_seed"], "generator_actor_seed"),
    )


def _budget(value: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("budget must be an object")
    return {
        name: _int(value[name], name)
        for name in ("tail_horizon", "action_budget", "policy_call_cap")
    }


def _budget_key(value: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        _int(value["tail_horizon"], "tail_horizon"),
        _int(value["action_budget"], "action_budget"),
        _int(value["policy_call_cap"], "policy_call_cap"),
    )


def _all_expected_keys(
    source: Mapping[tuple[str, str, str, str, int], Mapping[str, Any]],
    budget: Mapping[str, Any],
) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]]]:
    budget_key = _budget_key(budget)
    core: set[tuple[Any, ...]] = set()
    reference: set[tuple[Any, ...]] = set()
    for source_key in source:
        for recovery_seed in SEEDS:
            for operator in OPERATORS:
                for prefix in PHASE2_PREFIXES:
                    core.add(source_key + (recovery_seed, operator, prefix) + budget_key)
            for operator, prefix in REFERENCE_PREFIXES.items():
                reference.add(source_key + (recovery_seed, operator, prefix) + budget_key)
    return core, reference


def _expected_keys(
    source: Mapping[tuple[str, str, str, str, int], Mapping[str, Any]],
    budget: Mapping[str, Any],
) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]]]:
    """Public-compatible alias returning core and reference expected keys."""
    return _all_expected_keys(source, budget)


def _is_structural(row: Mapping[str, Any]) -> bool:
    status = _text(row.get("status")).strip().upper()
    return status == "BLOCKED" and _text(row.get("error_type")).strip() == (
        "PrefixOutsideTaskHorizon"
    )


def _diagnostic_marker(
    row: Mapping[str, Any],
    selection_sha: str,
    reasons: list[str],
    label: str,
) -> None:
    if row.get("diagnostic_continuation") is not True:
        _issue(reasons, f"{label}: diagnostic_continuation is not true")
    supplied = []
    for name in DIAGNOSTIC_SHA_FIELDS:
        if row.get(name) is not None:
            supplied.append((name, row.get(name)))
    if not supplied:
        _issue(reasons, f"{label}: diagnostic selection SHA is missing")
    for name, value in supplied:
        if value != selection_sha:
            _issue(
                reasons,
                f"{label}: {name}={value!r} does not match selection SHA",
            )
    if row.get("diagnostic_only") is not True and not _is_structural(row):
        _issue(reasons, f"{label}: diagnostic_only is not true")
    receipt_path = row.get("diagnostic_selection_receipt_path")
    if receipt_path is not None and receipt_path != DIAGNOSTIC_RECEIPT_REL:
        _issue(reasons, f"{label}: diagnostic selection receipt path is not canonical")
    binding = row.get("diagnostic_selection_binding")
    if binding is not None:
        if not isinstance(binding, Mapping):
            _issue(reasons, f"{label}: diagnostic selection binding is not an object")
        else:
            if binding.get("protocol_sha256") not in (None, DIAGNOSTIC_DOC_SHA256):
                _issue(reasons, f"{label}: diagnostic binding protocol SHA mismatch")
            if binding.get("input_sha256") is not None and not _valid_sha(binding.get("input_sha256"), HEX64):
                _issue(reasons, f"{label}: diagnostic binding input SHA is invalid")
            if binding.get("original_rank") is not None:
                try:
                    if _int(binding.get("original_rank"), "original_rank") < 0:
                        raise ValueError("original_rank must be non-negative")
                except (TypeError, ValueError, OverflowError) as exc:
                    _issue(reasons, f"{label}: invalid diagnostic binding rank: {exc}")
    if row.get("formal_positive_evidence_allowed") not in (None, False):
        _issue(reasons, f"{label}: formal_positive_evidence_allowed must be false")
    if row.get("new_idea_generated") not in (None, False):
        _issue(reasons, f"{label}: new_idea_generated must be false")
    if row.get("confirmatory") not in (None, False):
        _issue(reasons, f"{label}: confirmatory must be false")


def _admit_diagnostic_budget(
    input_root: Path,
    reasons: list[str],
) -> tuple[dict[str, int] | None, str | None, dict[str, Any]]:
    """Call matrix's proof-bound admission before opening any source shard."""
    admit = getattr(matrix, "selected_diagnostic_budget", None)
    if not callable(admit):
        admit = getattr(matrix, "diagnostic_selected_budget", None)
    if not callable(admit):
        _issue(reasons, "evaluation OPEN_DENY: selected_diagnostic_budget is unavailable")
        return None, None, {}

    try:
        api_result = admit(input_root)
    except Exception as exc:
        _issue(reasons, f"evaluation OPEN_DENY: {exc}")
        return None, None, {}
    if not isinstance(api_result, Mapping):
        _issue(reasons, "evaluation OPEN_DENY: admission result is not an object")
        return None, None, {}

    raw_budget = api_result.get("selected_budget", api_result)
    if not isinstance(raw_budget, Mapping):
        _issue(reasons, "evaluation OPEN_DENY: selected budget is missing")
        return None, None, {}
    try:
        budget = _budget(raw_budget)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        _issue(reasons, f"evaluation OPEN_DENY: invalid selected budget: {exc}")
        return None, None, {}
    if _budget_key(budget) not in BUDGETS:
        _issue(reasons, f"evaluation OPEN_DENY: budget outside preregistered grid: {budget!r}")
        return None, None, {}

    receipt_path = input_root / DIAGNOSTIC_RECEIPT_REL
    _check_owned(receipt_path, reasons, "diagnostic selection receipt")
    if not receipt_path.is_file():
        _issue(reasons, f"evaluation OPEN_DENY: missing diagnostic receipt: {receipt_path}")
        return None, None, {}
    try:
        raw = receipt_path.read_bytes()
        receipt = json.loads(
            raw.decode("utf-8"),
            parse_constant=_strict_constant,
        )
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _issue(reasons, f"evaluation OPEN_DENY: invalid diagnostic receipt: {exc}")
        return None, None, {}
    if not isinstance(receipt, Mapping):
        _issue(reasons, "evaluation OPEN_DENY: diagnostic receipt is not an object")
        return None, None, {}
    selection_sha = _sha256_bytes(raw)

    checks = (
        ("status", "DIAGNOSTIC_SELECTED"),
        ("selection_source", "calibration_only"),
        ("calibration_only", True),
        ("confirmatory", False),
        ("diagnostic_only", True),
        ("evaluation_read", False),
    )
    for name, expected in checks:
        if receipt.get(name) != expected:
            _issue(
                reasons,
                f"evaluation OPEN_DENY: receipt {name}={receipt.get(name)!r}, "
                f"expected {expected!r}",
            )
    if receipt.get("evaluation_authorized") not in (None, False):
        _issue(reasons, "evaluation OPEN_DENY: receipt evaluation_authorized must be false")
    for name in ("diagnostic_continuation_doc_sha256", "protocol_sha256"):
        if receipt.get(name) != DIAGNOSTIC_DOC_SHA256:
            _issue(reasons, f"evaluation OPEN_DENY: receipt {name} does not bind protocol")
    declared = receipt.get("selected_budget")
    try:
        if not isinstance(declared, Mapping) or _budget(declared) != budget:
            _issue(reasons, "evaluation OPEN_DENY: receipt budget disagrees with API")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        _issue(reasons, f"evaluation OPEN_DENY: invalid receipt budget: {exc}")
    for name in ("selection_receipt_sha256", "diagnostic_selection_receipt_sha256"):
        value = api_result.get(name)
        if value is not None and value != selection_sha:
            _issue(reasons, f"evaluation OPEN_DENY: API {name} disagrees with receipt")
    if reasons:
        return None, None, {}
    return budget, selection_sha, {
        "receipt_path": str(receipt_path),
        "receipt_sha256": selection_sha,
        "receipt": dict(receipt),
        "api_result": dict(api_result),
    }


def _qualification_rows(
    input_root: Path,
    reasons: list[str],
) -> dict[tuple[str, str, str, str, int], dict[str, Any]]:
    """Collect exactly the 66 qualified source identities."""
    qualified: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    for task in TASKS:
        directory = input_root / "phase0b" / "qualification" / task
        if not directory.is_dir():
            _issue(reasons, f"missing qualification directory: {directory}")
            continue
        for path in sorted(directory.glob("*.json")):
            label = str(path)
            _check_owned(path, reasons, label)
            try:
                payload = _read_json(path)
                if not isinstance(payload, Mapping):
                    raise ValueError("row is not an object")
                if _text(payload.get("task")) != task:
                    raise ValueError("task/path identity mismatch")
                if payload.get("status") != "COMPLETE":
                    raise ValueError(f"terminal status is {payload.get('status')!r}")
                split = _text(payload.get("split"))
                if split not in ("calibration", "evaluation"):
                    continue
                if payload.get("qualified_natural_failure") is not True:
                    continue
                key = _source_key(payload)
                if key in qualified:
                    raise ValueError(f"duplicate qualified source identity {key!r}")
                item = dict(payload)
                item["source_path"] = str(path)
                item["source_file_sha256"] = _sha256(path)
                qualified[key] = item
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                _issue(reasons, f"invalid qualification row {label}: {exc}")

    counts = {
        task: {
            split: sum(key[0] == task and key[3] == split for key in qualified)
            for split in ("calibration", "evaluation")
        }
        for task in TASKS
    }
    if counts != SOURCE_COUNTS:
        _issue(reasons, f"qualified source counts {counts!r} != {SOURCE_COUNTS!r}")
    if len(qualified) != 66:
        _issue(reasons, f"qualified source total {len(qualified)} != 66")
    return qualified


def _event_from_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("event")
    return nested if isinstance(nested, Mapping) else payload


def _source_events(
    input_root: Path,
    qualified: Mapping[tuple[str, str, str, str, int], Mapping[str, Any]],
    reasons: list[str],
) -> dict[tuple[str, str, str, str, int], dict[str, Any]]:
    """Reconstruct source events from calibration and sealed evaluation files."""
    source: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}

    calibration_path = input_root / "phase0b" / "calibration_events.jsonl"
    _check_owned(calibration_path, reasons, "calibration source")
    if not calibration_path.is_file():
        _issue(reasons, f"missing calibration source: {calibration_path}")
    else:
        try:
            for line_number, line in enumerate(
                calibration_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                payload = json.loads(line, parse_constant=_strict_constant)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"line {line_number} is not an object")
                event = dict(_event_from_payload(payload))
                if _text(event.get("split")) != "calibration":
                    raise ValueError(f"line {line_number} is not calibration")
                key = _source_key(event)
                if key in source:
                    raise ValueError(f"duplicate calibration source key {key!r}")
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
            label = str(path)
            _check_owned(path, reasons, label)
            try:
                payload = _read_json(path)
                if not isinstance(payload, Mapping):
                    raise ValueError("episode is not an object")
                if _text(payload.get("task")) != task:
                    raise ValueError("task/path identity mismatch")
                if payload.get("status") != "COMPLETE":
                    raise ValueError(f"terminal status is {payload.get('status')!r}")
                split = _text(payload.get("split"))
                if split != "evaluation":
                    continue
                if payload.get("qualified_natural_failure") is not True:
                    continue
                nested = payload.get("event")
                if not isinstance(nested, Mapping):
                    raise ValueError("qualified evaluation episode lacks event")
                event = dict(nested)
                outer_identity = (
                    _text(event.get("task")) == task
                    and _text(event.get("split")) == "evaluation"
                    and _text(event.get("init_state_id"))
                    == _text(payload.get("init_state_id"))
                    and _text(event.get("event_instance_id"))
                    == _text(payload.get("event_instance_id"))
                )
                if not outer_identity:
                    raise ValueError("evaluation event/shard identity mismatch")
                key = _source_key(event)
                if key in source:
                    raise ValueError(f"duplicate evaluation source key {key!r}")
                event["source_path"] = str(path)
                event["source_file_sha256"] = _sha256(path)
                source[key] = event
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                _issue(reasons, f"invalid sealed evaluation source {label}: {exc}")

    for key in sorted(set(qualified) - set(source), key=_source_sort):
        _issue(reasons, f"qualified source has no raw source event: {key!r}")
    for key in sorted(set(source) - set(qualified), key=_source_sort):
        _issue(reasons, f"raw source event is not qualified: {key!r}")

    for key, event in source.items():
        try:
            task = _text(event["task"])
            if task not in TASK_HORIZONS:
                raise ValueError(f"unknown task {task!r}")
            anchor = _int(event["anchor_global_step"], "anchor_global_step")
            if not 0 <= anchor < TASK_HORIZONS[task]:
                raise ValueError(
                    f"anchor_global_step={anchor} outside [0,{TASK_HORIZONS[task]})"
                )
            event["anchor_global_step"] = anchor
            if _source_key(event) != key:
                raise ValueError("source key changed during normalization")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            _issue(reasons, f"invalid source event {key!r}: {exc}")

    counts = {
        task: {
            split: sum(key[0] == task and key[3] == split for key in source)
            for split in ("calibration", "evaluation")
        }
        for task in TASKS
    }
    if counts != SOURCE_COUNTS:
        _issue(reasons, f"raw source counts {counts!r} != {SOURCE_COUNTS!r}")
    return source


def _structural_valid(
    row: Mapping[str, Any],
    source_event: Mapping[str, Any],
    reasons: list[str],
    label: str,
) -> bool:
    if not _is_structural(row):
        _issue(reasons, f"{label}: unknown/error row is not a structural marker")
        return False
    try:
        task = _text(source_event["task"])
        prefix = _int(row["prefix_k"], "prefix_k")
        anchor = _int(source_event["anchor_global_step"], "anchor_global_step")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        _issue(reasons, f"{label}: source anchor/prefix unavailable: {exc}")
        return False
    if task not in TASK_HORIZONS:
        _issue(reasons, f"{label}: unknown task horizon")
        return False
    horizon = TASK_HORIZONS[task]
    if not 0 <= anchor < horizon:
        _issue(reasons, f"{label}: source anchor {anchor} outside task horizon")
        return False
    if prefix <= 0:
        _issue(reasons, f"{label}: prefix_k must be positive")
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
            f"{label}: structural row is feasible: "
            f"anchor={anchor}+prefix={prefix}<=horizon={horizon}",
        )
        return False
    # The frozen dispatcher serializes safe_success=false on infeasible
    # branches.  Preserve it as metadata, but never treat it as an outcome.
    if row.get("safe_success") not in (None, False):
        _issue(reasons, f"{label}: structural row carries non-null positive/invalid safe_success")
    if row.get("trace_sha256") is not None or row.get("trace_path") is not None:
        # A dispatcher may retain a requested path, but a produced trace is
        # never allowed to silently turn a structural row into an observation.
        if row.get("trace_sha256") is not None:
            _issue(reasons, f"{label}: structural row carries trace SHA")
    if row.get("behavior_operator") not in (None, "fresh_h16" if row.get("is_reference") else row.get("operator")):
        _issue(reasons, f"{label}: behavior_operator disagrees with branch operator")
    return not any(item.startswith(f"{label}:") for item in reasons)


def _validate_reused_calibration(
    input_root: Path,
    row: Mapping[str, Any],
    reasons: list[str],
    label: str = "row",
    *,
    selection_sha: str | None = None,
    allowed_source_commits: set[str] | None = None,
    phase1_source_commits: set[str] | None = None,
    expected_runtime_receipt_sha256: str | None = None,
) -> bool:
    """Verify a Phase-1 raw shard and permit only appended diag/reuse fields."""
    supplied = [
        row.get("reused_calibration_shard"),
        row.get("reused_calibration_sha256"),
    ]
    if supplied == [None, None]:
        return True
    start = len(reasons)
    raw_path, raw_digest = supplied
    if raw_path in (None, "") or not _valid_sha(raw_digest, HEX64):
        _issue(reasons, f"{label}: reused calibration path/hash is incomplete")
        return False
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = input_root / path
    try:
        path = path.resolve()
    except OSError as exc:
        _issue(reasons, f"{label}: cannot resolve reused shard: {exc}")
        return False
    parts = path.parts
    try:
        phase1_index = parts.index("phase1")
        if parts[phase1_index + 1] != "shards" or parts[phase1_index + 2] != _text(row.get("task")):
            raise ValueError("reused shard is not under phase1/shards/<task>")
    except (ValueError, IndexError):
        _issue(reasons, f"{label}: reused shard is not task-scoped Phase-1 evidence")
        return False
    _check_owned(path, reasons, f"{label}: reused shard")
    try:
        if not path.is_file():
            raise OSError("reused shard is missing")
        if _sha256(path) != raw_digest:
            raise ValueError("reused calibration shard SHA-256 mismatch")
        previous = _read_json(path)
        if not isinstance(previous, Mapping):
            raise ValueError("reused shard is not an object")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _issue(reasons, f"{label}: invalid reused calibration evidence: {exc}")
        return False

    key_fields = (
        "task",
        "event_instance_id",
        "init_state_id",
        "split",
        "generator_actor_seed",
        "recovery_actor_seed",
        "operator",
        "prefix_k",
        "tail_horizon",
        "action_budget",
        "policy_call_cap",
        "is_reference",
    )
    try:
        for name in key_fields:
            if name not in previous or name not in row:
                raise ValueError(f"reused branch key missing {name}")
            if name in {
                "generator_actor_seed",
                "recovery_actor_seed",
                "prefix_k",
                "tail_horizon",
                "action_budget",
                "policy_call_cap",
            }:
                if _int(previous[name], name) != _int(row[name], name):
                    raise ValueError(f"reused branch key mismatch {name}")
            elif previous[name] != row[name]:
                raise ValueError(f"reused branch key mismatch {name}")
        prior_sources = set(PHASE1_SOURCE_COMMITS if phase1_source_commits is None else phase1_source_commits)
        # A direct helper caller may supply a synthetic allowed set.  The
        # complete acceptance path always passes both frozen Phase-1 sources.
        if phase1_source_commits is None and allowed_source_commits:
            if PHASE1_SOURCE_COMMIT not in allowed_source_commits:
                prior_sources = set(allowed_source_commits)
        if previous.get("source_commit") not in prior_sources:
            raise ValueError("reused source_commit is not the frozen Phase-1 commit")
        if row.get("source_commit") != previous.get("source_commit"):
            raise ValueError("reused source_commit differs from Phase-1 source")
        previous_receipt = previous.get("runtime_receipt_sha256")
        current_receipt = row.get("runtime_receipt_sha256")
        if not _valid_sha(previous_receipt, HEX64):
            raise ValueError("reused runtime receipt is not a SHA-256")
        if current_receipt != previous_receipt:
            raise ValueError("reused runtime receipt differs from Phase-1 source")
        if expected_runtime_receipt_sha256 is not None and (
            previous_receipt != expected_runtime_receipt_sha256
        ):
            raise ValueError("reused runtime receipt does not match expected binding")
        if selection_sha is not None:
            _diagnostic_marker(row, selection_sha, reasons, label)
        prior_without_internal = {
            key: value
            for key, value in previous.items()
            if not str(key).startswith("__")
        }
        current_without_allowed = {
            key: value
            for key, value in row.items()
            if not str(key).startswith("__") and key not in ALLOWED_REUSE_FIELDS
        }
        if current_without_allowed != prior_without_internal:
            prior_keys = set(prior_without_internal)
            current_keys = set(current_without_allowed)
            missing = sorted(prior_keys - current_keys)
            extra = sorted(current_keys - prior_keys)
            changed = sorted(
                key
                for key in prior_keys & current_keys
                if prior_without_internal[key] != current_without_allowed[key]
            )
            raise ValueError(
                "reused row differs outside allowed diag/reuse fields "
                f"(missing={missing}, extra={extra}, changed={changed})"
            )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        _issue(reasons, f"{label}: invalid reused calibration evidence: {exc}")
    return len(reasons) == start


def _check_d4_one(row: Mapping[str, Any], reasons: list[str], label: str) -> None:
    signatures = row.get("d4_signatures")
    if not isinstance(signatures, Mapping):
        _issue(reasons, f"{label}: d4_signatures is missing")
        return
    for part in ("detection", "pre_tail", "final"):
        value = signatures.get(part)
        if not isinstance(value, Mapping) or not _valid_sha(
            value.get("complete_signature_hash"), HEX64
        ):
            _issue(reasons, f"{label}: d4 {part} complete_signature_hash is invalid")


def _check_d4_groups(
    source: Mapping[tuple[str, str, str, str, int], Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    reasons: list[str],
) -> dict[str, Any]:
    """Check 4 core operators x 3 recovery seeds at one selected budget."""
    by_group: dict[tuple[str, str, str, str, int, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        if bool(row.get("is_reference")):
            continue
        try:
            key = _source_key_from_row(row)
            prefix = _int(row["prefix_k"], "prefix_k")
            by_group[key + (prefix,)].append(row)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            _issue(reasons, f"cannot construct D4 group: {exc}")

    report: dict[str, Any] = {
        "expected_rows_per_group": len(OPERATORS) * len(SEEDS),
        "groups": 0,
        "complete_groups": 0,
        "structural_groups": 0,
        "mismatch_groups": [],
    }
    for source_key in sorted(source, key=_source_sort):
        event = source[source_key]
        for prefix in PHASE2_PREFIXES:
            group_key = source_key + (prefix,)
            group = by_group.get(group_key, [])
            report["groups"] += 1
            label = (
                "d4 group task={} event={} init={} split={} prefix={}".format(
                    source_key[0],
                    source_key[1],
                    source_key[2],
                    source_key[3],
                    prefix,
                )
            )
            try:
                outside = (
                    _int(event["anchor_global_step"], "anchor_global_step") + prefix
                    > TASK_HORIZONS[source_key[0]]
                )
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                outside = False
                _issue(reasons, f"{label}: cannot determine horizon: {exc}")
            if outside:
                if len(group) != len(OPERATORS) * len(SEEDS) or not all(
                    _is_structural(row) for row in group
                ):
                    _issue(
                        reasons,
                        f"{label}: true-horizon group must retain 12 structural rows",
                    )
                    report["mismatch_groups"].append(
                        {"group": label, "kind": "structural", "observed": len(group)}
                    )
                else:
                    report["structural_groups"] += 1
                continue
            if len(group) != len(OPERATORS) * len(SEEDS) or not all(
                _text(row.get("status")) == "COMPLETE" for row in group
            ):
                _issue(
                    reasons,
                    f"{label}: feasible D4 group must contain 12 COMPLETE rows",
                )
                report["mismatch_groups"].append(
                    {"group": label, "kind": "complete", "observed": len(group)}
                )
                continue
            observed = {
                (_text(row.get("operator")), _int(row["recovery_actor_seed"], "recovery"))
                for row in group
            }
            expected = {(operator, seed) for operator in OPERATORS for seed in SEEDS}
            if observed != expected:
                _issue(reasons, f"{label}: operator/recovery combinations are incomplete")
                report["mismatch_groups"].append(
                    {"group": label, "kind": "combination", "observed": len(observed)}
                )
                continue
            pairs = set()
            pids = set()
            for row in group:
                _check_d4_one(row, reasons, label)
                try:
                    pid = _int(row["pid"], "pid")
                    if pid <= 0:
                        raise ValueError("pid must be positive")
                    pair = (_text(row["job_id"]), pid)
                    pairs.add(pair)
                    pids.add(pid)
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    _issue(reasons, f"{label}: invalid process isolation receipt: {exc}")
            if len(pairs) != len(group) or len(pids) != len(group):
                _issue(
                    reasons,
                    f"{label}: (job_id,pid) process isolation is not one-to-one",
                )
                report["mismatch_groups"].append(
                    {"group": label, "kind": "process_isolation"}
                )
                continue
            for part in ("detection", "pre_tail"):
                hashes = {
                    row.get("d4_signatures", {}).get(part, {}).get("complete_signature_hash")
                    for row in group
                    if isinstance(row.get("d4_signatures"), Mapping)
                    and isinstance(row["d4_signatures"].get(part), Mapping)
                }
                if len(hashes) != 1:
                    _issue(
                        reasons,
                        f"{label}: d4 {part} state differs across operators/seeds",
                    )
                    report["mismatch_groups"].append(
                        {
                            "group": label,
                            "kind": f"d4_{part}",
                            "distinct_hashes": sorted(str(item) for item in hashes),
                        }
                    )
                    break
            else:
                report["complete_groups"] += 1
    return report


# Publicly useful alias for callers that prefer a non-private spelling.
check_d4_groups = _check_d4_groups


def _job_status(job: Mapping[str, Any]) -> str | None:
    readback = job.get("readback")
    if isinstance(readback, Mapping) and readback.get("Status") is not None:
        return _text(readback.get("Status"))
    if job.get("status") is not None:
        return _text(job.get("status"))
    return None


def _job_sort_key(job: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        _text(job.get("created_at_utc") or ""),
        _text(job.get("run_id") or ""),
        _text(job.get("job_id") or ""),
    )


def _candidate_jobs_paths(input_root: Path) -> list[Path]:
    candidates = [
        input_root / "control" / "jobs.json",
        input_root / "pai" / "jobs.json",
    ]
    # The canonical registry lives beside artifacts/stage2f.  This derivation
    # is only used when --jobs is omitted; it never walks an arbitrary tree.
    try:
        repository = input_root.parents[1]
        candidates.append(repository / "experiments" / "r16_p14_stage2f" / "pai" / "jobs.json")
    except IndexError:
        pass
    return list(dict.fromkeys(candidates))


def _load_jobs(
    input_root: Path,
    reasons: list[str],
    *,
    jobs_path: Path | None = None,
    phase2_source_commits: set[str],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    candidates = [jobs_path] if jobs_path is not None else _candidate_jobs_paths(input_root)
    path = next((item for item in candidates if item is not None and item.is_file()), None)
    if path is None:
        _issue(reasons, "missing PAI jobs metadata for diagnostic atlas")
        return {}, {"path": None, "atlas_jobs": []}
    _check_owned(path, reasons, f"PAI jobs metadata {path}")
    try:
        payload = _read_json(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _issue(reasons, f"invalid PAI jobs metadata: {exc}")
        return {}, {"path": str(path), "atlas_jobs": []}
    jobs = payload.get("jobs") if isinstance(payload, Mapping) else None
    if not isinstance(jobs, list):
        _issue(reasons, "PAI jobs metadata has no jobs list")
        return {}, {"path": str(path), "atlas_jobs": []}

    atlas: list[Mapping[str, Any]] = []
    for item in jobs:
        if not isinstance(item, Mapping):
            _issue(reasons, "PAI jobs metadata contains a non-object job")
            continue
        phase = _text(item.get("phase")).strip().lower()
        if phase not in {"atlas", "phase2", "diagnostic_atlas"}:
            continue
        atlas.append(item)
        task = _text(item.get("task"))
        if task not in TASKS:
            _issue(reasons, f"unknown atlas task in PAI jobs metadata: {task!r}")
    selected: dict[str, Mapping[str, Any]] = {}
    for task in TASKS:
        candidates_for_task = [job for job in atlas if _text(job.get("task")) == task]
        if not candidates_for_task:
            _issue(reasons, f"missing latest atlas job for {task}")
            continue
        selected_job = sorted(candidates_for_task, key=_job_sort_key)[-1]
        selected[task] = selected_job
        label = f"latest atlas job {task}"
        status = _job_status(selected_job)
        if status != "Succeeded":
            _issue(reasons, f"{label}: PAI status is {status!r}, expected Succeeded")
        if selected_job.get("job_id") in (None, ""):
            _issue(reasons, f"{label}: job_id is missing")
        if selected_job.get("run_id") in (None, ""):
            _issue(reasons, f"{label}: run_id is missing")
        if selected_job.get("gpus") != 2:
            _issue(reasons, f"{label}: gpus must be exactly 2")
        # ``persisted_completion_verified`` is a publication-side summary.  The
        # acceptance decision derives persistence from the terminal state,
        # completion receipts, complete rows, traces, and ownership checks
        # below.  A stale false value must not reject otherwise valid evidence;
        # retain the field in the returned job metadata for auditability.
        source_commit = selected_job.get("source_commit")
        if not _valid_sha(source_commit, HEX40) or source_commit not in phase2_source_commits:
            _issue(reasons, f"{label}: source_commit is not an allowed Phase-2 commit")
        if selected_job.get("application_fatal_error") is True:
            _issue(reasons, f"{label}: application_fatal_error is true")
        if selected_job.get("placement_rejected") is True:
            _issue(reasons, f"{label}: placement_rejected is true")
        readback = selected_job.get("readback")
        if isinstance(readback, Mapping) and readback.get("JobId") not in (
            None,
            selected_job.get("job_id"),
        ):
            _issue(reasons, f"{label}: readback JobId mismatch")
        _validate_job_state(selected_job, task, phase2_source_commits, reasons, label)
    return selected, {
        "path": str(path),
        "atlas_jobs": [
            {key: value for key, value in job.items() if not str(key).startswith("__")}
            for job in atlas
        ],
        "selected": {
            task: {
                key: value
                for key, value in job.items()
                if not str(key).startswith("__")
            }
            for task, job in selected.items()
        },
    }


def _validate_job_state(
    job: Mapping[str, Any],
    task: str,
    phase2_source_commits: set[str],
    reasons: list[str],
    label: str,
) -> None:
    artifact_raw = job.get("artifact_dir")
    if not isinstance(artifact_raw, str) or not artifact_raw:
        _issue(reasons, f"{label}: artifact_dir is missing")
        return
    artifact = Path(artifact_raw)
    _check_owned(artifact, reasons, f"{label}: artifact_dir")
    state_dir = artifact / "pai_state"
    completed_path = state_dir / "COMPLETED.json"
    _check_owned(completed_path, reasons, f"{label}: COMPLETED.json")
    if not completed_path.is_file():
        _issue(reasons, f"{label}: pai_state/COMPLETED.json is missing")
        return
    try:
        completed = _read_json(completed_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _issue(reasons, f"{label}: invalid COMPLETED.json: {exc}")
        return
    if not isinstance(completed, Mapping):
        _issue(reasons, f"{label}: COMPLETED.json is not an object")
        return
    for name in ("uid", "gid"):
        if completed.get(name) != UID_GID:
            _issue(reasons, f"{label}: COMPLETED.json {name} is not {UID_GID}")
    if completed.get("task") not in (None, task):
        _issue(reasons, f"{label}: COMPLETED.json task mismatch")
    if completed.get("phase") not in (None, "atlas", "phase2"):
        _issue(reasons, f"{label}: COMPLETED.json phase mismatch")
    for name, expected in (
        ("job_id", job.get("job_id")),
        ("run_id", job.get("run_id")),
    ):
        if completed.get(name) not in (None, expected):
            _issue(reasons, f"{label}: COMPLETED.json {name} mismatch")
    source_commit = completed.get("source_commit")
    if source_commit is not None and source_commit not in phase2_source_commits:
        _issue(reasons, f"{label}: COMPLETED.json source_commit is not Phase-2")
    status_values = [completed.get("status")]
    result = completed.get("result")
    if isinstance(result, Mapping):
        status_values.append(result.get("status"))
    for value in status_values:
        if value is not None and _text(value) not in TERMINAL_COMPLETION_STATUSES:
            _issue(reasons, f"{label}: COMPLETED.json carries nonterminal status {value!r}")
    receipt_raw = job.get("completion_receipt")
    if receipt_raw is not None:
        receipt = Path(_text(receipt_raw))
        if not receipt.is_absolute():
            receipt = artifact / receipt
        _check_owned(receipt, reasons, f"{label}: completion receipt")
        if not receipt.is_file():
            _issue(reasons, f"{label}: declared completion receipt is missing")
        else:
            try:
                value = _read_json(receipt)
                if not isinstance(value, Mapping):
                    raise ValueError("completion receipt is not an object")
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                _issue(reasons, f"{label}: invalid declared completion receipt: {exc}")


def _read_rows(directory: Path, reasons: list[str], task: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not directory.is_dir():
        _issue(reasons, f"missing Phase-2 shard directory: {directory}")
        return rows
    _check_owned(directory, reasons, f"Phase-2 shard directory {directory}")
    paths = sorted(directory.glob("*.json"))
    if not paths:
        _issue(reasons, f"no Phase-2 shard JSON files in {directory}")
    for path in paths:
        _check_owned(path, reasons, f"Phase-2 shard {path}")
        try:
            payload = _read_json(path)
            if isinstance(payload, Mapping):
                list_value = None
                for key in ("rows", "data", "shards"):
                    if key in payload:
                        list_value = payload[key]
                        break
                if list_value is None:
                    material = [payload]
                elif isinstance(list_value, list):
                    material = list_value
                else:
                    raise ValueError(f"{path}: row container is not a list")
            elif isinstance(payload, list):
                material = payload
            else:
                raise ValueError(f"{path}: expected object or list")
            if not all(isinstance(item, Mapping) for item in material):
                raise ValueError(f"{path}: row list contains a non-object")
            for item in material:
                row = dict(item)
                row["__source_path"] = str(path)
                rows.append(row)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            _issue(reasons, f"invalid Phase-2 shard {path}: {exc}")
    return rows


def _trace_scope_is_valid(path: Path, phase: str, task: str) -> bool:
    parts = path.parts
    for index, value in enumerate(parts):
        if value == phase and index + 2 < len(parts):
            if parts[index + 1] == "contact_topology" and parts[index + 2] == task:
                return True
    return False


def _validate_completion_receipts(
    input_root: Path,
    source: Mapping[tuple[str, str, str, str, int], Mapping[str, Any]],
    selection_sha: str,
    budget: Mapping[str, Any],
    structural_counts: Mapping[str, int],
    reasons: list[str],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for task in TASKS:
        path = input_root / "phase2" / f"completion_{task}.json"
        _check_owned(path, reasons, f"Phase-2 completion receipt {path}")
        if not path.is_file():
            _issue(reasons, f"missing Phase-2 completion receipt: {path}")
            continue
        try:
            payload = _read_json(path)
            if not isinstance(payload, Mapping):
                raise ValueError("completion receipt is not an object")
            if payload.get("task") != task or payload.get("phase") != "atlas":
                raise ValueError("completion task/phase mismatch")
            status = payload.get("status")
            if status not in TERMINAL_COMPLETION_STATUSES:
                raise ValueError(f"status {status!r} is not terminal")
            if payload.get("diagnostic_continuation") is not True:
                raise ValueError("diagnostic_continuation is not true")
            if payload.get("diagnostic_selection_receipt_sha256") != selection_sha:
                raise ValueError("diagnostic selection SHA mismatch")
            expected_events = sum(
                key[0] == task for key in source
            )
            expected_rows = expected_events * (
                len(SEEDS) * (len(OPERATORS) * len(PHASE2_PREFIXES) + len(REFERENCE_OPERATORS))
            )
            for name, expected in (
                ("events", expected_events),
                ("requested_rows", expected_rows),
                ("persisted_rows", expected_rows),
                ("core_rows", expected_events * len(SEEDS) * len(OPERATORS) * len(PHASE2_PREFIXES)),
                ("reference_rows", expected_events * len(SEEDS) * len(REFERENCE_OPERATORS)),
            ):
                if name in payload:
                    if _int(payload[name], name) != expected:
                        raise ValueError(
                            f"{name}={payload[name]!r}, expected {expected}"
                        )
                elif name in ("events", "requested_rows", "persisted_rows"):
                    raise ValueError(f"missing {name}")
            blocked_count = _int(payload.get("blocked_contract_rows", 0), "blocked_contract_rows")
            if blocked_count != structural_counts.get(task, 0):
                raise ValueError(
                    f"blocked_contract_rows={blocked_count}, expected {structural_counts.get(task, 0)}"
                )
            expected_status = (
                "COMPLETE_WITH_BLOCKED_CONTRACT_ROWS"
                if blocked_count
                else "COMPLETE"
            )
            if status != expected_status:
                raise ValueError(
                    f"status={status!r}, expected {expected_status!r}"
                )
            declared = payload.get("selected_budget")
            if declared is not None and _budget(declared) != _budget(budget):
                raise ValueError("completion selected_budget differs from admission")
            result[task] = dict(payload)
        except (OSError, KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
            _issue(reasons, f"invalid Phase-2 completion receipt {path}: {exc}")
    return result


# Alias used by callers that mirror the consolidator's naming.
_completion_metadata = _validate_completion_receipts


def _validate_rows(
    input_root: Path,
    source: Mapping[tuple[str, str, str, str, int], Mapping[str, Any]],
    budget: Mapping[str, Any],
    selection_sha: str,
    phase2_source_commits: set[str],
    reasons: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[Path, Mapping[str, Any]]]:
    raw: list[dict[str, Any]] = []
    for task in TASKS:
        raw.extend(_read_rows(input_root / "phase2" / "shards" / task, reasons, task))

    expected_core, expected_reference = _all_expected_keys(source, budget)
    expected = expected_core | expected_reference
    observed: set[tuple[Any, ...]] = set()
    rows: list[dict[str, Any]] = []
    structural_counts = {task: 0 for task in TASKS}
    trace_rows: dict[Path, Mapping[str, Any]] = {}
    trace_paths_by_phase = {"phase1": set(), "phase2": set()}

    for index, item in enumerate(raw):
        label = _text(item.get("__source_path", f"row-{index}"))
        try:
            path = Path(label)
            if path.parent.name not in TASKS:
                _issue(reasons, f"{label}: shard path is not task-scoped")
                continue
            task = _text(item.get("task"))
            if task not in TASKS or path.parent.name != task:
                _issue(reasons, f"{label}: row task/path mismatch")
                continue
            required = (
                "event_instance_id",
                "init_state_id",
                "split",
                "generator_actor_seed",
                "recovery_actor_seed",
                "operator",
                "prefix_k",
                "tail_horizon",
                "action_budget",
                "policy_call_cap",
                "status",
            )
            missing = [name for name in required if name not in item]
            if missing:
                _issue(reasons, f"{label}: missing row fields {missing}")
                continue
            normalized = dict(item)
            normalized["task"] = task
            normalized["event_instance_id"] = _text(item["event_instance_id"])
            normalized["init_state_id"] = _text(item["init_state_id"])
            normalized["split"] = _text(item["split"])
            for name in (
                "generator_actor_seed",
                "recovery_actor_seed",
                "prefix_k",
                "tail_horizon",
                "action_budget",
                "policy_call_cap",
            ):
                normalized[name] = _int(item[name], name)
            normalized["operator"] = _text(item["operator"])
            if normalized["generator_actor_seed"] not in SEEDS:
                raise ValueError("generator_actor_seed is outside frozen seeds")
            if normalized["recovery_actor_seed"] not in SEEDS:
                raise ValueError("recovery_actor_seed is outside frozen seeds")
            if normalized["split"] not in ("calibration", "evaluation"):
                raise ValueError("split is invalid")
            if not isinstance(item.get("is_reference"), bool):
                raise ValueError("is_reference must be an explicit boolean")
            is_reference = bool(item["is_reference"])
            normalized["is_reference"] = is_reference
            if is_reference:
                if normalized["operator"] not in REFERENCE_PREFIXES:
                    raise ValueError("reference row has unknown operator")
                if normalized["prefix_k"] != REFERENCE_PREFIXES[normalized["operator"]]:
                    raise ValueError("reference prefix does not match reference operator")
            else:
                if normalized["operator"] not in OPERATORS:
                    raise ValueError("core row has unknown operator")
                if normalized["prefix_k"] not in PHASE2_PREFIXES:
                    raise ValueError("core prefix is outside Phase-2 atlas")
            if normalized["behavior_operator"] if "behavior_operator" in normalized else False:
                expected_behavior = "fresh_h16" if is_reference else normalized["operator"]
                if normalized["behavior_operator"] != expected_behavior:
                    raise ValueError(
                        f"behavior_operator={normalized['behavior_operator']!r}, "
                        f"expected {expected_behavior!r}"
                    )
            branch_key = _row_key(normalized)
            if branch_key in observed:
                raise ValueError(f"duplicate branch key {branch_key!r}")
            observed.add(branch_key)
            if branch_key not in expected:
                raise ValueError(f"unexpected Phase-2 branch key {branch_key!r}")
            source_key = _source_key_from_row(normalized)
            source_event = source.get(source_key)
            if source_event is None:
                raise ValueError(f"row source identity is absent: {source_key!r}")
            if (normalized["tail_horizon"], normalized["action_budget"], normalized["policy_call_cap"]) != _budget_key(budget):
                raise ValueError("row budget differs from selected budget")
            configured = normalized.get("configured_budget")
            if configured is not None and (
                not isinstance(configured, Mapping) or _budget(configured) != _budget(budget)
            ):
                raise ValueError("configured_budget differs from selected budget")
            if "anchor_global_step" in normalized:
                if _int(normalized["anchor_global_step"], "anchor_global_step") != _int(
                    source_event["anchor_global_step"], "anchor_global_step"
                ):
                    raise ValueError("anchor_global_step differs from source")
            _diagnostic_marker(normalized, selection_sha, reasons, label)
            structural = _is_structural(normalized)
            if structural:
                if not _structural_valid(normalized, source_event, reasons, label):
                    raise ValueError("invalid PrefixOutsideTaskHorizon marker")
                structural_counts[task] += 1
                if normalized.get("trace_path") is not None and normalized.get("trace_sha256") is not None:
                    raise ValueError("structural row cannot provide a trace")
                if "source_commit" in normalized and normalized["source_commit"] not in (
                    set(PHASE1_SOURCE_COMMITS) | set(phase2_source_commits)
                ):
                    raise ValueError("structural source_commit is not allowed")
            else:
                if _text(normalized.get("status")) != "COMPLETE":
                    raise ValueError(
                        f"unknown/nonterminal status {normalized.get('status')!r}"
                    )
                if normalized.get("blocked") not in (None, False):
                    raise ValueError("COMPLETE row has blocked=true")
                if normalized.get("error") not in (None, "", False, []):
                    raise ValueError("COMPLETE row carries error")
                if normalized.get("error_type") not in (None, "", False, []):
                    raise ValueError("COMPLETE row carries error_type")
                if not isinstance(normalized.get("safe_success"), bool):
                    raise ValueError("COMPLETE row safe_success must be boolean")
                source_commit = normalized.get("source_commit")
                if not _valid_sha(source_commit, HEX40):
                    raise ValueError("COMPLETE row source_commit is not a SHA-1")
                reused = normalized.get("reused_calibration_shard") is not None or normalized.get(
                    "reused_calibration_sha256"
                ) is not None
                if reused:
                    if normalized["split"] != "calibration":
                        raise ValueError("only calibration rows may be reused")
                    if not _validate_reused_calibration(
                        input_root,
                        item,  # Compare immutable raw fields before normalization.
                        reasons,
                        label,
                        selection_sha=selection_sha,
                        allowed_source_commits=set(PHASE1_SOURCE_COMMITS) | set(phase2_source_commits),
                        phase1_source_commits=set(PHASE1_SOURCE_COMMITS),
                    ):
                        raise ValueError("reused calibration evidence failed")
                elif source_commit not in phase2_source_commits:
                    raise ValueError("fresh row source_commit is not an allowed Phase-2 commit")
                runtime_receipt = normalized.get("runtime_receipt_sha256")
                if not _valid_sha(runtime_receipt, HEX64):
                    raise ValueError("COMPLETE row runtime receipt is invalid")
                for name in ("job_id", "pai_run_id", "env_hash", "chunk_hash"):
                    if normalized.get(name) in (None, ""):
                        raise ValueError(f"COMPLETE row {name} is missing")
                pid = normalized.get("pid")
                if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                    raise ValueError("COMPLETE row pid is invalid")
                if normalized.get("process_start_method") != "spawn":
                    raise ValueError("COMPLETE row process_start_method is not spawn")
                if normalized.get("dispatch_process_start_method") not in (None, "spawn"):
                    raise ValueError("COMPLETE row dispatch process start method is not spawn")
                if normalized.get("fresh_environment_created") not in (None, True):
                    raise ValueError("COMPLETE row did not create a fresh environment")
                labels = normalized.get("labels")
                if not isinstance(labels, Mapping):
                    raise ValueError("COMPLETE row labels are missing")
                trace_raw = normalized.get("trace_path")
                if not isinstance(trace_raw, str) or not trace_raw.endswith(".jsonl.gz"):
                    raise ValueError("COMPLETE row trace_path is missing or not gzip JSONL")
                trace_path = Path(trace_raw)
                if not trace_path.is_absolute():
                    trace_path = input_root / trace_path
                trace_path = trace_path.resolve()
                expected_phase = "phase1" if reused else "phase2"
                if not _trace_scope_is_valid(trace_path, expected_phase, task):
                    raise ValueError(
                        f"COMPLETE row trace_path is not under {expected_phase}/contact_topology/{task}"
                    )
                _check_owned(trace_path, reasons, f"{label}: trace")
                trace_paths_by_phase[expected_phase].add(trace_path)
                if trace_path in trace_rows:
                    raise ValueError(f"trace path is reused by multiple rows: {trace_path}")
                trace_rows[trace_path] = normalized
                if not _valid_sha(normalized.get("trace_sha256"), HEX64):
                    raise ValueError("COMPLETE row trace_sha256 is invalid")
                if not _valid_sha(normalized.get("trace_content_sha256"), HEX64):
                    raise ValueError("COMPLETE row trace_content_sha256 is invalid")
                if _int(normalized.get("trace_records"), "trace_records") <= 0:
                    raise ValueError("COMPLETE row trace_records must be positive")
            rows.append(normalized)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            _issue(reasons, f"{label}: invalid Phase-2 row: {exc}")

    missing = expected - observed
    extra = observed - expected
    if missing:
        _issue(reasons, f"missing expected Phase-2 branch keys: {len(missing)}")
    if extra:
        _issue(reasons, f"unexpected Phase-2 branch keys: {len(extra)}")

    # There may be no phase-2 trace directory only when every row was a reuse;
    # diagnostic atlas always has evaluation rows, so a fresh phase is expected.
    phase2_directory = input_root / "phase2" / "contact_topology"
    if not phase2_directory.is_dir():
        if trace_paths_by_phase["phase2"]:
            _issue(reasons, f"missing Phase-2 trace directory: {phase2_directory}")
    else:
        _check_owned(phase2_directory, reasons, f"Phase-2 trace directory {phase2_directory}")
        actual = {
            path.resolve()
            for task in TASKS
            for path in (phase2_directory / task).glob("*.jsonl.gz")
            if path.is_file()
        }
        expected_phase2 = trace_paths_by_phase["phase2"]
        if actual != expected_phase2:
            _issue(
                reasons,
                "Phase-2 trace file set differs from COMPLETE row references "
                f"(actual={len(actual)}, expected={len(expected_phase2)})",
            )

    report = {
        "expected_rows": len(expected),
        "expected_core_rows": len(expected_core),
        "expected_reference_rows": len(expected_reference),
        "observed_rows": len(observed),
        "complete_rows": sum(_text(row.get("status")) == "COMPLETE" for row in rows),
        "structural_rows": sum(_is_structural(row) for row in rows),
        "rows_by_task": {
            task: sum(row.get("task") == task for row in rows) for task in TASKS
        },
        "structural_rows_by_task": structural_counts,
        "trace_paths": len(trace_rows),
    }
    return rows, report, trace_rows


def _validate_trace_tree(
    input_root: Path,
    trace_rows: Mapping[Path, Mapping[str, Any]],
    trace_workers: int,
    reasons: list[str],
) -> dict[str, Any]:
    try:
        workers = _validate_trace_workers(trace_workers)
    except ValueError as exc:
        _issue(reasons, str(exc))
        return {"scope": "full", "files": 0, "records": 0}
    if not trace_rows:
        _issue(reasons, "no COMPLETE traces to validate")
        return {"scope": "full", "files": 0, "records": 0}
    trace_reasons, report = _validate_trace_files(trace_rows, "full", workers)
    for reason in trace_reasons:
        _issue(reasons, reason)
    return report


def _normalize_phase2_source_commits(
    values: Iterable[str] | str | None,
) -> set[str]:
    if values is None:
        return set(DEFAULT_PHASE2_SOURCE_COMMITS)
    if isinstance(values, str):
        values = [values]
    result = set()
    for value in values:
        if not _valid_sha(value, HEX40):
            raise ValueError(f"invalid Phase-2 source commit: {value!r}")
        result.add(str(value).lower())
    if not result:
        raise ValueError("at least one Phase-2 source commit is required")
    return result


def evaluate_phase2(
    input_root: str | Path,
    *,
    output: str | Path | None = None,
    jobs_path: str | Path | None = None,
    trace_workers: int = 8,
    phase2_source_commits: Iterable[str] | str | None = None,
    phase2_source_commit: str | None = None,
) -> dict[str, Any]:
    """Return a machine-readable PASS/BLOCKED Phase-2 acceptance result."""
    input_path = Path(input_root).resolve()
    reasons: list[str] = []
    try:
        source_commits = _normalize_phase2_source_commits(phase2_source_commits)
        if phase2_source_commit is not None:
            source_commits |= _normalize_phase2_source_commits(phase2_source_commit)
    except ValueError as exc:
        source_commits = set()
        _issue(reasons, str(exc))

    budget, selection_sha, admission = _admit_diagnostic_budget(input_path, reasons)
    base_report: dict[str, Any] = {
        "schema_version": 1,
        "phase": "phase2",
        "status": "BLOCKED",
        "accepted": False,
        "blocked": True,
        "confirmatory": False,
        "diagnostic_continuation": True,
        "evaluation_read": False,
        "evaluation_read_stage": "NOT_ADMITTED",
        "no_evaluation_outcomes_read": True,
        "selection_receipt_sha256": selection_sha,
        "selected_budget": budget,
        "phase2_source_commits": sorted(source_commits),
        "blocking_reasons": list(dict.fromkeys(reasons)),
        "trace_workers": trace_workers,
    }
    if reasons or budget is None or selection_sha is None:
        if output is not None:
            write_report(output, base_report)
        return base_report

    # The matrix API has now verified the real proof.  Only from this point
    # onward may the checker read sealed evaluation source metadata.
    base_report["evaluation_read"] = True
    base_report["evaluation_read_stage"] = "ADMITTED_RAW_RECORD_VALIDATION"
    base_report["no_evaluation_outcomes_read"] = False
    qualified = _qualification_rows(input_path, reasons)
    source = _source_events(input_path, qualified, reasons)
    expected_core, expected_reference = _all_expected_keys(source, budget)
    jobs, jobs_report = _load_jobs(
        input_path,
        reasons,
        jobs_path=Path(jobs_path).resolve() if jobs_path is not None else None,
        phase2_source_commits=source_commits,
    )
    rows, row_report, trace_rows = _validate_rows(
        input_path,
        source,
        budget,
        selection_sha,
        source_commits,
        reasons,
    )
    structural_counts = row_report.get("structural_rows_by_task", {})
    completions = _validate_completion_receipts(
        input_path,
        source,
        selection_sha,
        budget,
        structural_counts,
        reasons,
    )
    trace_report = _validate_trace_tree(
        input_path,
        trace_rows,
        trace_workers,
        reasons,
    )
    d4_report = _check_d4_groups(source, rows, reasons)

    source_counts = {
        task: {
            split: sum(key[0] == task and key[3] == split for key in source)
            for split in ("calibration", "evaluation")
        }
        for task in TASKS
    }
    base_report.update(
        {
            "status": "PASS" if not reasons else "BLOCKED",
            "accepted": not reasons,
            "blocked": bool(reasons),
            "blocking_reasons": list(dict.fromkeys(reasons))[:500],
            "source_counts": source_counts,
            "qualified_source_count": len(qualified),
            "raw_source_count": len(source),
            "expected_rows": len(expected_core | expected_reference),
            "expected_core_rows": len(expected_core),
            "expected_reference_rows": len(expected_reference),
            "observed_rows": row_report.get("observed_rows", 0),
            "row_counts": row_report,
            "jobs": jobs_report,
            "selected_jobs": {
                task: {
                    key: value
                    for key, value in job.items()
                    if not str(key).startswith("__")
                }
                for task, job in jobs.items()
            },
            "completion_receipts": {
                task: dict(value) for task, value in completions.items()
            },
            "trace": trace_report,
            "d4": d4_report,
            "no_evaluation_outcomes_read": False,
        }
    )
    if output is not None:
        write_report(output, base_report)
    return base_report


def write_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    destination = Path(path).resolve()
    if destination.exists() and destination.is_dir():
        destination = destination / "phase2_acceptance.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        dict(report),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", "--base", "--input", dest="input_root", type=Path, required=True)
    parser.add_argument("--output", "--output-root", dest="output", type=Path, required=True)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--trace-workers", type=int, default=8)
    parser.add_argument(
        "--phase2-source-commit",
        "--allowed-source-commit",
        dest="phase2_source_commits",
        action="append",
    )
    args = parser.parse_args(argv)
    try:
        report = evaluate_phase2(
            args.input_root,
            output=args.output,
            jobs_path=args.jobs,
            trace_workers=args.trace_workers,
            phase2_source_commits=args.phase2_source_commits,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": 1,
            "phase": "phase2",
            "status": "BLOCKED",
            "accepted": False,
            "blocked": True,
            "evaluation_read": None,
            "evaluation_read_stage": "UNKNOWN_AFTER_EXCEPTION",
            "no_evaluation_outcomes_read": None,
            "blocking_reasons": [f"acceptance crashed closed: {exc}"],
        }
        write_report(args.output, report)
    print(
        json.dumps(
            {
                "phase": "phase2",
                "status": report.get("status"),
                "accepted": report.get("accepted"),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if report.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
