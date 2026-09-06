"""Pure, fail-closed consolidation for the Stage-2F/S1 artifacts.

The consolidator deliberately treats shard completeness and provenance as a
precondition.  It never fills a missing branch with a synthetic row, and the
phase-2 reader is called only after the receipt/authorization binding has been
checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # package invocation
    from .statistics import (
        FAMILY_A,
        FAMILY_B,
        OPERATORS,
        PHASE1_PREFIXES,
        PHASE2_PREFIXES,
        analyze_crossing,
        select_budget,
        summarize_grid,
    )
except ImportError:  # direct script invocation from the s1 directory
    from statistics import (  # type: ignore
        FAMILY_A,
        FAMILY_B,
        OPERATORS,
        PHASE1_PREFIXES,
        PHASE2_PREFIXES,
        analyze_crossing,
        select_budget,
        summarize_grid,
    )

TASKS = (
    "put_the_cream_cheese_in_the_bowl",
    "put_the_bowl_on_the_plate",
)
SEEDS = (7, 17, 29)
SEED_TEXT = tuple(str(seed) for seed in SEEDS)
BUDGETS = tuple((tail, action, 8) for tail in (4, 8, 16) for action in (8, 16, 32))
REFERENCE_PREFIXES = {
    "immediate_fresh": 2,
    "fixed_delay_2": 4,
    "fixed_delay_4": 6,
    "fixed_delay_8": 10,
}
REFERENCE_OPERATORS = tuple(REFERENCE_PREFIXES)
CORE_FIELDS = (
    "event_instance_id",
    "task",
    "init_state_id",
    "split",
    "generator_actor_seed",
    "recovery_actor_seed",
    "operator",
    "prefix_k",
    "tail_horizon",
    "action_budget",
    "policy_call_cap",
    "safe_success",
    "pid",
    "env_hash",
    "chunk_hash",
    "status",
)
QUALIFICATION_FIELDS = (
    "event_instance_id",
    "task",
    "init_state_id",
    "actor_seed",
    "split",
    "qualified_natural_failure",
    "status",
)


class ConsolidationError(RuntimeError):
    """Raised only for an immutable-output conflict or programmer error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_immutable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ConsolidationError(f"immutable artifact differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _write_text_immutable(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )
    _write_text_immutable(path, text)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_rows(payload: Any, source: Path) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        for key in ("rows", "data", "shards"):
            if key in payload:
                value = payload[key]
                if not isinstance(value, list):
                    raise ValueError(f"{source}: {key} must be a list")
                return [dict(row) for row in value if isinstance(row, Mapping)]
        return [dict(payload)]
    if isinstance(payload, list):
        if not all(isinstance(row, Mapping) for row in payload):
            raise ValueError(f"{source}: list contains a non-object row")
        return [dict(row) for row in payload]
    raise ValueError(f"{source}: expected object or list")


def _read_rows(directory: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    if not directory.is_dir():
        return [], [f"missing directory: {directory}"]
    for path in sorted(directory.glob("*.json")):
        try:
            material = _as_rows(_read_json(path), path)
            for row in material:
                row["__source_path"] = str(path)
                rows.append(row)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            reasons.append(f"cannot read {path}: {exc}")
    if not rows and not reasons:
        reasons.append(f"no JSON rows in {directory}")
    return rows, reasons


def _text(value: Any) -> str:
    return str(value)


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    number = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer")
    return number


def _seed(value: Any, field: str = "seed") -> int:
    number = _int(value, field)
    if number not in SEEDS:
        raise ValueError(f"{field} must be one of {list(SEEDS)}")
    return number


def _value(row: Mapping[str, Any], name: str, *containers: Mapping[str, Any] | None) -> Any:
    if name in row:
        return row[name]
    for container in containers:
        if container is not None and name in container:
            return container[name]
    return None


def _row_containers(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    containers: list[Mapping[str, Any]] = []
    for name in ("configured_budget", "request", "parameters", "config"):
        value = row.get(name)
        if isinstance(value, Mapping):
            containers.append(value)
    return containers


def _canonical_row(row: Mapping[str, Any], phase: str) -> tuple[dict[str, Any] | None, list[str]]:
    containers = _row_containers(row)
    result: dict[str, Any] = {
        key: value for key, value in row.items() if not str(key).startswith("__")
    }
    reasons: list[str] = []
    for field in CORE_FIELDS:
        value = _value(row, field, *containers)
        if value is None:
            reasons.append(f"missing {field}")
            continue
        result[field] = value
    if reasons:
        return None, reasons
    try:
        result["task"] = _text(result["task"])
        result["event_instance_id"] = _text(result["event_instance_id"])
        result["init_state_id"] = _text(result["init_state_id"])
        result["split"] = _text(result["split"])
        result["generator_actor_seed"] = _seed(result["generator_actor_seed"], "generator_actor_seed")
        result["recovery_actor_seed"] = _seed(result["recovery_actor_seed"], "recovery_actor_seed")
        result["operator"] = _text(result["operator"])
        result["prefix_k"] = _int(result["prefix_k"], "prefix_k")
        for field in ("tail_horizon", "action_budget", "policy_call_cap"):
            result[field] = _int(result[field], field)
        result["safe_success"] = float(result["safe_success"])
        if not 0 <= result["safe_success"] <= 1:
            raise ValueError("safe_success must be in [0,1]")
    except (TypeError, ValueError, OverflowError) as exc:
        reasons.append(str(exc))
    if str(result.get("status", "")).strip().upper() != "COMPLETE":
        reasons.append(f"status is not COMPLETE: {result.get('status')!r}")
    for field in ("pid", "env_hash", "chunk_hash"):
        if result.get(field) is None or not str(result.get(field)).strip():
            reasons.append(f"missing provenance {field}")
    if result.get("task") not in TASKS:
        reasons.append(f"unexpected task: {result.get('task')!r}")
    if phase == "phase1" and result.get("split") != "calibration":
        reasons.append(f"Phase-1 row has non-calibration split: {result.get('split')!r}")
    if phase == "phase2" and result.get("split") not in ("calibration", "evaluation"):
        reasons.append(f"Phase-2 row has invalid split: {result.get('split')!r}")
    if reasons:
        return None, reasons
    return result, []


def _branch_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in (
        "task", "event_instance_id", "init_state_id", "split", "generator_actor_seed",
        "recovery_actor_seed", "operator", "prefix_k", "tail_horizon", "action_budget",
        "policy_call_cap",
    ))


def _event_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in ("task", "event_instance_id", "init_state_id", "split", "generator_actor_seed"))


def _sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, int, str, int, int, int]:
    return (
        _text(row["task"]), _text(row["split"]), _text(row["init_state_id"]),
        _text(row["event_instance_id"]), int(row["generator_actor_seed"]),
        _text(row["operator"]), int(row["prefix_k"]), int(row["tail_horizon"]), int(row["action_budget"]),
    )


def _blocked(phase: str, output_root: Path, reasons: Iterable[str], **extra: Any) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "phase": phase,
        "status": "BLOCKED",
        "blocked": True,
        "blocking_reasons": list(dict.fromkeys(str(reason) for reason in reasons))[:100] or ["unspecified blocker"],
    }
    result.update(extra)
    _write_json(output_root / phase / "summary.json", result)
    return result


def _qualification_rows(input_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    seen: set[tuple[str, str, int]] = set()
    for task in TASKS:
        material, read_reasons = _read_rows(input_root / "phase0b" / "qualification" / task)
        reasons.extend(read_reasons)
        for row in material:
            missing = [field for field in QUALIFICATION_FIELDS if field not in row]
            if missing:
                reasons.append(f"qualification row missing {missing}: {row.get('__source_path')}")
                continue
            try:
                key = (_text(row["task"]), _text(row["init_state_id"]), _seed(row["actor_seed"], "actor_seed"))
                row["task"] = key[0]
                row["init_state_id"] = key[1]
                row["actor_seed"] = key[2]
                row["qualified_natural_failure"] = bool(row["qualified_natural_failure"])
            except (TypeError, ValueError) as exc:
                reasons.append(f"invalid qualification row {row.get('__source_path')}: {exc}")
                continue
            if key in seen:
                reasons.append(f"duplicate qualification key: {key!r}")
            seen.add(key)
            if row["task"] != task:
                reasons.append(f"qualification task/path mismatch: {row.get('__source_path')}")
            try:
                init_number = int(row["init_state_id"])
                expected_split = "infrastructure" if init_number < 10 else "calibration" if init_number < 50 else "evaluation" if init_number < 90 else "reserve"
                if row["split"] != expected_split:
                    reasons.append(f"qualification split mismatch for {key!r}")
            except (TypeError, ValueError):
                reasons.append(f"qualification init_state_id is not an integer for {key!r}")
            if str(row["status"]).upper() != "COMPLETE":
                reasons.append(f"qualification status is not COMPLETE for {key!r}")
            row.pop("__source_path", None)
            rows.append(row)
    expected = {(task, str(init), seed) for task in TASKS for init in range(100) for seed in SEEDS}
    observed = set((row["task"], row["init_state_id"], row["actor_seed"]) for row in rows)
    if len(rows) != 600:
        reasons.append(f"expected exactly 600 qualification rows; observed {len(rows)}")
    if observed != expected:
        reasons.append(f"qualification key set is incomplete: expected {len(expected)}, observed {len(observed)}")
    return sorted(rows, key=lambda row: (_text(row["task"]), int(row["init_state_id"]), int(row["actor_seed"]))), reasons


def _calibration_events(input_root: Path, qualifications: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    reasons: list[str] = []
    for row in qualifications:
        if row["split"] != "calibration" or not row["qualified_natural_failure"]:
            continue
        event_id = _text(row["event_instance_id"])
        try:
            collector_name = f"init{int(row['init_state_id']):03d}__actor{int(row['actor_seed'])}.json"
        except (TypeError, ValueError):
            collector_name = ""
        candidates = [
            input_root / "phase0b" / "episodes" / row["task"] / collector_name,
            input_root / "phase0b" / "episodes" / row["task"] / f"{event_id}.json",
        ]
        candidate = next((path for path in candidates if path and path.is_file()), candidates[0])
        if not candidate.is_file():
            reasons.append(f"missing calibration episode for {event_id}: {candidate}")
            continue
        try:
            episode = _read_json(candidate)
            if not isinstance(episode, Mapping):
                raise ValueError("episode is not an object")
            if episode.get("status") != "COMPLETE" or not episode.get("qualified_natural_failure"):
                raise ValueError("episode is not a complete qualified failure")
            event = episode.get("event")
            if not isinstance(event, Mapping):
                raise ValueError("qualified episode has no event")
            event = dict(event)
            if event.get("split") != "calibration" or event.get("event_instance_id") != event_id:
                raise ValueError("event identity or split mismatch")
            if _text(event.get("task")) != row["task"] or _text(event.get("init_state_id")) != row["init_state_id"]:
                raise ValueError("event task/init mismatch")
            event["episode_shard_sha256"] = _sha256(candidate)
            event["episode_shard_path"] = str(candidate)
            events.append(event)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            reasons.append(f"invalid calibration episode {candidate}: {exc}")
    keys = [(_text(row.get("task")), _text(row.get("event_instance_id"))) for row in events]
    if len(keys) != len(set(keys)):
        reasons.append("duplicate exported calibration event identity")
    return sorted(events, key=lambda row: (_text(row["task"]), int(row["init_state_id"]), int(row.get("actor_seed", 0)), _text(row["event_instance_id"]))), reasons


def consolidate_phase0b(input_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    input_root, output_root = Path(input_root), Path(output_root)
    qualifications, reasons = _qualification_rows(input_root)
    k1_counts = {
        task: {
            split: sum(
                int(row["qualified_natural_failure"])
                for row in qualifications
                if row["task"] == task and row["split"] == split
            )
            for split in ("infrastructure", "calibration", "evaluation", "reserve")
        }
        for task in TASKS
    }
    k1_pass = all(counts["calibration"] >= 25 and counts["evaluation"] >= 25 for counts in k1_counts.values())
    if reasons:
        return _blocked(
            "phase0b", output_root, reasons,
            qualification_count=len(qualifications),
            k1={"counts": k1_counts, "pass": k1_pass, "source": "qualification metadata only"},
            evaluation_clean_read=False,
            evaluation_outcome_read=False,
        )
    events, event_reasons = _calibration_events(input_root, qualifications)
    if event_reasons:
        return _blocked(
            "phase0b", output_root, event_reasons,
            qualification_count=len(qualifications),
            k1={"counts": k1_counts, "pass": k1_pass, "source": "qualification metadata only"},
            evaluation_clean_read=False,
            evaluation_outcome_read=False,
        )
    qualification_output = [
        {key: value for key, value in row.items() if not str(key).startswith("__")}
        for row in qualifications
    ]
    _write_jsonl(output_root / "phase0b" / "qualification_rows.jsonl", qualification_output)
    _write_jsonl(output_root / "phase0b" / "calibration_events.jsonl", events)
    summary = {
        "schema_version": 1,
        "phase": "phase0b",
        "status": "COMPLETE",
        "blocked": False,
        "qualification_count": len(qualifications),
        "k1": {"counts": k1_counts, "pass": k1_pass, "source": "qualification metadata only"},
        "calibration_event_count": len(events),
        "evaluation_clean_read": False,
        "evaluation_outcome_read": False,
        "selection_eligible": False,
    }
    _write_json(output_root / "phase0b" / "summary.json", summary)
    return summary


def _phase_shard_rows(input_root: Path, phase: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    for task in TASKS:
        material, read_reasons = _read_rows(input_root / phase / "shards" / task)
        reasons.extend(read_reasons)
        rows.extend(material)
    return rows, reasons


def _normalize_shards(input_root: Path, phase: str) -> tuple[list[dict[str, Any]], list[str]]:
    raw, reasons = _phase_shard_rows(input_root, phase)
    normalized: list[dict[str, Any]] = []
    for raw_row in raw:
        row, row_reasons = _canonical_row(raw_row, phase)
        if row is None:
            reasons.extend(f"{raw_row.get('__source_path')}: {reason}" for reason in row_reasons)
            continue
        row["is_reference"] = bool(raw_row.get("is_reference", row["operator"] in REFERENCE_OPERATORS))
        normalized.append(row)
    return normalized, reasons


def _check_duplicates(rows: Sequence[Mapping[str, Any]], reasons: list[str]) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = _branch_key(row)
        if key in seen:
            reasons.append(f"exact duplicate branch key rejected: {key!r}")
        seen.add(key)


def _event_inventory(rows: Sequence[Mapping[str, Any]], phase: str, reasons: list[str]) -> list[tuple[Any, ...]]:
    identities = sorted({_event_key(row) for row in rows})
    tasks = {identity[0] for identity in identities}
    if tasks != set(TASKS):
        reasons.append(f"expected both tasks; observed {sorted(tasks)}")
    if phase == "phase1":
        for task in TASKS:
            count = sum(identity[0] == task for identity in identities)
            if count != 20:
                reasons.append(f"Phase-1 requires exactly 20 events for {task}; observed {count}")
    return identities


def _phase1_reference_events(input_root: Path) -> tuple[list[tuple[Any, ...]], list[str]]:
    """Use the sealed calibration export to reject a residual/substituted grid."""
    path = input_root / "phase0b" / "calibration_events.jsonl"
    if not path.is_file():
        return [], []
    material: list[dict[str, Any]] = []
    reasons: list[str] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"line {line_number} is not an object")
                material.append(dict(value))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return [], [f"cannot read calibration event export: {exc}"]
    expected: list[tuple[Any, ...]] = []
    for task in TASKS:
        task_events = [row for row in material if _text(row.get("task")) == task and row.get("split") == "calibration"]
        task_events.sort(key=lambda row: (int(row["init_state_id"]), int(row.get("actor_seed", row.get("generator_actor_seed", 0))), _text(row["event_instance_id"])))
        if len(task_events) < 20:
            reasons.append(f"calibration event export has fewer than 20 events for {task}")
            continue
        expected.extend(
            (
                task,
                _text(row["event_instance_id"]),
                _text(row["init_state_id"]),
                "calibration",
                _int(row.get("actor_seed", row.get("generator_actor_seed")), "generator_actor_seed"),
            )
            for row in task_events[:20]
        )
    if len(expected) != len(set(expected)):
        reasons.append("calibration event export contains duplicate Phase-1 identities")
    return sorted(expected), reasons


def _expected_grid_keys(events: Sequence[tuple[Any, ...]]) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]]]:
    core: set[tuple[Any, ...]] = set()
    reference: set[tuple[Any, ...]] = set()
    for event in events:
        task, event_id, init_id, split, generator_seed = event
        for recovery_seed in SEEDS:
            for tail, action, cap in BUDGETS:
                for operator in OPERATORS:
                    for prefix in PHASE1_PREFIXES:
                        core.add((task, event_id, init_id, split, generator_seed, recovery_seed, operator, prefix, tail, action, cap))
                for operator, prefix in REFERENCE_PREFIXES.items():
                    reference.add((task, event_id, init_id, split, generator_seed, recovery_seed, operator, prefix, tail, action, cap))
    return core, reference


def _expected_atlas_keys(events: Sequence[tuple[Any, ...]], selected_budget: Mapping[str, Any]) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]]]:
    tail, action, cap = (_int(selected_budget[name], name) for name in ("tail_horizon", "action_budget", "policy_call_cap"))
    core: set[tuple[Any, ...]] = set()
    reference: set[tuple[Any, ...]] = set()
    for task, event_id, init_id, split, generator_seed in events:
        for recovery_seed in SEEDS:
            for operator in OPERATORS:
                for prefix in PHASE2_PREFIXES:
                    core.add((task, event_id, init_id, split, generator_seed, recovery_seed, operator, prefix, tail, action, cap))
            for operator, prefix in REFERENCE_PREFIXES.items():
                reference.add((task, event_id, init_id, split, generator_seed, recovery_seed, operator, prefix, tail, action, cap))
    return core, reference


def _validate_matrix_rows(rows: Sequence[Mapping[str, Any]], events: Sequence[tuple[Any, ...]], phase: str, selected_budget: Mapping[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    reasons: list[str] = []
    _check_duplicates(rows, reasons)
    expected_core, expected_reference = (
        _expected_grid_keys(events) if phase == "phase1" else _expected_atlas_keys(events, selected_budget or {})
    )
    core: list[dict[str, Any]] = []
    reference: list[dict[str, Any]] = []
    for row in rows:
        key = _branch_key(row)
        actual_reference = bool(row.get("is_reference"))
        if actual_reference:
            reference.append(dict(row))
            if key not in expected_reference:
                reasons.append(f"unexpected reference branch key: {key!r}")
            if row["operator"] not in REFERENCE_OPERATORS or row["prefix_k"] != REFERENCE_PREFIXES.get(row["operator"]):
                reasons.append(f"invalid reference operator/prefix: {key!r}")
        else:
            core.append(dict(row))
            if key not in expected_core:
                reasons.append(f"unexpected core branch key: {key!r}")
            if row["operator"] not in OPERATORS:
                reasons.append(f"invalid core operator: {key!r}")
    actual_core = {_branch_key(row) for row in core}
    actual_reference = {_branch_key(row) for row in reference}
    missing_core = expected_core - actual_core
    missing_reference = expected_reference - actual_reference
    if missing_core:
        reasons.append(f"missing {len(missing_core)} core matrix branches")
    if missing_reference:
        reasons.append(f"missing {len(missing_reference)} reference matrix branches")
    if actual_core != expected_core:
        reasons.append(f"core matrix has {len(actual_core)} keys; expected {len(expected_core)}")
    if actual_reference != expected_reference:
        reasons.append(f"reference matrix has {len(actual_reference)} keys; expected {len(expected_reference)}")
    return sorted(core, key=_sort_key), sorted(reference, key=_sort_key), reasons


def _summary_grid(output_root: Path, rows: Sequence[Mapping[str, Any]], reasons: list[str]) -> dict[str, Any]:
    if reasons:
        return _blocked("phase1", output_root, reasons, grid_row_count=len(rows), selection_receipt_written=False)
    try:
        summary = summarize_grid(rows)
    except (TypeError, ValueError, KeyError) as exc:
        return _blocked("phase1", output_root, [f"grid statistics failed closed: {exc}"]) 
    if summary.get("status") != "COMPLETE":
        return _blocked("phase1", output_root, summary.get("blocking_reasons", ["grid statistics is incomplete"]), grid_summary=summary, selection_receipt_written=False)
    receipt = select_budget(summary)
    summary = dict(summary)
    summary["selection"] = receipt
    summary["selection_source"] = "calibration_only"
    _write_json(output_root / "phase1" / "summary.json", summary)
    receipt = dict(receipt)
    receipt["selected_budget"] = receipt.get("selected_budget")
    receipt["selection_source"] = "calibration_only"
    receipt["grid_summary_sha256"] = _sha256(output_root / "phase1" / "summary.json")
    _write_json(output_root / "phase1" / "selection_receipt.json", receipt)
    return summary


def consolidate_phase1(input_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    input_root, output_root = Path(input_root), Path(output_root)
    rows, reasons = _normalize_shards(input_root, "phase1")
    events = _event_inventory(rows, "phase1", reasons)
    reference_events, reference_reasons = _phase1_reference_events(input_root)
    reasons.extend(reference_reasons)
    if reference_events and set(events) != set(reference_events):
        reasons.append("Phase-1 shard event identities do not match the first 20 exported calibration events")
    core, reference, matrix_reasons = _validate_matrix_rows(rows, events, "phase1")
    reasons.extend(matrix_reasons)
    if reasons:
        return _blocked("phase1", output_root, reasons, grid_row_count=len(core), reference_row_count=len(reference), selection_receipt_written=False)
    _write_jsonl(output_root / "phase1" / "grid_rows.jsonl", core)
    _write_jsonl(output_root / "phase1" / "reference_rows.jsonl", reference)
    return _summary_grid(output_root, core, [])


def _read_selection(input_root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        # matrix.selected_budget is the single admission point.  In the
        # integrated tree it verifies the real Git proof and receipt tree
        # membership; duplicating a weaker hash/length check here would create
        # a second, less restrictive evaluation gate.
        try:
            from .matrix import selected_budget as admit_budget
        except ImportError:
            from matrix import selected_budget as admit_budget  # type: ignore
        selected = admit_budget(input_root)
    except Exception as exc:  # matrix deliberately raises on every OPEN_DENY
        return None, [f"evaluation OPEN_DENY: {exc}"]
    if not isinstance(selected, Mapping):
        return None, ["evaluation OPEN_DENY: matrix admission returned no budget"]
    try:
        selected = {name: _int(selected[name], name) for name in ("tail_horizon", "action_budget", "policy_call_cap")}
    except (KeyError, TypeError, ValueError) as exc:
        return None, [f"evaluation OPEN_DENY: invalid admitted budget: {exc}"]
    if tuple(selected.values()) not in BUDGETS:
        return None, [f"evaluation OPEN_DENY: admitted budget is outside preregistered grid: {selected}"]
    return selected, []


def _phase2_event_inventory(input_root: Path, reasons: list[str]) -> list[tuple[Any, ...]]:
    qualifications, qualification_reasons = _qualification_rows(input_root)
    reasons.extend(qualification_reasons)
    events = sorted(
        {
            (row["task"], _text(row["event_instance_id"]), _text(row["init_state_id"]), row["split"], row["actor_seed"])
            for row in qualifications
            if row.get("qualified_natural_failure") and row.get("split") in ("calibration", "evaluation")
        }
    )
    if not events:
        reasons.append("no qualified calibration/evaluation events in qualification metadata")
    return events


def consolidate_phase2(input_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    input_root, output_root = Path(input_root), Path(output_root)
    selected_budget, admission_reasons = _read_selection(input_root)
    if admission_reasons:
        return _blocked("phase2", output_root, admission_reasons, evaluation_read=False, selection_authorized=False)
    # This is the first read of qualification/evaluation-related data.  The
    # receipt binding above is intentionally checked before opening phase2/
    # shards, so an OPEN_DENY test cannot trigger an outcome read.
    rows, reasons = _normalize_shards(input_root, "phase2")
    events = _phase2_event_inventory(input_root, reasons)
    core, reference, matrix_reasons = _validate_matrix_rows(rows, events, "phase2", selected_budget)
    reasons.extend(matrix_reasons)
    if reasons:
        return _blocked("phase2", output_root, reasons, evaluation_read=True, selection_authorized=True, selected_budget=selected_budget)
    _write_jsonl(output_root / "phase2" / "atlas_rows.jsonl", core)
    _write_jsonl(output_root / "phase2" / "reference_rows.jsonl", reference)
    try:
        crossing = analyze_crossing(core, split="evaluation", replicates=10000, seed=216214)
    except (TypeError, ValueError, KeyError) as exc:
        return _blocked("phase2", output_root, [f"crossing statistics failed closed: {exc}"], evaluation_read=True, selection_authorized=True)
    if crossing.get("status") != "COMPLETE":
        return _blocked("phase2", output_root, crossing.get("blocking_reasons", ["crossing statistics is incomplete"]), evaluation_read=True, selection_authorized=True)
    _write_jsonl(output_root / "phase2" / "boundaries.jsonl", crossing.get("boundaries", []))
    _write_json(output_root / "phase2" / "crossing.json", crossing.get("crossing", {}))
    _write_json(output_root / "phase2" / "null_distribution.json", crossing.get("split_half_null", {}))
    summary = {
        "schema_version": 1,
        "phase": "phase2",
        "status": "COMPLETE",
        "blocked": False,
        "selection_authorized": True,
        "selected_budget": selected_budget,
        "atlas_row_count": len(core),
        "reference_row_count": len(reference),
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 216214,
        "k3": crossing.get("k3"),
        "evaluation_read": True,
    }
    _write_json(output_root / "phase2" / "summary.json", summary)
    return summary


def consolidate(phase: str, input_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    if phase == "phase0b":
        return consolidate_phase0b(input_root, output_root)
    if phase == "phase1":
        return consolidate_phase1(input_root, output_root)
    if phase == "phase2":
        return consolidate_phase2(input_root, output_root)
    raise ValueError(f"unknown phase: {phase}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("phase0b", "phase1", "phase2"), required=True)
    parser.add_argument("--input-root", "--input", dest="input_root", type=Path, required=True)
    parser.add_argument("--output-root", "--output", dest="output_root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = consolidate(args.phase, args.input_root, args.output_root)
    print(json.dumps({"phase": args.phase, "status": result.get("status"), "output_root": str(args.output_root)}, sort_keys=True))
    return 0 if result.get("status") != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
