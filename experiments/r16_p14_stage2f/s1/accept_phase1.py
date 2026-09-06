"""Fail-closed final acceptance for the Stage-2F Phase-1 grid.

The checker names every input subtree it may read.  It never walks the
experiment root and cannot accidentally consume sealed evaluation artifacts.
Traces are hashed and, in full mode, decompressed one record at a time to
verify their canonical content digest and normalized contact pairs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

TASKS = (
    "put_the_cream_cheese_in_the_bowl",
    "put_the_bowl_on_the_plate",
)
TASK_HORIZONS = {TASKS[0]: 360, TASKS[1]: 320}
SEEDS = (7, 17, 29)
OPERATORS = (
    "fresh_h4",
    "fresh_h16",
    "hold_1+fresh_h4",
    "rollback_1+fresh_h16",
)
REFERENCE_PREFIXES = {
    "immediate_fresh": 2,
    "fixed_delay_2": 4,
    "fixed_delay_4": 6,
    "fixed_delay_8": 10,
}
PHASE1_PREFIXES = (2, 4, 8, 12, 16)
TAILS = (4, 8, 16)
ACTION_BUDGETS = (8, 16, 32)
POLICY_CALL_CAP = 8
EXPECTED_EVENTS = {TASKS[0]: 19, TASKS[1]: 11}
SOURCE_COMMIT = "a0888d751117cbf7c5a73080d1ee1f421689e8bf"
UID_GID = 2254
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    number = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer")
    return number


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _issue(issues: list[str], message: str) -> None:
    if message not in issues:
        issues.append(message)


def _source_key(task: Any, event_id: Any, init_id: Any, split: Any, seed: Any) -> tuple[str, str, str, str, int]:
    return (str(task), str(event_id), str(init_id), str(split), _int(seed, "generator_actor_seed"))


def _row_key(row: Mapping[str, Any], is_reference: bool) -> tuple[Any, ...]:
    return (
        str(row["task"]), str(row["event_instance_id"]), str(row["init_state_id"]),
        str(row["split"]), _int(row["generator_actor_seed"], "generator_actor_seed"),
        _int(row["recovery_actor_seed"], "recovery_actor_seed"), str(row["operator"]),
        _int(row["prefix_k"], "prefix_k"), _int(row["tail_horizon"], "tail_horizon"),
        _int(row["action_budget"], "action_budget"), _int(row["policy_call_cap"], "policy_call_cap"),
        bool(is_reference),
    )


def _load_source(base: Path, issues: list[str]) -> tuple[list[dict[str, Any]], dict[tuple[str, ...], dict[str, Any]]]:
    path = base / "artifacts" / "stage2f" / "phase0b" / "calibration_events.jsonl"
    if not path.is_file():
        _issue(issues, f"missing calibration source: {path}")
        return [], {}
    events: list[dict[str, Any]] = []
    by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"line {line_number} is not an object")
                task, split = str(value.get("task")), str(value.get("split"))
                if task not in TASKS or split != "calibration":
                    raise ValueError(f"line {line_number} has invalid task/split")
                key = _source_key(
                    task, value.get("event_instance_id"), value.get("init_state_id"), split,
                    value.get("actor_seed", value.get("generator_actor_seed")),
                )
                anchor = _int(value.get("anchor_global_step"), "anchor_global_step")
                if anchor < 0:
                    raise ValueError(f"line {line_number} has negative anchor")
                if key in by_key:
                    raise ValueError(f"duplicate calibration event identity: {key!r}")
                event = {
                    "task": key[0], "event_instance_id": key[1], "init_state_id": key[2],
                    "split": key[3], "generator_actor_seed": key[4],
                    "anchor_global_step": anchor,
                }
                by_key[key] = event
                events.append(event)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, f"invalid calibration source: {exc}")
        return [], {}
    counts = {task: sum(event["task"] == task for event in events) for task in TASKS}
    if counts != EXPECTED_EVENTS:
        _issue(issues, f"calibration event counts {counts!r} != {EXPECTED_EVENTS!r}")
    if len(events) != 30:
        _issue(issues, f"calibration event total {len(events)} != 30")
    events.sort(key=lambda event: (
        int(event["init_state_id"]), int(event["generator_actor_seed"]), event["event_instance_id"]
    ))
    return events, by_key


def _expected_keys(events: Iterable[Mapping[str, Any]]) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]]]:
    core: set[tuple[Any, ...]] = set()
    reference: set[tuple[Any, ...]] = set()
    for event in events:
        event_key = (
            str(event["task"]), str(event["event_instance_id"]), str(event["init_state_id"]),
            str(event["split"]), int(event["generator_actor_seed"]),
        )
        for recovery_seed in SEEDS:
            for tail in TAILS:
                for action_budget in ACTION_BUDGETS:
                    for operator in OPERATORS:
                        for prefix in PHASE1_PREFIXES:
                            core.add(event_key + (
                                recovery_seed, operator, prefix, tail, action_budget,
                                POLICY_CALL_CAP, False,
                            ))
                    for operator, prefix in REFERENCE_PREFIXES.items():
                        reference.add(event_key + (
                            recovery_seed, operator, prefix, tail, action_budget,
                            POLICY_CALL_CAP, True,
                        ))
    return core, reference


def structural_exclusion_is_valid(
    row: Mapping[str, Any], source_event: Mapping[str, Any]
) -> tuple[bool, str]:
    """Validate a horizon marker against the immutable calibration source."""
    task = str(row.get("task"))
    if not str(row.get("status")).upper().startswith("BLOCKED") or str(row.get("error_type")) != "PrefixOutsideTaskHorizon":
        return False, "row is not a PrefixOutsideTaskHorizon structural marker"
    if task not in TASK_HORIZONS:
        return False, f"unknown task horizon: {task!r}"
    try:
        prefix = _int(row["prefix_k"], "prefix_k")
        anchor = _int(source_event["anchor_global_step"], "anchor_global_step")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return False, f"structural source anchor/prefix unavailable: {exc}"
    horizon = TASK_HORIZONS[task]
    if anchor + prefix <= horizon:
        return False, (
            "structural marker is feasible: "
            f"anchor_global_step={anchor} + prefix_k={prefix} <= task_horizon={horizon}"
        )
    row_anchor = row.get("anchor_global_step")
    if row_anchor is not None:
        try:
            if _int(row_anchor, "row anchor_global_step") != anchor:
                return False, "row anchor_global_step disagrees with calibration source"
        except (TypeError, ValueError, OverflowError) as exc:
            return False, f"invalid row anchor_global_step: {exc}"
    return True, ""


def _job_status(job: Mapping[str, Any]) -> str | None:
    readback = job.get("readback")
    if isinstance(readback, Mapping) and readback.get("Status") is not None:
        return str(readback["Status"])
    return str(job["status"]) if job.get("status") is not None else None


def _job_sort_key(job: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(job.get("created_at_utc") or ""),
        str(job.get("run_id") or ""),
        str(job.get("job_id") or ""),
    )


def _load_grid_jobs(
    base: Path, issues: list[str], allowed_source_commits: set[str]
) -> tuple[dict[str, Mapping[str, Any]], bool, dict[str, list[Mapping[str, Any]]]]:
    path = base / "control" / "jobs.json"
    if not path.is_file():
        _issue(issues, f"missing control jobs metadata: {path}")
        return {}, False, {}
    try:
        payload = _read_json(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, f"invalid control jobs metadata: {exc}")
        return {}, False, {}
    jobs = payload.get("jobs") if isinstance(payload, Mapping) else None
    if not isinstance(jobs, list):
        _issue(issues, "control jobs metadata has no jobs list")
        return {}, False, {}
    grid = [job for job in jobs if isinstance(job, Mapping) and str(job.get("phase")) == "grid"]
    by_task_all: dict[str, list[Mapping[str, Any]]] = {task: [] for task in TASKS}
    for job in grid:
        task = str(job.get("task"))
        if task not in by_task_all:
            _issue(issues, f"unknown Phase-1 grid task binding: {task!r}")
            continue
        by_task_all[task].append(job)
    if any(not by_task_all[task] for task in TASKS):
        return {}, False, by_task_all
    selected: dict[str, Mapping[str, Any]] = {}
    terminal = True
    for task, candidates in by_task_all.items():
        ordered = sorted(candidates, key=_job_sort_key)
        by_task_all[task] = ordered
        selected_job = ordered[-1]
        selected[task] = selected_job
        for candidate in ordered:
            if candidate.get("source_commit") not in allowed_source_commits:
                _issue(issues, f"{task}: historical grid source commit is not allowed")
            if candidate.get("gpus") != 2:
                _issue(issues, f"{task}: historical grid job does not declare gpus=2")
            if not candidate.get("artifact_dir"):
                _issue(issues, f"{task}: historical grid artifact_dir missing")
        if _job_status(selected_job) != "Succeeded":
            terminal = False
        readback = selected_job.get("readback")
        if not isinstance(readback, Mapping):
            _issue(issues, f"{task}: latest grid job missing canonical PAI readback")
        elif readback.get("JobId") not in (None, selected_job.get("job_id")):
            _issue(issues, f"{task}: latest grid readback JobId mismatch")
    return selected, terminal, by_task_all


def _load_completion(
    base: Path, events: Iterable[Mapping[str, Any]], issues: list[str]
) -> tuple[dict[str, Mapping[str, Any]], bool]:
    completions: dict[str, Mapping[str, Any]] = {}
    source_counts = {task: sum(event["task"] == task for event in events) for task in TASKS}
    present = True
    for task in TASKS:
        path = base / "artifacts" / "stage2f" / "phase1" / f"completion_{task}.json"
        if not path.is_file():
            present = False
            continue
        try:
            payload = _read_json(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            _issue(issues, f"{task}: invalid completion JSON: {exc}")
            continue
        if not isinstance(payload, Mapping):
            _issue(issues, f"{task}: completion JSON is not an object")
            continue
        completions[task] = payload
        if payload.get("task") not in (None, task):
            _issue(issues, f"{task}: completion task mismatch")
        if payload.get("phase") not in (None, "grid"):
            _issue(issues, f"{task}: completion phase mismatch")
        rows_per_event = 3 * 9 * (4 * 5 + 4)
        for name, expected in (
            ("events", source_counts[task]),
            ("planned_events", 20),
            ("requested_rows", source_counts[task] * rows_per_event),
            ("persisted_rows", source_counts[task] * rows_per_event),
        ):
            if name in payload and payload.get(name) != expected:
                _issue(issues, f"{task}: completion {name}={payload.get(name)!r}, expected {expected}")
        if payload.get("planned_sample_complete") is not (source_counts[task] == 20):
            _issue(issues, f"{task}: planned_sample_complete mismatch")
        if payload.get("missing_planned_events") != max(0, 20 - source_counts[task]):
            _issue(issues, f"{task}: missing_planned_events mismatch")
    return completions, present and set(completions) == set(TASKS)


def _validate_state_metadata(
    base: Path, jobs: Mapping[str, Mapping[str, Any]], issues: list[str],
    allowed_source_commits: set[str],
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for task, job in jobs.items():
        artifact_raw = job.get("artifact_dir")
        if not isinstance(artifact_raw, str):
            _issue(issues, f"{task}: invalid artifact_dir")
            continue
        artifact = Path(artifact_raw).resolve()
        pai_runs = (base / "pai_runs").resolve()
        if not _under(artifact, pai_runs):
            _issue(issues, f"{task}: artifact_dir escapes base/pai_runs")
            continue
        state_dir = artifact / "pai_state"
        state: dict[str, Any] = {}
        for name in ("RUNTIME.json", "ENVIRONMENT.json", "COMPLETED.json"):
            path = state_dir / name
            if not path.is_file():
                _issue(issues, f"{task}: missing pai_state/{name}")
                continue
            try:
                value = _read_json(path)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                _issue(issues, f"{task}: invalid pai_state/{name}: {exc}")
                continue
            if not isinstance(value, Mapping):
                _issue(issues, f"{task}: pai_state/{name} is not an object")
                continue
            state[name] = value
            if value.get("uid") != UID_GID or value.get("gid") != UID_GID:
                _issue(issues, f"{task}: pai_state/{name} is not owned by 2254:2254")
        runtime, environment, completed = (
            state.get("RUNTIME.json"), state.get("ENVIRONMENT.json"), state.get("COMPLETED.json")
        )
        if isinstance(runtime, Mapping):
            if runtime.get("phase") != "grid" or runtime.get("task") not in (None, task):
                _issue(issues, f"{task}: RUNTIME.json phase/task mismatch")
            if runtime.get("source_commit") not in allowed_source_commits:
                _issue(issues, f"{task}: RUNTIME.json source commit is not allowed")
            if runtime.get("job_id") != job.get("job_id"):
                _issue(issues, f"{task}: RUNTIME.json job_id mismatch")
        if isinstance(environment, Mapping):
            if environment.get("source_commit") not in allowed_source_commits:
                _issue(issues, f"{task}: ENVIRONMENT.json source commit is not allowed")
            if environment.get("job_id") != job.get("job_id"):
                _issue(issues, f"{task}: ENVIRONMENT.json job_id mismatch")
            if not isinstance(environment.get("runtime_identity"), Mapping):
                _issue(issues, f"{task}: ENVIRONMENT.json runtime_identity missing")
        if isinstance(completed, Mapping):
            if completed.get("phase") != "grid" or completed.get("task") not in (None, task):
                _issue(issues, f"{task}: COMPLETED.json phase/task mismatch")
            if completed.get("source_commit") not in allowed_source_commits:
                _issue(issues, f"{task}: COMPLETED.json source commit is not allowed")
        metadata[task] = {
            "artifact_dir": str(artifact), "run_id": job.get("run_id"),
            "job_id": job.get("job_id"), "state": state,
        }
    return metadata


def _valid_hex(value: Any) -> bool:
    return isinstance(value, str) and bool(HEX64.fullmatch(value))


def _check_reconstruction(row: Mapping[str, Any], issues: list[str], label: str) -> None:
    reconstruction = row.get("reconstruction")
    if not isinstance(reconstruction, Mapping):
        _issue(issues, f"{label}: reconstruction receipt missing")
        return
    for key in (
        "action_history_exact", "actor_inference_side_effect_free", "anchor_state_exact",
        "event_chunk_exact", "state_history_exact",
    ):
        if reconstruction.get(key) is not True:
            _issue(issues, f"{label}: reconstruction {key} is not true")
    try:
        if float(reconstruction.get("max_anchor_state_error")) > 1e-9:
            _issue(issues, f"{label}: reconstruction error exceeds 1e-9")
    except (TypeError, ValueError):
        _issue(issues, f"{label}: reconstruction max_anchor_state_error missing")


def _check_d4(row: Mapping[str, Any], issues: list[str], label: str) -> None:
    signatures = row.get("d4_signatures")
    if not isinstance(signatures, Mapping):
        _issue(issues, f"{label}: d4_signatures missing")
        return
    for part in ("detection", "pre_tail", "final"):
        value = signatures.get(part)
        if not isinstance(value, Mapping) or not _valid_hex(value.get("complete_signature_hash")):
            _issue(issues, f"{label}: d4 {part} signature missing")


def _check_trace(
    row: Mapping[str, Any], path: Path, mode: str, issues: list[str], trace_report: dict[str, Any]
) -> None:
    label = str(path)
    try:
        raw_sha, raw_bytes = _sha256_file(path)
    except OSError as exc:
        _issue(issues, f"{label}: cannot hash trace: {exc}")
        return
    trace_report["bytes"] += raw_bytes
    trace_report["files"] += 1
    trace_report["bytes_by_task"][str(row["task"])] += raw_bytes
    if row.get("trace_sha256") != raw_sha:
        _issue(issues, f"{label}: trace_sha256 mismatch")
    if mode == "hash-only":
        trace_report["scope_note"] = "raw compressed SHA-256 only; contact content was not parsed"
        return
    content_sha = hashlib.sha256()
    count, nonempty_normalized = 0, False
    try:
        with gzip.open(path, "rb") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    raise ValueError("blank trace line")
                record = json.loads(raw_line)
                if not isinstance(record, Mapping):
                    raise ValueError("trace record is not an object")
                canonical = json.dumps(
                    record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8") + b"\n"
                content_sha.update(canonical)
                count += 1
                if record.get("event_instance_id") != row.get("event_instance_id"):
                    raise ValueError("event_instance_id mismatch")
                if record.get("operator") != row.get("operator"):
                    raise ValueError("operator mismatch")
                if _int(record.get("prefix_k"), "trace prefix_k") != _int(row["prefix_k"], "prefix_k"):
                    raise ValueError("prefix_k mismatch")
                if _int(record.get("actor_seed"), "trace actor_seed") != _int(row["recovery_actor_seed"], "recovery_actor_seed"):
                    raise ValueError("actor_seed mismatch")
                pairs = record.get("normalized_contact_pairs")
                if not isinstance(pairs, list):
                    raise ValueError("normalized_contact_pairs missing")
                for pair in pairs:
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        raise ValueError("normalized contact pair is not a two-item pair")
                    if not all(isinstance(item, str) and item for item in pair):
                        raise ValueError("normalized contact pair contains a non-string")
                nonempty_normalized |= bool(pairs)
                if record.get("contact_stream_available") is not True:
                    raise ValueError("contact stream is not marked available")
    except (OSError, EOFError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, f"{label}: invalid gzip trace: {exc}")
        return
    trace_report["records"] += count
    if row.get("trace_records") != count:
        _issue(issues, f"{label}: trace_records mismatch")
    if row.get("trace_content_sha256") != content_sha.hexdigest():
        _issue(issues, f"{label}: trace_content_sha256 mismatch")
    if count == 0 or not nonempty_normalized:
        _issue(issues, f"{label}: no normalized contact pairs")
    labels = row.get("labels")
    if not isinstance(labels, Mapping) or labels.get("record_count") != count or labels.get("label_complete") is not True:
        _issue(issues, f"{label}: label receipt incomplete")


def _validate_rows(
    base: Path,
    events: list[dict[str, Any]],
    source_by_key: Mapping[tuple[str, ...], Mapping[str, Any]],
    jobs: Mapping[str, Mapping[str, Any]],
    job_history: Mapping[str, list[Mapping[str, Any]]],
    mode: str,
    allowed_source_commits: set[str],
    issues: list[str],
) -> dict[str, Any]:
    phase1 = base / "artifacts" / "stage2f" / "phase1"
    expected_core, expected_reference = _expected_keys(events)
    expected_all = expected_core | expected_reference
    observed: set[tuple[Any, ...]] = set()
    duplicate_keys: list[str] = []
    counts = {
        task: {"rows": 0, "core_rows": 0, "reference_rows": 0,
               "complete_rows": 0, "structural_rows": 0}
        for task in TASKS
    }
    trace_rows: dict[Path, Mapping[str, Any]] = {}
    runtime_receipts: set[str] = set()
    structural_by_task = {task: 0 for task in TASKS}
    for task in TASKS:
        shard_dir = phase1 / "shards" / task
        paths = sorted(shard_dir.glob("*.json")) if shard_dir.is_dir() else []
        expected_task_rows = sum(1 for key in expected_all if key[0] == task)
        if len(paths) != expected_task_rows:
            _issue(issues, f"{task}: shard file count {len(paths)} != expected {expected_task_rows}")
        for path in paths:
            label = str(path)
            try:
                row = _read_json(path)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                _issue(issues, f"{label}: invalid shard JSON: {exc}")
                continue
            if not isinstance(row, Mapping):
                _issue(issues, f"{label}: shard is not an object")
                continue
            counts[task]["rows"] += 1
            if str(row.get("task")) != task:
                _issue(issues, f"{label}: task/path mismatch")
                continue
            try:
                is_reference = row.get("is_reference")
                if not isinstance(is_reference, bool):
                    raise ValueError("is_reference must be boolean")
                key = _row_key(row, is_reference)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                _issue(issues, f"{label}: invalid branch key: {exc}")
                continue
            if key in observed:
                duplicate_keys.append(repr(key))
            observed.add(key)
            counts[task]["reference_rows" if is_reference else "core_rows"] += 1
            source_key = (
                str(row.get("task")), str(row.get("event_instance_id")),
                str(row.get("init_state_id")), str(row.get("split")),
                row.get("generator_actor_seed"),
            )
            source_event = source_by_key.get(source_key)
            if source_event is None:
                _issue(issues, f"{label}: row event identity does not match calibration source")
                continue
            operator = str(row.get("operator"))
            expected_reference = operator in REFERENCE_PREFIXES
            if is_reference != expected_reference:
                _issue(issues, f"{label}: is_reference/operator mismatch")
            if key not in expected_all:
                _issue(issues, f"{label}: branch key outside expected Phase-1 grid")
            if str(row.get("split")) != "calibration":
                _issue(issues, f"{label}: split is not calibration")
            if int(row.get("generator_actor_seed", -1)) != int(source_event["generator_actor_seed"]):
                _issue(issues, f"{label}: generator actor seed mismatch")
            if operator not in OPERATORS + tuple(REFERENCE_PREFIXES):
                _issue(issues, f"{label}: unknown operator {operator!r}")
            if str(row.get("behavior_operator", operator)) != operator:
                _issue(issues, f"{label}: behavior_operator mismatch")
            if row.get("configured_budget") != {
                "action_budget": row.get("action_budget"),
                "policy_call_cap": row.get("policy_call_cap"),
                "tail_horizon": row.get("tail_horizon"),
            }:
                _issue(issues, f"{label}: configured_budget mismatch")
            if row.get("task_horizon") not in (None, TASK_HORIZONS[task]):
                _issue(issues, f"{label}: task_horizon mismatch")
            status = str(row.get("status"))
            if status == "COMPLETE":
                counts[task]["complete_rows"] += 1
                for name in ("pid", "env_hash", "chunk_hash", "source_commit", "runtime_receipt_sha256"):
                    if row.get(name) in (None, ""):
                        _issue(issues, f"{label}: missing {name}")
                if not isinstance(row.get("pid"), int) or isinstance(row.get("pid"), bool) or row.get("pid") <= 0:
                    _issue(issues, f"{label}: invalid pid")
                if not _valid_hex(row.get("env_hash")) or not _valid_hex(row.get("chunk_hash")):
                    _issue(issues, f"{label}: invalid env_hash/chunk_hash")
                if row.get("source_commit") not in allowed_source_commits:
                    _issue(issues, f"{label}: source commit is not allowed")
                if not _valid_hex(row.get("runtime_receipt_sha256")):
                    _issue(issues, f"{label}: invalid runtime receipt hash")
                runtime_receipts.add(str(row.get("runtime_receipt_sha256")))
                bindings = {
                    (str(candidate.get("job_id")), str(candidate.get("run_id")))
                    for candidate in job_history.get(task, [jobs[task]])
                }
                if (str(row.get("job_id")), str(row.get("pai_run_id"))) not in bindings:
                    _issue(issues, f"{label}: row job/run binding is not in Phase-1 grid history")
                try:
                    if _int(row["anchor_global_step"], "anchor_global_step") != int(source_event["anchor_global_step"]):
                        _issue(issues, f"{label}: anchor_global_step mismatch")
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    _issue(issues, f"{label}: invalid anchor_global_step: {exc}")
                if row.get("safe_success") not in (True, False, 0, 1):
                    _issue(issues, f"{label}: safe_success is not boolean")
                if row.get("error") not in (None, ""):
                    _issue(issues, f"{label}: COMPLETE row has error")
                if row.get("error_type") not in (None, ""):
                    _issue(issues, f"{label}: COMPLETE row has error_type")
                _check_reconstruction(row, issues, label)
                _check_d4(row, issues, label)
                if not isinstance(row.get("labels"), Mapping):
                    _issue(issues, f"{label}: labels missing")
                trace_raw = row.get("trace_path")
                if not isinstance(trace_raw, str):
                    _issue(issues, f"{label}: trace_path missing")
                else:
                    trace_path = Path(trace_raw).resolve()
                    trace_root = (phase1 / "contact_topology" / task).resolve()
                    if not _under(trace_path, trace_root) or trace_path.suffixes[-2:] != [".jsonl", ".gz"]:
                        _issue(issues, f"{label}: trace path escapes Phase-1 topology root")
                    elif trace_path in trace_rows:
                        _issue(issues, f"{label}: duplicate trace path")
                    else:
                        trace_rows[trace_path] = row
            elif status.startswith("BLOCKED") and str(row.get("error_type")) == "PrefixOutsideTaskHorizon":
                valid, reason = structural_exclusion_is_valid(row, source_event)
                if not valid:
                    _issue(issues, f"{label}: {reason}")
                else:
                    counts[task]["structural_rows"] += 1
                    structural_by_task[task] += 1
            else:
                _issue(issues, f"{label}: unknown/error row cannot count as failure")
    if duplicate_keys:
        _issue(issues, f"duplicate branch keys: {duplicate_keys[:5]!r}")
    missing, extra = expected_all - observed, observed - expected_all
    if missing:
        _issue(issues, f"missing expected branch keys: {len(missing)}")
    if extra:
        _issue(issues, f"unexpected branch keys: {len(extra)}")
    expected_structural = {TASKS[0]: 0, TASKS[1]: 378}
    if structural_by_task != expected_structural:
        _issue(issues, f"structural rows {structural_by_task!r} != {expected_structural!r}")
    for task in TASKS:
        expected_core_count = EXPECTED_EVENTS[task] * 3 * 9 * 4 * 5
        expected_ref_count = EXPECTED_EVENTS[task] * 3 * 9 * 4
        if counts[task]["core_rows"] != expected_core_count:
            _issue(issues, f"{task}: core row count mismatch")
        if counts[task]["reference_rows"] != expected_ref_count:
            _issue(issues, f"{task}: reference row count mismatch")
    trace_roots = {task: phase1 / "contact_topology" / task for task in TASKS}
    actual_trace_paths: set[Path] = set()
    listed_bytes_by_task, listed_files_by_task = {}, {}
    for task, directory in trace_roots.items():
        actual = {path.resolve() for path in directory.glob("*.jsonl.gz")} if directory.is_dir() else set()
        actual_trace_paths |= actual
        listed_files_by_task[task] = len(actual)
        listed_bytes_by_task[task] = sum(path.stat().st_size for path in actual if path.is_file())
    if actual_trace_paths != set(trace_rows):
        _issue(issues, "trace file set differs from COMPLETE shard references")
    trace_report = {
        "scope": mode,
        "scope_note": (
            "raw compressed SHA-256 only; contact content was not parsed"
            if mode == "hash-only"
            else "gzip JSONL parsed record-by-record with canonical content SHA-256"
        ),
        "files": 0, "records": 0, "bytes": 0,
        "bytes_by_task": {task: 0 for task in TASKS},
        "files_by_task": {task: 0 for task in TASKS},
        "listed_files_by_task": listed_files_by_task,
        "listed_bytes_by_task": listed_bytes_by_task,
    }
    for path, row in sorted(trace_rows.items(), key=lambda item: str(item[0])):
        _check_trace(row, path, mode, issues, trace_report)
        trace_report["files_by_task"][str(row["task"])] += 1
    return {
        "counts": counts,
        "expected_rows_by_task": {task: EXPECTED_EVENTS[task] * 3 * 9 * (4 * 5 + 4) for task in TASKS},
        "observed_rows": len(observed), "expected_rows": len(expected_all),
        "missing_key_count": len(missing), "unexpected_key_count": len(extra),
        "duplicate_key_count": len(duplicate_keys),
        "runtime_receipt_hashes": sorted(runtime_receipts),
        "trace": trace_report,
    }


def evaluate_base(
    base: str | Path, mode: str = "full", allowed_source_commits: Iterable[str] | None = None
) -> dict[str, Any]:
    if mode not in ("full", "hash-only"):
        raise ValueError(f"unsupported trace mode: {mode}")
    allowed = set(allowed_source_commits or (SOURCE_COMMIT,))
    if not allowed or any(not HEX40.fullmatch(value) for value in allowed):
        raise ValueError("allowed source commits must be 40-character lowercase hex")
    base_path = Path(base).resolve()
    report: dict[str, Any] = {
        "schema_version": 1, "status": "INCOMPLETE", "base": str(base_path),
        "phase": "phase1", "trace_validation_scope": mode,
        "evaluation_raw_read": False, "source_commit": SOURCE_COMMIT,
        "allowed_source_commits": sorted(allowed),
        "uid_gid": [UID_GID, UID_GID], "blocking_reasons": [],
        "incomplete_reasons": [], "checks": {},
    }
    if not base_path.is_dir():
        report["status"] = "BLOCKED"
        report["blocking_reasons"].append(f"base does not exist: {base_path}")
        return report
    source_issues: list[str] = []
    events, source_by_key = _load_source(base_path, source_issues)
    report["source"] = {
        "events_by_task": {task: sum(event["task"] == task for event in events) for task in TASKS},
        "total_events": len(events),
    }
    report["blocking_reasons"].extend(source_issues)
    report["checks"]["calibration_source"] = not source_issues
    jobs_issues: list[str] = []
    jobs, terminal, job_history = _load_grid_jobs(base_path, jobs_issues, allowed)
    report["blocking_reasons"].extend(jobs_issues)
    report["jobs"] = {
        task: {
            "job_id": job.get("job_id"), "run_id": job.get("run_id"),
            "status": _job_status(job), "artifact_dir": job.get("artifact_dir"),
            "selected_latest": True,
        }
        for task, job in jobs.items()
    }
    report["job_history"] = {
        task: [
            {
                "job_id": candidate.get("job_id"),
                "run_id": candidate.get("run_id"),
                "status": _job_status(candidate),
                "source_commit": candidate.get("source_commit"),
                "artifact_dir": candidate.get("artifact_dir"),
            }
            for candidate in candidates
        ]
        for task, candidates in job_history.items()
    }
    report["checks"]["grid_jobs_terminal_succeeded"] = terminal and not jobs_issues and set(jobs) == set(TASKS)
    if source_issues:
        report["status"] = "BLOCKED"
        return report
    if not terminal or set(jobs) != set(TASKS):
        report["status"] = "INCOMPLETE"
        report["incomplete_reasons"].append("both Phase-1 grid jobs are not terminal Succeeded")
        return report
    completion_issues: list[str] = []
    completions, completion_present = _load_completion(base_path, events, completion_issues)
    report["completions"] = {task: dict(value) for task, value in completions.items()}
    report["blocking_reasons"].extend(completion_issues)
    report["checks"]["task_completions_present"] = completion_present and not completion_issues
    if completion_issues:
        report["status"] = "BLOCKED"
        return report
    if not completion_present:
        report["status"] = "INCOMPLETE"
        report["incomplete_reasons"].append("one or both task completion JSON files are absent")
        return report
    state_issues: list[str] = []
    metadata = _validate_state_metadata(base_path, jobs, state_issues, allowed)
    report["pai_state"] = metadata
    report["blocking_reasons"].extend(state_issues)
    report["checks"]["pai_state_complete_and_owned"] = not state_issues and set(metadata) == set(TASKS)
    if state_issues:
        report["status"] = "BLOCKED"
        return report
    row_issues: list[str] = []
    report["rows"] = _validate_rows(
        base_path, events, source_by_key, jobs, job_history, mode, allowed, row_issues
    )
    report["blocking_reasons"].extend(row_issues)
    report["checks"]["phase1_grid_keys_and_provenance"] = not row_issues
    report["status"] = "PASS" if not report["blocking_reasons"] else "BLOCKED"
    return report


def write_report(report: Mapping[str, Any], output: str | Path, base: str | Path) -> Path:
    output_path, base_path = Path(output).resolve(), Path(base).resolve()
    if _under(output_path, base_path):
        raise ValueError("acceptance output must be outside the GPU artifact base")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="explicit Stage-2F external artifact base")
    parser.add_argument("--output", required=True, help="explicit acceptance JSON output path")
    parser.add_argument("--mode", choices=("full", "hash-only"), default="full")
    parser.add_argument("--allowed-source-commit", action="append", dest="allowed_source_commits")
    args = parser.parse_args(argv)
    report = evaluate_base(args.base, args.mode, args.allowed_source_commits)
    write_report(report, args.output, args.base)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

