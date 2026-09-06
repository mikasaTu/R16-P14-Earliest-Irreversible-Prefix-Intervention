"""S1 exact configured grid and atlas with sealed evaluation admission."""
from __future__ import annotations

import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import (
    SEEDS,
    OPERATORS,
    TASKS,
    atomic_json,
    digest,
    file_sha,
    guard_execution,
)

BUDGETS = tuple(itertools.product((4, 8, 16), (8, 16, 32)))
GRID_K = (2, 4, 8, 12, 16)
ATLAS_K = (2, 4, 6, 8, 10, 12, 14, 16)
REFERENCES = (
    ("immediate_fresh", 2),
    ("fixed_delay_2", 4),
    ("fixed_delay_4", 6),
    ("fixed_delay_8", 10),
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROTOCOL = _REPO_ROOT / "experiments/r16_p14_stage2f/DIAGNOSTIC_ATLAS_CONTINUATION.md"
_APPROVED_PROTOCOL_SHA256 = "2d14427c2391422dd372b0528d6d784d1066d8d059e1405597cfb1954f260c6a"
_DIAGNOSTIC_RECEIPT_REL = "phase1/diagnostic_selection_receipt.json"


def load_events(root, task, split, *, diagnostic_atlas=False):
    if split == "evaluation":
        if diagnostic_atlas:
            diagnostic_selected_budget(root)
        else:
            selected_budget(root)
    folder = "sealed_evaluation" if split == "evaluation" else "episodes"
    events = []
    for p in sorted((Path(root) / "phase0b" / folder / task).glob("*.json")):
        row = json.loads(p.read_text())
        if row.get("task") != task or row.get("status") != "COMPLETE":
            raise RuntimeError("invalid event shard binding")
        if row["split"] == split and row["qualified_natural_failure"]:
            if row.get("event") is None:
                raise RuntimeError("qualified row lacks event")
            event = row["event"]
            if (
                event.get("task") != task
                or event.get("split") != split
                or event.get("init_state_id") != row["init_state_id"]
            ):
                raise RuntimeError("event/shard identity mismatch")
            event["episode_shard_sha256"] = file_sha(p)
            events.append(event)
    return sorted(
        events,
        key=lambda e: (
            int(e["init_state_id"]),
            int(e["actor_seed"]),
            e["event_instance_id"],
        ),
    )


def selected_budget(root, *, diagnostic_atlas=False):
    """Admit the formal selection; opt in explicitly for diagnostic atlas."""
    if diagnostic_atlas:
        return diagnostic_selected_budget(root)
    root = Path(root)
    receipt = root / "phase1/selection_receipt.json"
    authorization = root / "phase1/selection_authorization.json"
    if not receipt.is_file() or not authorization.is_file():
        raise RuntimeError("evaluation OPEN_DENY: selection not committed")
    auth = json.loads(authorization.read_text())
    from .selection import verify_commit_proof

    verify_commit_proof(receipt, auth)
    data = json.loads(receipt.read_text())
    if (
        data.get("status") != "SELECTED"
        or data.get("selection_source") != "calibration_only"
    ):
        raise RuntimeError("no valid calibration selection")
    if data.get("planned_sample_complete") is False or data.get("sample_complete") is False:
        raise RuntimeError("evaluation OPEN_DENY: planned calibration sample incomplete")
    budget = data.get("selected_budget") or {}
    if (
        budget.get("tail_horizon") not in (4, 8, 16)
        or budget.get("action_budget") not in (8, 16, 32)
        or budget.get("policy_call_cap") != 8
    ):
        raise RuntimeError("budget outside preregistered grid")
    candidates = data.get("ranked_candidates", [])
    if (
        not candidates
        or candidates[0].get("budget") != budget
        or candidates[0].get("qualifies") is not True
    ):
        raise RuntimeError("selection rank binding invalid")
    for field, check in (
        ("oracle_by_task", lambda x: 0.25 <= x <= 0.85),
        ("gap_by_task", lambda x: x >= 0.15),
    ):
        values = candidates[0].get(field, {})
        if set(values) != set(TASKS) or not all(check(float(x)) for x in values.values()):
            raise RuntimeError("K2 two-task qualification missing")
    return data["selected_budget"]


def _recompute_diagnostic_receipt(summary_path, summary_sha256, protocol_sha256):
    """Recompute calibration selection from the bound summary before eval."""
    try:
        from .diagnostic_selection import build_diagnostic_selection_receipt
    except ImportError as exc:
        raise RuntimeError(
            "diagnostic selection implementation missing from payload"
        ) from exc
    try:
        summary = json.loads(Path(summary_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("diagnostic calibration summary unreadable") from exc
    try:
        return build_diagnostic_selection_receipt(
            summary,
            input_summary_sha256=summary_sha256,
            input_summary_path=summary_path,
            diagnostic_continuation_doc_sha256=protocol_sha256,
        )
    except Exception as exc:
        raise RuntimeError("diagnostic calibration selection recomputation failed") from exc


def _assert_diagnostic_recomputed(receipt, recomputed):
    fields = (
        "status",
        "selection_source",
        "confirmatory",
        "input_sha256",
        "protocol_sha256",
        "selected_budget",
        "original_rank",
        "all_candidates",
        "selected_candidate",
        "k2_numeric_status",
        "fallback_used",
    )
    for field in fields:
        if recomputed.get(field) != receipt.get(field):
            raise RuntimeError(
                f"diagnostic selection recomputation mismatch: {field}"
            )


def _diagnostic_binding(root):
    """Verify proof and recompute calibration selection before evaluation."""
    root = Path(root)
    receipt = root / _DIAGNOSTIC_RECEIPT_REL
    authorization = root / "phase1/diagnostic_selection_authorization.json"
    if not receipt.is_file() or not authorization.is_file():
        raise RuntimeError("evaluation OPEN_DENY: diagnostic selection not committed")
    protocol_sha256 = file_sha(_PROTOCOL)
    if protocol_sha256 != _APPROVED_PROTOCOL_SHA256:
        raise RuntimeError("diagnostic protocol SHA256 is not the approved document")
    auth = json.loads(authorization.read_text())
    from .selection import validate_diagnostic_receipt, verify_commit_proof

    verify_commit_proof(receipt, auth, diagnostic=True)
    data = validate_diagnostic_receipt(
        json.loads(receipt.read_text()),
        protocol_sha256=protocol_sha256,
    )
    summary_path = root / "phase1/summary.json"
    if not summary_path.is_file():
        raise RuntimeError("evaluation OPEN_DENY: calibration summary missing")
    summary_sha256 = file_sha(summary_path)
    if data.get("input_sha256") != summary_sha256:
        raise RuntimeError("diagnostic input summary SHA256 mismatch")
    if data.get("input_summary_sha256", summary_sha256) != summary_sha256:
        raise RuntimeError("diagnostic input summary alias SHA256 mismatch")
    recomputed = _recompute_diagnostic_receipt(
        summary_path, summary_sha256, protocol_sha256
    )
    _assert_diagnostic_recomputed(data, recomputed)
    return {
        "receipt_sha256": file_sha(receipt),
        "receipt_path": _DIAGNOSTIC_RECEIPT_REL,
        "input_sha256": summary_sha256,
        "protocol_sha256": protocol_sha256,
        "original_rank": int(data["original_rank"]),
        "selected_budget": dict(data["selected_budget"]),
    }


def diagnostic_selected_budget(root):
    """Admit only a valid, independently sealed diagnostic selection."""
    return _diagnostic_binding(root)["selected_budget"]


def selected_diagnostic_budget(root):
    """Stable alias used by diagnostic statistics/atlas callers."""
    return diagnostic_selected_budget(root)


def compatible_runtime_row(row, module_hashes):
    previous = row.get("runtime_module_hashes")
    if previous == module_hashes:
        return True
    receipt = json.loads(
        (Path(__file__).parent / "dispatch_compatibility.json").read_text()
    )
    if row.get("source_commit") not in receipt["allowed_existing_source_commits"]:
        return False
    if module_hashes.get("dispatch.py") != receipt["new_dispatch_sha256"]:
        return False
    if not isinstance(previous, dict) or set(previous) != set(module_hashes):
        return False
    for name, expected in receipt["other_module_sha256"].items():
        if previous.get(name) != expected or module_hashes.get(name) != expected:
            return False
    return previous.get("dispatch.py") == receipt["old_dispatch_sha256"]


def record_first_work(row_path):
    state_dir = os.environ.get("PAI_CANARY_RUN_DIR")
    if not state_dir:
        return
    import fcntl

    first = Path(state_dir) / "pai_state/FIRST_REAL_WORK.json"
    first.parent.mkdir(parents=True, exist_ok=True)
    with first.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not first.exists():
            atomic_json(
                first,
                {
                    "uid": os.getuid(),
                    "gid": os.getgid(),
                    "shard": str(row_path),
                    "sha256": file_sha(row_path),
                },
            )


def _attach_diagnostic_binding(row, binding):
    """Annotate a row while preserving its original source and trace fields."""
    row["diagnostic_continuation"] = True
    row["diagnostic_selection_receipt_sha256"] = binding["receipt_sha256"]
    row["diagnostic_selection_receipt_path"] = binding["receipt_path"]
    row["diagnostic_selection_binding"] = {
        "input_sha256": binding["input_sha256"],
        "protocol_sha256": binding["protocol_sha256"],
        "original_rank": binding["original_rank"],
    }
    return row


def run_task(
    phase,
    task,
    output_root,
    workers=12,
    device="cuda",
    *,
    diagnostic_atlas=False,
):
    if diagnostic_atlas and phase != "atlas":
        raise ValueError("--diagnostic-atlas requires phase=atlas")

    # For atlas, complete selection proof validation before loading any
    # calibration/evaluation event shards.  Evaluation admission is repeated
    # in load_events as a defense against callers bypassing this branch.
    diagnostic_binding = None
    if phase == "atlas":
        if diagnostic_atlas:
            diagnostic_binding = _diagnostic_binding(output_root)
        else:
            selected_budget(output_root)

    from .dispatch import run_spawned_branch

    root = Path(output_root)
    phase_dir = root / ("phase1" if phase == "grid" else "phase2")
    if phase == "grid":
        events = load_events(root, task, "calibration")[:20]
        budgets = BUDGETS
        prefixes = GRID_K
        # Execute every available independent request without fabricating
        # missing events.  Incomplete planned sampling cannot authorize
        # formal evaluation selection.
    elif phase == "atlas":
        b = (
            diagnostic_binding["selected_budget"]
            if diagnostic_binding is not None
            else selected_budget(root)
        )
        budgets = ((int(b["tail_horizon"]), int(b["action_budget"])),)
        prefixes = ATLAS_K
        events = load_events(
            root, task, "calibration", diagnostic_atlas=diagnostic_atlas
        ) + load_events(
            root, task, "evaluation", diagnostic_atlas=diagnostic_atlas
        )
    else:
        raise ValueError(phase)

    module_hashes = {
        name: file_sha(Path(__file__).parent / name)
        for name in (
            "runtime.py",
            "measurement.py",
            "dispatch.py",
            "common.py",
            "assets.py",
            "runtime_identity.py",
        )
    }
    requests = []
    for event, (tail, budget), seed in itertools.product(events, budgets, SEEDS):
        for op, k in itertools.product(OPERATORS, prefixes):
            requests.append(
                (
                    event,
                    dict(
                        recovery_actor_seed=seed,
                        operator=op,
                        prefix_k=k,
                        tail_horizon=tail,
                        action_budget=budget,
                        policy_call_cap=8,
                    ),
                    False,
                )
            )
        for op, k in REFERENCES:
            requests.append(
                (
                    event,
                    dict(
                        recovery_actor_seed=seed,
                        operator=op,
                        prefix_k=k,
                        tail_horizon=tail,
                        action_budget=budget,
                        policy_call_cap=8,
                    ),
                    True,
                )
            )

    import threading

    cancelled = threading.Event()

    def run(item):
        if cancelled.is_set():
            return
        index, (event, req, reference) = item
        guard_execution()
        key = digest(
            {
                "event": event["event_instance_id"],
                "event_hash": digest(event),
                **req,
                "reference": reference,
            }
        )
        path = phase_dir / "shards" / task / f"{key}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if not compatible_runtime_row(row, module_hashes) or row.get(
                "runtime_receipt_sha256"
            ) != os.environ.get("S1_RUNTIME_RECEIPT_SHA256"):
                raise RuntimeError(f"existing shard runtime binding changed: {path}")
            if diagnostic_binding is not None and (
                row.get("diagnostic_continuation") is not True
                or row.get("diagnostic_selection_receipt_sha256")
                != diagnostic_binding["receipt_sha256"]
            ):
                raise RuntimeError(f"existing diagnostic shard binding changed: {path}")
            if row.get("status") != "COMPLETE" and row.get(
                "error_type"
            ) != "PrefixOutsideTaskHorizon":
                raise RuntimeError(f"immutable failed shard {path}")
            return

        # Phase1 calibration branches are identical interventions at shared
        # k/budget.  Reuse preserves all original source and trace fields;
        # only the independent diagnostic binding is appended.
        prior = root / "phase1/shards" / task / f"{key}.json"
        if phase == "atlas" and event["split"] == "calibration" and prior.exists():
            row = json.loads(prior.read_text())
            if row.get("status") != "COMPLETE" and row.get(
                "error_type"
            ) != "PrefixOutsideTaskHorizon":
                raise RuntimeError("cannot reuse failed calibration")
            if not compatible_runtime_row(row, module_hashes) or row.get(
                "runtime_receipt_sha256"
            ) != os.environ.get("S1_RUNTIME_RECEIPT_SHA256"):
                raise RuntimeError("calibration runtime changed before atlas")
            row = {
                **row,
                "reused_calibration_shard": str(prior),
                "reused_calibration_sha256": file_sha(prior),
            }
        else:
            trace = phase_dir / "contact_topology" / task / f"{key}.jsonl.gz"
            row = run_spawned_branch(
                event=event,
                **req,
                device=(f"cuda:{index % 2}" if device == "cuda" else device),
                trace_path=str(trace),
                cancel_event=cancelled,
            )
            row["trace_path"] = str(trace)
            row["runtime_status"] = row.get("status")
            if row.get("status") == "OK" and not row.get("blocked") and not row.get(
                "error"
            ):
                row["status"] = "COMPLETE"
            if row.get("prefix_k") != req["prefix_k"] or row.get(
                "operator"
            ) != req["operator"]:
                row["status"] = "BLOCKED"
                row["request_mismatch"] = True
            row["returned_configuration"] = {key: row.get(key) for key in req}
            row.update(**req)
            row.update(
                source_commit=os.environ.get("S1_SOURCE_COMMIT"),
                runtime_module_hashes=module_hashes,
                runtime_receipt_sha256=os.environ.get("S1_RUNTIME_RECEIPT_SHA256"),
                pai_run_id=os.environ.get("PAI_CANARY_RUN_ID"),
                job_id=os.environ.get("S1_JOB_ID"),
                task=task,
                event_instance_id=event["event_instance_id"],
                init_state_id=event["init_state_id"],
                generator_actor_seed=event["actor_seed"],
                split=event["split"],
                is_reference=reference,
                configured_budget={
                    "tail_horizon": req["tail_horizon"],
                    "action_budget": req["action_budget"],
                    "policy_call_cap": 8,
                },
            )

        if diagnostic_binding is not None:
            _attach_diagnostic_binding(row, diagnostic_binding)
        if any(not row.get(k) for k in ("pid", "env_hash", "chunk_hash")):
            row["status"] = "BLOCKED"
            row["provenance_missing"] = True
        atomic_json(path, row)
        if row.get("error_type") == "PrefixOutsideTaskHorizon":
            print(
                f"preserved infeasible prefix; continuing independent requests: {path}",
                flush=True,
            )
            return
        if row.get("status") != "COMPLETE":
            raise RuntimeError(f"branch blocked: {path}")
        record_first_work(path)
        print(
            f"persisted {phase} task={task} index={index + 1}/{len(requests)}",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, item) for item in enumerate(requests)]
        try:
            for f in as_completed(futures):
                f.result()
        except BaseException:
            cancelled.set()
            for f in futures:
                f.cancel()
            marker = os.environ.get("S1_STOP_FILE")
            if marker:
                Path(marker).write_text("S1_MATRIX_FAILURE")
            raise

    paths = sorted((phase_dir / "shards" / task).glob("*.json"))
    blocked = sum(json.loads(p.read_text()).get("status") != "COMPLETE" for p in paths)
    result = {
        "task": task,
        "phase": phase,
        "events": len(events),
        "requested_rows": len(requests),
        "persisted_rows": len(paths),
        "core_rows": sum(
            not json.loads(p.read_text()).get("is_reference", False) for p in paths
        ),
        "blocked_contract_rows": blocked,
        "status": "COMPLETE_WITH_BLOCKED_CONTRACT_ROWS" if blocked else "COMPLETE",
    }
    result.update(
        planned_events=20 if phase == "grid" else len(events),
        planned_sample_complete=phase != "grid" or len(events) == 20,
        missing_planned_events=max(0, 20 - len(events)) if phase == "grid" else 0,
    )
    if diagnostic_binding is not None:
        result.update(
            diagnostic_continuation=True,
            diagnostic_selection_receipt_sha256=diagnostic_binding["receipt_sha256"],
            diagnostic_selection_receipt_path=diagnostic_binding["receipt_path"],
            diagnostic_selection_binding={
                "input_sha256": diagnostic_binding["input_sha256"],
                "protocol_sha256": diagnostic_binding["protocol_sha256"],
                "original_rank": diagnostic_binding["original_rank"],
            },
        )
    if not result["planned_sample_complete"]:
        result["status"] = "COMPLETE_AVAILABLE_REQUESTS_SAMPLE_SHORTFALL"
    if len(paths) != len(requests):
        raise RuntimeError("shard count mismatch")
    atomic_json(phase_dir / f"completion_{task}.json", result)
    return result
