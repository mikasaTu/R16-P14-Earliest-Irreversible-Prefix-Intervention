"""Spawn-isolated Stage-2F branch dispatcher."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import random
import time
import traceback
from queue import Empty
from typing import Any

import numpy as np

from .runtime import OPERATORS, REFERENCE_OPERATORS


def _branch_seed(event: dict[str, Any], recovery_actor_seed: int, repeat: int = 0) -> int:
    payload = f"{event.get('event_instance_id', event.get('event_id', ''))}|{int(recovery_actor_seed)}|{int(repeat)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def _seed_process(seed: int) -> None:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except (ImportError, RuntimeError):
        pass


def _blocked_row(
    event: dict[str, Any],
    recovery_actor_seed: int,
    operator: str,
    error: BaseException | str,
    *,
    repeat: int,
    trace_path: str | None,
) -> dict[str, Any]:
    event_id = event.get("event_id")
    event_instance_id = event.get("event_instance_id", event_id)
    message = str(error)
    return {
        "schema_version": 2,
        "event_id": event_id,
        "event_instance_id": event_instance_id,
        "task": event.get("task"),
        "split": event.get("split"),
        "init_state_id": event.get("init_state_id"),
        "operator": operator,
        "recovery_actor_seed": int(recovery_actor_seed),
        "repeat": int(repeat),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "process_start_method": "spawn",
        "fresh_environment_created": False,
        "env_hash": None,
        "chunk_hash": event.get("original_chunk_hash"),
        "generator_chunk_hash": event.get("original_chunk_hash"),
        "recovery_chunk_hashes": [],
        "trace_path": trace_path,
        "trace_sha256": None,
        "zero_injection": True,
        "injection_calls": 0,
        "safe_success": False,
        "safe_success_unblocked_observation": False,
        "task_success": False,
        "diagnostic_only": True,
        "formal_positive_evidence_allowed": False,
        "new_idea_generated": False,
        "status": "BLOCKED_BY_RUNTIME_ERROR",
        "blocked": True,
        "missing_provenance": [
            key for key in ("env_hash", "runtime_result", "chunk_hash")
            if key == "runtime_result" or (key == "env_hash") or not event.get("original_chunk_hash")
        ],
        "error": message,
        "error_type": type(error).__name__ if isinstance(error, BaseException) else "RuntimeError",
        "traceback": traceback.format_exc() if isinstance(error, BaseException) else None,
        "dispatch_repeat": int(repeat),
        "dispatch_seed": None,
    }


def _add_request_metadata(row: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Keep bounded configuration visible even when a child fails closed."""
    row.update({
        "prefix_k": int(request.get("prefix_k", 0)),
        "configured_tail_horizon": int(request.get("tail_horizon", 0)),
        "configured_action_budget": int(request.get("action_budget", 0)),
        "configured_policy_call_cap": int(request.get("policy_call_cap", 8)),
        "validation_policy_calls": None,
        "recovery_policy_calls": None,
        "total_policy_calls": None,
        "actual_new_recovery_actions": None,
    })
    return row


def _child_entry(request: dict[str, Any], result_queue: Any) -> None:
    event = request["event"]
    recovery_actor_seed = int(request["recovery_actor_seed"])
    operator = str(request["operator"])
    repeat = int(request.get("repeat", 0))
    dispatch_seed = _branch_seed(event, recovery_actor_seed, repeat)
    _seed_process(dispatch_seed)
    try:
        from .runtime import execute_branch, execute_reference_branch

        kwargs = {
            "tail_horizon": int(request["tail_horizon"]),
            "action_budget": int(request["action_budget"]),
            "policy_call_cap": int(request.get("policy_call_cap", 8)),
            "device": request.get("device", "cpu"),
            "trace_path": request.get("trace_path"),
        }
        if operator in REFERENCE_OPERATORS:
            row = execute_reference_branch(
                event, recovery_actor_seed, operator, **kwargs,
            )
        elif operator in OPERATORS:
            row = execute_branch(
                event, recovery_actor_seed, operator, int(request["prefix_k"]), **kwargs,
            )
        else:
            raise ValueError(f"unsupported operator: {operator}")
        row["repeat"] = int(repeat)
        row["dispatch_seed"] = int(dispatch_seed)
        row["dispatch_repeat"] = int(repeat)
        row["dispatch_process_start_method"] = "spawn"
        result_queue.put(row)
    except BaseException as exc:
        row = _blocked_row(
            event, recovery_actor_seed, operator, exc,
            repeat=repeat, trace_path=request.get("trace_path"),
        )
        _add_request_metadata(row, request)
        row["dispatch_seed"] = int(dispatch_seed)
        result_queue.put(row)


def run_spawned_branch(
    event: dict[str, Any],
    recovery_actor_seed: int,
    operator: str,
    prefix_k: int,
    tail_horizon: int,
    action_budget: int,
    policy_call_cap: int = 8,
    device: str = "cpu",
    trace_path: str | None = None,
    timeout_s: float = 600.0,
    repeat: int = 0,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Run exactly one branch in a fresh ``spawn`` process and return its row."""
    if operator not in OPERATORS and operator not in REFERENCE_OPERATORS:
        raise ValueError(f"unsupported operator: {operator}")
    if float(timeout_s) <= 0:
        raise ValueError("timeout_s must be positive")
    request = {
        "event": dict(event),
        "recovery_actor_seed": int(recovery_actor_seed),
        "operator": str(operator),
        "prefix_k": int(prefix_k),
        "tail_horizon": int(tail_horizon),
        "action_budget": int(action_budget),
        "policy_call_cap": int(policy_call_cap),
        "device": str(device),
        "trace_path": None if trace_path is None else str(trace_path),
        "repeat": int(repeat),
    }
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_child_entry, args=(request, result_queue), name=f"r16p14-s1-{operator}")
    process.daemon = False
    process.start()
    deadline = time.monotonic() + float(timeout_s)
    cancel_reason = None
    received_row = None
    while process.is_alive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            cancel_reason = f"branch exceeded timeout_s={float(timeout_s)}"
            break
        if cancel_event is not None:
            try:
                if bool(cancel_event.is_set()):
                    cancel_reason = "cancel_event_set"
                    break
            except (AttributeError, RuntimeError):
                cancel_reason = "cancel_event_unreadable"
                break
        if any(os.environ.get(name) for name in ("R16_P14_STOP", "R16_P14_BLACKOUT", "R16_P14_STAGE2F_STOP")):
            cancel_reason = "stop_or_blackout_requested"
            break
        # A child Queue feeder may be waiting for the parent to drain a
        # result larger than the OS pipe.  Joining first deadlocks that child.
        if received_row is None:
            try:
                received_row = result_queue.get(timeout=min(0.25, remaining))
            except Empty:
                pass
        process.join(min(0.25, max(0.0, deadline - time.monotonic())))
    if process.is_alive():
        process.terminate()
        process.join(30.0)
        status = "BLOCKED_BY_TIMEOUT" if cancel_reason and cancel_reason.startswith("branch exceeded") else "BLOCKED_BY_CANCELLATION"
        row = _blocked_row(
            event, int(recovery_actor_seed), operator, cancel_reason or "branch cancelled",
            repeat=int(repeat), trace_path=request["trace_path"],
        )
        _add_request_metadata(row, request)
        row.update({
            "status": status,
            "error_type": "TimeoutError" if status == "BLOCKED_BY_TIMEOUT" else "CancelledError",
            "dispatch_seed": _branch_seed(event, int(recovery_actor_seed), int(repeat)),
            "cancel_reason": cancel_reason,
        })
        result_queue.close()
        result_queue.join_thread()
        return row
    try:
        row = received_row if received_row is not None else result_queue.get(timeout=5.0)
    except Empty:
        row = _blocked_row(
            event, int(recovery_actor_seed), operator,
            f"spawn child exited without result (exitcode={process.exitcode})",
            repeat=int(repeat), trace_path=request["trace_path"],
        )
        _add_request_metadata(row, request)
        row["status"] = "BLOCKED_BY_CHILD_EXIT"
        row["error_type"] = "ChildProcessError"
        row["dispatch_seed"] = _branch_seed(event, int(recovery_actor_seed), int(repeat))
    finally:
        result_queue.close()
        result_queue.join_thread()
    row.setdefault("repeat", int(repeat))
    row.setdefault("pid", int(process.pid))
    row.setdefault("parent_pid", os.getpid())
    row["dispatcher_pid"] = os.getpid()
    row["dispatcher_child_exitcode"] = process.exitcode
    row["dispatcher_process_start_method"] = "spawn"
    return row


def dispatch_branch(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return run_spawned_branch(*args, **kwargs)


def execute_spawned_branch(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return run_spawned_branch(*args, **kwargs)


__all__ = ["run_spawned_branch", "dispatch_branch", "execute_spawned_branch"]
