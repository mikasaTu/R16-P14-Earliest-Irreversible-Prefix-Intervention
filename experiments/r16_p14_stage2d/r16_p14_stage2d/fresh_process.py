from __future__ import annotations

import multiprocessing as mp
import os
import queue
import traceback
from typing import Any

import torch

from .runtime import execute_branch


def _child(request: dict[str, Any], output) -> None:
    torch.set_num_threads(1)
    try:
        result = execute_branch(**request)
    except BaseException as exc:
        result = {
            "event_id": request["event"].get("event_id"),
            "task": request["event"].get("task"),
            "parameter": request.get("parameter"),
            "prefix_k": request.get("prefix_k"),
            "requested_arm": request.get("arm"),
            "repeat": request.get("repeat"),
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "process_start_method": "spawn",
            "fresh_environment_created": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    output.put(result)


def run_spawned_branch(
    *,
    event: dict[str, Any],
    parameter: dict[str, Any],
    prefix_k: int,
    arm: str,
    repeat: int,
    device: str = "cpu",
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Run exactly one branch in a unique spawn-created process."""

    context = mp.get_context("spawn")
    output = context.Queue(maxsize=1)
    request = {
        "event": event,
        "parameter": parameter,
        "prefix_k": int(prefix_k),
        "arm": arm,
        "repeat": int(repeat),
        "device": device,
    }
    process = context.Process(target=_child, args=(request, output), daemon=False)
    process.start()
    try:
        result = output.get(timeout=timeout_s)
    except queue.Empty:
        process.terminate()
        process.join(timeout=30)
        return {
            "event_id": event.get("event_id"),
            "task": event.get("task"),
            "parameter": parameter,
            "prefix_k": prefix_k,
            "requested_arm": arm,
            "repeat": repeat,
            "pid": process.pid,
            "process_start_method": "spawn",
            "error": f"fresh branch timed out after {timeout_s}s",
        }
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=30)
        result["error"] = result.get("error") or "child did not exit after result"
    result["child_exitcode"] = process.exitcode
    result["unique_process_contract"] = bool(
        result.get("pid") == process.pid and result.get("process_start_method") == "spawn"
    )
    return result
