"""Bounded CUDA 12.4 compatibility smoke against the current main collector.

This harness owns only the smoke script and its output directory. It imports the
main checkout explicitly, runs a fixed clean cream/init0/actor7 episode, then
runs at most one independent fresh_h4 branch from that newly collected event.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import subprocess
from pathlib import Path
from typing import Any

MAIN_ROOT = Path("/mnt/cpfs/zbl-cpfs-new/share/leon/codex-archives/Explore-claude-local-worktrees/R16-P14-stage2f-step8-20260907").resolve()
WORKER_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = WORKER_ROOT / "artifacts/stage2f/preflight/runtime_compat/gpu_smoke"
TASK = "put_the_cream_cheese_in_the_bowl"
INIT_STATE_ID = 0
ACTOR_SEED = 7
RECOVERY_ACTOR_SEED = 17
DEVICE = "cuda:0"
ACTION_BUDGET = 8
TAIL_HORIZON = 4
POLICY_CALL_CAP = 8
OUTER_BUDGET_S = 285.0
POOL_PATH = MAIN_ROOT / "artifacts/stage2f/phase0b/init_pool/put_the_cream_cheese_in_the_bowl.json"

# Make the current main source win over this worker checkout and site packages.
_IMPORT_PATHS = [
    MAIN_ROOT,
    MAIN_ROOT / "experiments",
    MAIN_ROOT / "experiments/r16_p14_stage2a",
    MAIN_ROOT / "experiments/r16_p14_stage2b",
    MAIN_ROOT / "experiments/r16_p14_stage2c",
    MAIN_ROOT / "experiments/r16_p14_stage2d",
]
for _path in reversed(_IMPORT_PATHS):
    _value = str(_path)
    while _value in sys.path:
        sys.path.remove(_value)
    sys.path.insert(0, _value)

from r16_p14_stage2f.s1 import assets as main_assets  # noqa: E402
from r16_p14_stage2f.s1.common import spawn_call  # noqa: E402
from r16_p14_stage2f.s1 import collector as main_collector  # noqa: E402
from r16_p14_stage2f.s1 import dispatch as main_dispatch  # noqa: E402

SOURCE_FILES = (
    "experiments/r16_p14_stage2f/s1/assets.py",
    "experiments/r16_p14_stage2f/s1/common.py",
    "experiments/r16_p14_stage2f/s1/collector.py",
    "experiments/r16_p14_stage2f/s1/dispatch.py",
    "experiments/r16_p14_stage2f/s1/runtime.py",
    "experiments/r16_p14_stage2f/s1/measurement.py",
    "experiments/r16_p14_stage2d/r16_p14_stage2d/runtime.py",
    "experiments/r16_p14_stage2d/r16_p14_stage2d/io_utils.py",
)
RUNTIME_RECEIPT = WORKER_ROOT / "artifacts/stage2f/preflight/runtime_compat/runtime_compat_receipt.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _source_manifest() -> dict[str, Any]:
    files = {}
    for rel in SOURCE_FILES:
        path = MAIN_ROOT / rel
        files[rel] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        }
    receipt = {
        "path": str(RUNTIME_RECEIPT),
        "exists": RUNTIME_RECEIPT.is_file(),
        "sha256": _sha256(RUNTIME_RECEIPT) if RUNTIME_RECEIPT.is_file() else None,
    }
    main_commit = os.environ.get("S1_MAIN_COMMIT")
    if not main_commit:
        try:
            main_commit = subprocess.check_output(
                ["git", "-C", str(MAIN_ROOT), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except BaseException:
            main_commit = None
    return {
        "main_root": str(MAIN_ROOT),
        "main_git_commit": main_commit,
        "files": files,
        "runtime_receipt": receipt,
    }


def _module_manifest() -> dict[str, Any]:
    result = {}
    for name in ("torch", "numpy", "robosuite", "mujoco", "wandb", "libero.libero"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        result[name] = {
            "version": getattr(module, "__version__", None),
            "file": str(path),
            "sha256": _sha256(path),
        }
    import torch
    result["torch"].update({
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    })
    return result


def _compat_metadata() -> dict[str, Any]:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    assets = main_assets.configure_assets()
    return {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "venv_python": str((Path(sys.prefix) / "bin/python").resolve()),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "device": DEVICE,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "assets": assets,
        "source": _source_manifest(),
        "modules": _module_manifest(),
    }


def _enrich(row: dict[str, Any], *, label: str, elapsed_s: float, metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["compat_smoke"] = {
        "label": label,
        "elapsed_seconds": float(elapsed_s),
        "device": DEVICE,
        "batch_size": 1,
        "source": metadata["source"],
        "modules": metadata["modules"],
        "runtime": {
            key: metadata[key]
            for key in ("python", "python_executable", "venv_python", "prefix", "base_prefix", "cuda_visible_devices")
        },
        "assets": metadata["assets"],
    }
    return result


def _branch_checks(row: dict[str, Any]) -> dict[str, Any]:
    reconstruction = row.get("reconstruction") or {}
    missing = []
    for key in ("pid", "env_hash", "chunk_hash", "trace_sha256"):
        if not row.get(key):
            missing.append(key)
    max_error = reconstruction.get("max_anchor_state_error")
    return {
        "status_ok": row.get("status") == "OK",
        "anchor_max_error_zero": max_error == 0 or max_error == 0.0,
        "required_pid_env_chunk_trace": not missing,
        "missing_fields": missing,
        "missing_fields_ok": not missing,
        "zero_injection": row.get("zero_injection") is True and row.get("injection_calls") == 0,
        "d1_physics_instrumented": row.get("physics_instrumented") is True,
        "d1_physics_count_positive": int(row.get("physics_step_count") or 0) > 0,
        "actual_recovery_budget": int(row.get("actual_new_recovery_actions") or 0) <= ACTION_BUDGET,
        "configured_budget_visible": row.get("configured_action_budget") == ACTION_BUDGET,
        "configured_tail_visible": row.get("configured_tail_horizon") == TAIL_HORIZON,
        "policy_cap": int(row.get("total_policy_calls") or 0) <= POLICY_CALL_CAP,
        "actual_global_step_within_horizon": (
            row.get("actual_global_step") is None
            or int(row["actual_global_step"]) <= int(row.get("task_horizon", 10**9))
        ),
    }


def _error_row(label: str, started: float, exc: BaseException) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "BLOCKED_BY_RUNTIME_ERROR",
        "label": label,
        "pid": os.getpid(),
        "device": DEVICE,
        "elapsed_seconds": float(time.monotonic() - started),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def run() -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + OUTER_BUDGET_S
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        metadata = _compat_metadata()
        _atomic_json(OUT_ROOT / "source_module_manifest.json", metadata)
    except BaseException as exc:
        # Persist startup failures before returning; a failed import or asset
        # binding must remain reviewable and must never look like no evidence.
        error = _error_row("compat_startup", started, exc)
        _atomic_json(OUT_ROOT / "startup_error.json", error)
        summary = {
            "schema_version": 1,
            "status": "BLOCKED_BY_RUNTIME_ERROR",
            "engineering_evidence_only": True,
            "formal_gate_claim_allowed": False,
            "device": DEVICE,
            "visible_gpu_expected": 1,
            "batch_size": 1,
            "configured_outer_budget_s": OUTER_BUDGET_S,
            "actual_wall_s": float(time.monotonic() - started),
            "actual_gpu_hours_1gpu": float(time.monotonic() - started) / 3600.0,
            "source": _source_manifest(),
            "error": error,
            "rows_persisted_before_summary": True,
        }
        _atomic_json(OUT_ROOT / "summary.json", summary)
        _atomic_json(OUT_ROOT / "LATEST.json", {"status": summary["status"], "summary": str(OUT_ROOT / "summary.json")})
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary
    rows: dict[str, Any] = {}
    collector_started = time.monotonic()
    try:
        if not POOL_PATH.is_file():
            raise FileNotFoundError(POOL_PATH)
        collector_result = spawn_call(
            "experiments.r16_p14_stage2f.s1.collector",
            "collect_episode",
            {
                "task": TASK,
                "init_state_id": INIT_STATE_ID,
                "actor_seed": ACTOR_SEED,
                "pool_path": str(POOL_PATH),
                "output_root": str(OUT_ROOT / "collector"),
                "device": DEVICE,
            },
            timeout=max(1.0, min(220.0, deadline - time.monotonic())),
        )
        collector_result = _enrich(
            collector_result,
            label="collector_clean_cream_init000_actor7",
            elapsed_s=time.monotonic() - collector_started,
            metadata=metadata,
        )
    except BaseException as exc:
        collector_result = _error_row("collector_clean_cream_init000_actor7", collector_started, exc)
    collector_path = OUT_ROOT / "collector_row.json"
    _atomic_json(collector_path, collector_result)
    rows["collector"] = collector_result

    event = collector_result.get("event") if isinstance(collector_result, dict) else None
    branch_result = None
    branch_checks = {
        "qualified_event_present": bool(collector_result.get("qualified_natural_failure") and event),
        "branch_started": False,
    }
    if branch_checks["qualified_event_present"] and time.monotonic() < deadline:
        branch_started = time.monotonic()
        try:
            branch_result = main_dispatch.run_spawned_branch(
                event,
                RECOVERY_ACTOR_SEED,
                "fresh_h4",
                prefix_k=2,
                tail_horizon=TAIL_HORIZON,
                action_budget=ACTION_BUDGET,
                policy_call_cap=POLICY_CALL_CAP,
                device=DEVICE,
                trace_path=str(OUT_ROOT / "branch" / "fresh_h4__seed17.jsonl.gz"),
                timeout_s=max(1.0, min(110.0, deadline - time.monotonic())),
                repeat=0,
            )
            branch_result = _enrich(
                branch_result,
                label="branch_fresh_h4_k2_seed17",
                elapsed_s=time.monotonic() - branch_started,
                metadata=metadata,
            )
            # The runtime row carries the content digest; retain the exact
            # requested worker-owned path alongside it for auditability.
            branch_result.setdefault(
                "trace_path",
                str(OUT_ROOT / "branch" / "fresh_h4__seed17.jsonl.gz"),
            )
        except BaseException as exc:
            branch_result = _error_row("branch_fresh_h4_k2_seed17", branch_started, exc)
        _atomic_json(OUT_ROOT / "branch_row.json", branch_result)
        rows["branch"] = branch_result
        branch_checks["branch_started"] = True
        branch_checks["branch_checks"] = _branch_checks(branch_result)
    elif not branch_checks["qualified_event_present"]:
        branch_result = {
            "status": "NO_QUALIFIED_EVENT_IN_INIT0",
            "label": "branch_fresh_h4_k2_seed17",
            "device": DEVICE,
        }
        _atomic_json(OUT_ROOT / "branch_row.json", branch_result)
        rows["branch"] = branch_result
    else:
        branch_result = {
            "status": "BLOCKED_BY_OUTER_DEADLINE",
            "label": "branch_fresh_h4_k2_seed17",
            "device": DEVICE,
        }
        _atomic_json(OUT_ROOT / "branch_row.json", branch_result)
        rows["branch"] = branch_result

    collector_pass = collector_result.get("status") == "COMPLETE"
    branch_values = branch_checks.get("branch_checks") or {}
    branch_pass = bool(branch_values) and all(
        value for key, value in branch_values.items() if key != "missing_fields"
    )
    summary = {
        "schema_version": 1,
        "status": "PASS" if collector_pass and branch_pass else "BLOCKED_ENGINEERING_ACCEPTANCE",
        "engineering_evidence_only": True,
        "formal_gate_claim_allowed": False,
        "device": DEVICE,
        "visible_gpu_expected": 1,
        "batch_size": 1,
        "configured_outer_budget_s": OUTER_BUDGET_S,
        "actual_wall_s": float(time.monotonic() - started),
        "actual_gpu_hours_1gpu": float(time.monotonic() - started) / 3600.0,
        "source": metadata["source"],
        "modules": metadata["modules"],
        "collector_row": str(collector_path),
        "branch_row": str(OUT_ROOT / "branch_row.json"),
        "collector_trace_path": collector_result.get("trace_path"),
        "branch_trace_path": None if branch_result is None else branch_result.get("trace_path"),
        "collector_status": collector_result.get("status"),
        "collector_qualified_natural_failure": collector_result.get("qualified_natural_failure"),
        "branch_checks": branch_checks,
        "rows_persisted_before_summary": True,
    }
    _atomic_json(OUT_ROOT / "summary.json", summary)
    _atomic_json(OUT_ROOT / "LATEST.json", {"status": summary["status"], "summary": str(OUT_ROOT / "summary.json")})
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    raise SystemExit(0 if run()["status"] == "PASS" else 2)
