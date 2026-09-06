"""Bounded dev14 live acceptance smoke for the Stage-2F S1 backend.

This consumes only the repaired infrastructure event staged under
``artifacts/stage2f/preflight/live_backend_s1/input``.  It never reads
calibration/evaluation event pools and writes every branch row immediately
before any summary/validation is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "experiments"
_IMPORT_ROOTS = (
    ROOT,
    EXPERIMENTS,
    EXPERIMENTS / "r16_p14_stage2a",
    EXPERIMENTS / "r16_p14_stage2b",
    EXPERIMENTS / "r16_p14_stage2c",
    EXPERIMENTS / "r16_p14_stage2d",
)
# The tracked repository LIBERO package must win over site-packages.  Rebuild
# this prefix deterministically in spawned children before any env import.
for _path in reversed(_IMPORT_ROOTS):
    _value = str(_path)
    while _value in sys.path:
        sys.path.remove(_value)
    sys.path.insert(0, _value)

from r16_p14_stage2a.envs import state_sha256  # noqa: E402
from r16_p14_stage2a.settings import TASK_SPECS  # noqa: E402
from r16_p14_stage2b.io_utils import sha256_array  # noqa: E402
from r16_p14_stage2b.runtime import ActorBundle, ActorHistory  # noqa: E402
from r16_p14_stage2d.io_utils import sha256_array as d_sha256_array  # noqa: E402
from r16_p14_stage2d.runtime import reconstruct_anchor  # noqa: E402
from r16_p14_stage2f.s1.dispatch import run_spawned_branch  # noqa: E402
from r16_p14_stage2f.s1.measurement import SimulationStepRecorder, trace_content_sha256  # noqa: E402
from r16_p14_stage2f.s1.runtime import OPERATORS, REFERENCE_OPERATORS  # noqa: E402

TASK = "put_the_cream_cheese_in_the_bowl"
DEFAULT_EVENT = ROOT / "artifacts/stage2f/preflight/live_backend_s1/input/event_repaired_row.json"
DEFAULT_OUT = ROOT / "artifacts/stage2f/preflight/live_backend_s1"
DEFAULT_ASSETS = Path("/mnt/cpfs/zbl-cpfs-new/dataset/leon/libero/assets/90001343cb134b7e26e18fde0fa2416f3ed6e6a3")
ACTION_BUDGET = 8
TAIL_HORIZON = 4
POLICY_CALL_CAP = 8
RECOVERY_ACTOR_SEED = 17
REPLAY_ACTIONS = 16
OUTER_BUDGET_S = 780.0
EXPECTED_LIBERO_SOURCE = ROOT / "libero/libero/__init__.py"
EXPECTED_LIBERO_SOURCE_SHA256 = "999901d941ac107d29c11fd0e0f4dbf2587470794b779b5cb8353681b4267e81"


def _configure_live_libero_assets() -> dict[str, str]:
    # Bind exact frozen assets/config and reject pip LIBERO imports.
    config = ROOT / "experiments/r16_p14_libero_stage1/libero_config"
    scene = DEFAULT_ASSETS / "scenes/libero_tabletop_base_style.xml"
    config_file = config / "config.yaml"
    if not scene.is_file() or not config_file.is_file():
        raise RuntimeError("exact frozen LIBERO assets/config unavailable")
    os.environ["LIBERO_CONFIG_PATH"] = str(config)
    import libero.libero as package

    source = Path(package.__file__).resolve()
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if source != EXPECTED_LIBERO_SOURCE.resolve() or source_sha256 != EXPECTED_LIBERO_SOURCE_SHA256:
        raise RuntimeError(
            f"wrong LIBERO package; expected {EXPECTED_LIBERO_SOURCE.resolve()} "
            f"sha={EXPECTED_LIBERO_SOURCE_SHA256}, got {source} sha={source_sha256}"
        )
    package.libero_config_path = str(config)
    package.config_file = str(config_file)
    package._assets_path_cache = str(DEFAULT_ASSETS)
    if Path(package.get_assets_path()).resolve() != DEFAULT_ASSETS.resolve():
        raise RuntimeError("LIBERO asset selection drift")
    return {
        "assets": str(DEFAULT_ASSETS),
        "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "package_source": str(source),
        "package_sha256": source_sha256,
    }


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _replay_child(event: dict[str, Any], with_recorder: bool, trace_path: str | None, result_queue: Any, device: str) -> None:
    env = None
    recorder = None
    libero_identity = None
    try:
        libero_identity = _configure_live_libero_assets()
        bundle = ActorBundle.load(int(event["actor_seed"]), device)
        env, history, reconstruction = reconstruct_anchor(event, bundle)
        if not all(bool(value) for key, value in reconstruction.items() if key != "max_anchor_state_error"):
            raise RuntimeError(f"reconstruction checks failed: {reconstruction}")
        if float(reconstruction.get("max_anchor_state_error", float("inf"))) > 1e-9:
            raise RuntimeError(f"anchor max error={reconstruction.get('max_anchor_state_error')}")
        if with_recorder:
            initial_z = float(np.asarray(event["initial_manipulated_qpos"], dtype=np.float64)[2])
            recorder = SimulationStepRecorder(
                env,
                event["task"],
                event_instance_id=event["event_instance_id"],
                prefix_k=0,
                operator="replay_original16",
                actor_seed=int(event["actor_seed"]),
                initial_object_z=initial_z,
                trace_path=trace_path,
                baseline_contacts=event.get("anchor_contacts"),
            )
        previous_gripper = float(history.actions[-1][-1])
        old = np.asarray(event["original_chunk"], dtype=np.float32).reshape(-1, 7)
        for step, action in enumerate(old[:REPLAY_ACTIONS]):
            if recorder is not None:
                recorder.set_action(step, action, previous_gripper, bool(event["task_phase"].get("stable_lift_two_steps")))
            observation, _, _, _ = env.step(action)
            history.update(observation, action)
            if recorder is not None:
                recorder.finish_action()
            previous_gripper = float(action[-1])
        if recorder is not None:
            recorder.close()
        result_queue.put({
            "status": "OK",
            "with_recorder": bool(with_recorder),
            "reconstruction": _json_safe(reconstruction),
            "terminal_state_hash": state_sha256(np.asarray(env.get_sim_state(), dtype=np.float64)),
            "history_state_hash": sha256_array(history.state_array(), np.float32),
            "history_action_hash": sha256_array(history.action_array(), np.float32),
            "history_state_shape": list(history.state_array().shape),
            "history_action_shape": list(history.action_array().shape),
            "physics_instrumented": None if recorder is None else bool(recorder.physics_instrumented),
            "physics_step_count": None if recorder is None else int(recorder.physics_step_count),
            "trace_records": None if recorder is None else int(len(recorder.records)),
            "trace_content_sha256": None if recorder is None else trace_content_sha256(recorder.records),
            "trace_sha256": None if recorder is None else recorder.trace_sha256,
            "libero_identity": libero_identity,
            "pid": os.getpid(),
        })
    except BaseException as exc:
        result_queue.put({
            "status": "ERROR",
            "with_recorder": bool(with_recorder),
            "libero_identity": libero_identity,
            "pid": os.getpid(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except BaseException:
                pass
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass


def _run_replay(event: dict[str, Any], with_recorder: bool, trace_path: Path | None, device: str, timeout_s: float) -> dict[str, Any]:
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_replay_child,
        args=(event, bool(with_recorder), None if trace_path is None else str(trace_path), result_queue, device),
        name=f"r16p14-s1-replay-{int(with_recorder)}",
    )
    start = time.perf_counter()
    process.start()
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    while process.is_alive() and time.monotonic() < deadline:
        process.join(min(1.0, max(0.0, deadline - time.monotonic())))
    if process.is_alive():
        process.terminate()
        process.join(30.0)
        row = {"status": "BLOCKED_BY_TIMEOUT", "error_type": "TimeoutError", "error": f"replay timeout {timeout_s}s"}
    else:
        try:
            row = result_queue.get(timeout=5.0)
        except queue.Empty:
            row = {"status": "BLOCKED_BY_CHILD_EXIT", "error_type": "ChildProcessError", "error": f"exitcode={process.exitcode}"}
    row["wall_s"] = float(time.perf_counter() - start)
    row["child_exitcode"] = process.exitcode
    result_queue.close()
    result_queue.join_thread()
    return row


def _load_repaired_row(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    row = json.loads(path.read_text())
    event = dict(row["event"])
    if event.get("event_instance_id") != event.get("event_id"):
        raise RuntimeError("event_instance_id != event_id")
    expected = d_sha256_array(event["init_state"], np.float64)
    if event.get("init_state_hash") != expected:
        raise RuntimeError(f"repaired init_state_hash mismatch: {event.get('init_state_hash')} != {expected}")
    return row, event


def _append_raw(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _validate_branch_row(row: dict[str, Any], event: dict[str, Any], *, expected_prefix: int | None) -> list[str]:
    missing = []
    for key in ("pid", "env_hash", "chunk_hash", "process_start_method"):
        value = row.get(key)
        if key == "process_start_method":
            if value != "spawn":
                missing.append(key)
        elif value in (None, ""):
            missing.append(key)
    if expected_prefix is not None and row.get("prefix_k") != expected_prefix:
        missing.append("prefix_k")
    if not bool(row.get("policy_call_cap_respected")):
        missing.append("policy_call_cap")
    if not bool(row.get("action_budget_respected")):
        missing.append("action_budget")
    if row.get("actual_global_step") is not None and int(row["actual_global_step"]) > int(TASK_SPECS[event["task"]].horizon):
        missing.append("task_horizon")
    return missing


def _branch_request_label(operator: str, repeat: int = 0) -> str:
    return f"{operator}__seed{RECOVERY_ACTOR_SEED}__repeat{repeat}"


def run(event_path: Path, output_root: Path, device: str = "cuda:0", budget_s: float = OUTER_BUDGET_S) -> dict[str, Any]:
    run_id = time.strftime("attempt_%Y%m%dT%H%M%S", time.gmtime()) + f"_{os.getpid()}"
    run_root = output_root / run_id
    traces = run_root / "traces"
    run_root.mkdir(parents=True, exist_ok=True)
    row, event = _load_repaired_row(event_path)
    start = time.perf_counter()
    deadline = start + float(budget_s)
    raw_rows_path = run_root / "rows_raw.jsonl"
    results: list[dict[str, Any]] = []
    branch_specs = [(op, 0) for op in OPERATORS] + [(op, 0) for op in REFERENCE_OPERATORS]
    branch_specs.insert(1, ("fresh_h4", 1))
    with raw_rows_path.open("a", encoding="utf-8") as raw_handle:
        replay_budget = min(180.0, max(1.0, deadline - time.monotonic()))
        no_recorder = _run_replay(event, False, None, device, replay_budget)
        no_recorder["label"] = "replay_no_recorder"
        _append_raw(raw_handle, no_recorder)
        results.append(no_recorder)
        replay_budget = min(180.0, max(1.0, deadline - time.monotonic()))
        recorder_trace = traces / "replay_original16.jsonl.gz"
        with_recorder = _run_replay(event, True, recorder_trace, device, replay_budget)
        with_recorder["label"] = "replay_with_recorder"
        _append_raw(raw_handle, with_recorder)
        results.append(with_recorder)
        for operator, repeat in branch_specs:
            if time.monotonic() >= deadline:
                row_out = {"status": "BLOCKED_BY_OUTER_DEADLINE", "operator": operator, "repeat": repeat, "pid": os.getpid()}
            else:
                branch_start = time.perf_counter()
                timeout = min(150.0, max(1.0, deadline - time.monotonic()))
                row_out = run_spawned_branch(
                    event,
                    RECOVERY_ACTOR_SEED,
                    operator,
                    prefix_k=2,
                    tail_horizon=TAIL_HORIZON,
                    action_budget=ACTION_BUDGET,
                    policy_call_cap=POLICY_CALL_CAP,
                    device=device,
                    trace_path=str(traces / f"{_branch_request_label(operator, repeat)}.jsonl.gz"),
                    timeout_s=timeout,
                    repeat=repeat,
                )
                row_out["dispatcher_wall_s"] = float(time.perf_counter() - branch_start)
            row_out["label"] = _branch_request_label(operator, repeat)
            row_out["engineering_missing"] = _validate_branch_row(row_out, event, expected_prefix=None if operator in REFERENCE_OPERATORS else 2)
            _append_raw(raw_handle, row_out)
            results.append(row_out)
    replay_equal = False
    if no_recorder.get("status") == "OK" and with_recorder.get("status") == "OK":
        replay_equal = all(no_recorder.get(key) == with_recorder.get(key) for key in ("terminal_state_hash", "history_state_hash", "history_action_hash"))
    branch_rows = [item for item in results if item.get("label", "").startswith(tuple(OPERATORS + REFERENCE_OPERATORS))]
    core_rows = [item for item in branch_rows if item.get("operator") in OPERATORS]
    reference_rows = [item for item in branch_rows if item.get("operator") in REFERENCE_OPERATORS]
    repeat_a = next((item for item in core_rows if item.get("operator") == "fresh_h4" and item.get("repeat") == 0), None)
    repeat_b = next((item for item in core_rows if item.get("operator") == "fresh_h4" and item.get("repeat") == 1), None)
    repeat_equal = bool(repeat_a and repeat_b and repeat_a.get("d4_signatures", {}).get("pre_tail") == repeat_b.get("d4_signatures", {}).get("pre_tail") and repeat_a.get("safe_success") == repeat_b.get("safe_success"))
    pids = [item.get("pid") for item in branch_rows if item.get("pid")]
    checks = {
        "event_is_repaired_copy": True,
        "anchor_reconstruction_bit_exact": all(item.get("status") == "OK" and all(bool(v) for k, v in item.get("reconstruction", {}).items() if k != "max_anchor_state_error") and float(item.get("reconstruction", {}).get("max_anchor_state_error", 1.0)) == 0.0 for item in branch_rows if item.get("status") == "OK"),
        "replay_terminal_and_history_hashes_equal": replay_equal,
        "recorder_d1_25_physics_per_control": with_recorder.get("status") == "OK" and with_recorder.get("physics_step_count") == REPLAY_ACTIONS * 25 and with_recorder.get("trace_records") == REPLAY_ACTIONS * 26,
        "four_core_operators_present": {item.get("operator") for item in core_rows} == set(OPERATORS),
        "four_reference_operators_present": {item.get("operator") for item in reference_rows} == set(REFERENCE_OPERATORS),
        "fresh_h4_repeat_pre_tail_and_safe_success_equal": repeat_equal,
        "branch_pids_unique": len(pids) == len(set(pids)) and len(pids) == len(branch_rows),
        "branch_required_pid_env_chunk": all(not item.get("engineering_missing") for item in branch_rows),
        "all_branch_budgets_respected": all(not item.get("engineering_missing") for item in branch_rows),
    }
    status = "PASS" if all(checks.values()) else "BLOCKED_ENGINEERING_ACCEPTANCE"
    wall_s = float(time.perf_counter() - start)
    summary = {
        "schema_version": 1,
        "status": status,
        "engineering_evidence_only": True,
        "formal_gate_claim_allowed": False,
        "run_id": run_id,
        "event_instance_id": event["event_instance_id"],
        "source_event_path": str(event_path),
        "device": device,
        "visible_gpu_count_expected": 1,
        "batch_size": 1,
        "libero_identity": with_recorder.get("libero_identity") or no_recorder.get("libero_identity"),
        "libero_source_expected": str(EXPECTED_LIBERO_SOURCE.resolve()),
        "libero_source_sha256_expected": EXPECTED_LIBERO_SOURCE_SHA256,
        "configured_outer_budget_s": float(budget_s),
        "actual_wall_s": wall_s,
        "actual_gpu_hours_1gpu": wall_s / 3600.0,
        "checks": checks,
        "rows_file": str(raw_rows_path),
        "row_count": len(results),
        "branch_labels": [item.get("label") for item in results],
        "failed_rows_preserved_before_summary": True,
    }
    _write_json_atomic(run_root / "summary.json", summary)
    _write_json_atomic(output_root / "LATEST.json", {"run_id": run_id, "summary": str(run_root / "summary.json"), "status": status})
    print(json.dumps(summary, sort_keys=True, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=Path, default=DEFAULT_EVENT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget-s", type=float, default=OUTER_BUDGET_S)
    args = parser.parse_args()
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("LIBERO_ASSETS_PATH", str(DEFAULT_ASSETS))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    result = run(args.event, args.output_root, args.device, args.budget_s)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
