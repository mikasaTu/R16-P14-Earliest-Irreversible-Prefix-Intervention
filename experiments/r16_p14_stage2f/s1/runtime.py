"""Bounded Stage-2F zero-injection recovery runtime.

This module does not import or call any perturbation routine. A branch
reconstructs the frozen four-state/three-action anchor with the generator
actor, replays the original clean chunk prefix, then loads a separate
recovery actor and executes bounded new actions.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from r16_p14_stage2a.envs import state_sha256
from r16_p14_stage2a.settings import ACTION_DIM, TASK_SPECS
from r16_p14_stage2b.io_utils import sha256_array as raw_array_hash
try:
    from r16_p14_stage2c.runtime import hold_action as _hold_action
    from r16_p14_stage2c.runtime import rollback_action as _rollback_action
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    def _hold_action(previous: Any) -> np.ndarray:
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[-1] = float(np.asarray(previous, dtype=np.float32).reshape(-1)[-1])
        return action
    def _rollback_action(previous: Any) -> np.ndarray:
        action = np.asarray(previous, dtype=np.float32).reshape(-1).copy()
        action[:6] *= -1.0
        return action

# Keep measurement and static contract tests usable on CPU-only hosts.  The
# real branch imports the frozen actor/D runtime in a worker with torch; an
# unavailable actor stack fails closed when execution is requested.
try:
    from r16_p14_stage2b.runtime import ActorBundle, ActorHistory, chunk_hash
    from r16_p14_stage2d.runtime import branch_signature as _frozen_branch_signature
    from r16_p14_stage2d.runtime import reconstruct_anchor as _frozen_reconstruct_anchor
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    class ActorBundle:  # type: ignore[no-redef]
        @classmethod
        def load(cls, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("torch actor runtime is unavailable")
    ActorHistory = Any  # type: ignore[assignment,misc]
    def chunk_hash(chunk: Any) -> str:  # type: ignore[no-redef]
        return hashlib.sha256(np.ascontiguousarray(chunk, dtype=np.float32).tobytes()).hexdigest()
    _frozen_branch_signature = None  # type: ignore[assignment]
    _frozen_reconstruct_anchor = None  # type: ignore[assignment]

from .measurement import SimulationStepRecorder, label_trace, trace_content_sha256

OPERATORS = ("fresh_h4", "fresh_h16", "hold_1+fresh_h4", "rollback_1+fresh_h16")
REFERENCE_OPERATORS = ("immediate_fresh", "fixed_delay_2", "fixed_delay_4", "fixed_delay_8")
_OPERATOR_HORIZON = {"fresh_h4": 4, "fresh_h16": 16, "hold_1+fresh_h4": 4, "rollback_1+fresh_h16": 16}
_OPERATOR_PRELUDE = {"fresh_h4": None, "fresh_h16": None, "hold_1+fresh_h4": "hold", "rollback_1+fresh_h16": "rollback"}


def execution_contract(
    task: str,
    operator: str,
    prefix_k: int,
    tail_horizon: int,
    action_budget: int,
    policy_call_cap: int,
    anchor_global_step: int = 0,
) -> dict[str, int | bool]:
    """Resolve configured and effective bounds before opening a branch env."""
    if task not in TASK_SPECS:
        raise ValueError(f"unknown task: {task}")
    if operator not in OPERATORS:
        raise ValueError(f"unsupported operator: {operator}")
    prefix_k = int(prefix_k)
    tail_horizon = int(tail_horizon)
    action_budget = int(action_budget)
    policy_call_cap = int(policy_call_cap)
    anchor_global_step = int(anchor_global_step)
    if not 2 <= prefix_k <= 16:
        raise ValueError("prefix_k must be in [2,16]")
    if tail_horizon <= 0 or action_budget < 0:
        raise ValueError("tail_horizon must be positive and action_budget nonnegative")
    if not 0 <= policy_call_cap <= 8:
        raise ValueError("policy_call_cap must be in [0,8]")
    if anchor_global_step < 0:
        raise ValueError("anchor_global_step must be nonnegative")
    configured_operator_horizon = int(_OPERATOR_HORIZON[operator])
    effective_execution_horizon = min(configured_operator_horizon, tail_horizon)
    task_horizon = int(TASK_SPECS[task].horizon)
    prefix_end_global_step = anchor_global_step + prefix_k
    if prefix_end_global_step > task_horizon:
        raise PrefixOutsideTaskHorizon(
            "cached prefix exceeds fixed task horizon: "
            f"anchor_global_step={anchor_global_step} + prefix_k={prefix_k} "
            f"> task_horizon={task_horizon}"
        )
    remaining_after_prefix = max(0, task_horizon - prefix_end_global_step)
    effective_action_budget = min(action_budget, remaining_after_prefix)
    return {
        "configured_operator_horizon": configured_operator_horizon,
        "configured_tail_horizon": tail_horizon,
        "effective_execution_horizon": effective_execution_horizon,
        "configured_action_budget": action_budget,
        "effective_action_budget": effective_action_budget,
        "configured_policy_call_cap": policy_call_cap,
        "task_horizon": task_horizon,
        "anchor_global_step": anchor_global_step,
        "prefix_end_global_step": prefix_end_global_step,
        "task_horizon_remaining_after_prefix": remaining_after_prefix,
        "task_horizon_budget_clipped": bool(effective_action_budget < action_budget),
    }


@dataclass
class TraceTracker:
    task: str
    initial_object_z: float
    previous_gripper: float
    ever_stably_lifted: bool
    post_detection_steps: int = 0
    gripper_transition_indices: list[int] = field(default_factory=list)
    injected: bool = False
    injection_contact: bool = False
    violation: bool = False
    immediate_violation: bool = False
    violation_type: str | None = None
    first_violation_offset: int | None = None
    shifted_target_qpos: list[float] | None = None
    contact_event_indices: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task, "initial_object_z": float(self.initial_object_z),
            "previous_gripper": float(self.previous_gripper),
            "ever_stably_lifted": bool(self.ever_stably_lifted),
            "post_detection_steps": int(self.post_detection_steps),
            "gripper_transition_indices": list(self.gripper_transition_indices),
            "injected": False, "injection_contact": False, "violation": False,
            "immediate_violation": False, "violation_type": None,
            "first_violation_offset": None, "shifted_target_qpos": None,
            "contact_event_indices": [],
        }

    def observe(self, row: dict[str, Any], action: Any, new_action: bool) -> None:
        phase = row.get("task_phase") or {}
        if new_action:
            self.post_detection_steps += 1
        if bool(phase.get("ever_lifted", False)):
            self.ever_stably_lifted = True
        if bool(phase.get("release_transition", False)):
            self.gripper_transition_indices.append(int(row["step"]))
        current = np.asarray(action, dtype=np.float32).reshape(-1)
        if current.size:
            self.previous_gripper = float(current[-1])


class RuntimeBlocked(RuntimeError):
    """A required provenance, replay, or budget contract failed."""


class PrefixOutsideTaskHorizon(RuntimeBlocked):
    """The cached clean prefix would execute beyond the fixed task horizon."""


def _event_id(event: dict[str, Any]) -> str:
    value, event_id = event.get("event_instance_id"), event.get("event_id")
    if not event_id or value != event_id:
        raise RuntimeBlocked("event_instance_id must be present and equal to event_id")
    return str(value)


def _validate_event_shape(event: dict[str, Any]) -> None:
    _event_id(event)
    required = (
        "task", "actor_seed", "checkpoint_sha256", "init_state", "init_state_hash",
        "pre_anchor_actions", "pre_anchor_actions_hash", "anchor_state", "anchor_state_hash",
        "state_history", "state_history_hash", "action_history", "action_history_hash",
        "original_chunk", "original_chunk_hash", "anchor_global_step",
        "initial_manipulated_qpos", "task_phase", "anchor_contacts",
    )
    missing = [key for key in required if key not in event]
    if missing:
        raise RuntimeBlocked(f"event missing required D provenance: {missing}")
    if "stable_lift_two_steps" not in event["task_phase"]:
        raise RuntimeBlocked("event task_phase lacks stable_lift_two_steps")
    if event.get("source_is_demonstration_chunk"):
        raise RuntimeBlocked("demonstration chunk is not admissible")
    if not event.get("source_is_actor_generated_chunk"):
        raise RuntimeBlocked("actor-generated source provenance is missing")
    if event.get("global_step_fallback_used"):
        raise RuntimeBlocked("global-step fallback event is not admissible")
    if event["task"] not in TASK_SPECS:
        raise RuntimeBlocked(f"unknown task: {event['task']}")
    try:
        int(event["actor_seed"])
    except (TypeError, ValueError) as exc:
        raise RuntimeBlocked("invalid generator actor_seed") from exc


def _configure_local_libero_assets() -> None:
    """Require the same repository LIBERO and asset bundle as collection."""
    from .assets import configure_assets
    configure_assets()


def _environment_hash(env: Any, task: str) -> str:
    model = getattr(getattr(env, "sim", None), "model", None)
    descriptor: dict[str, Any] = {
        "task": task,
        "env_class": f"{type(env).__module__}.{type(env).__qualname__}",
        "sim_class": None if model is None else f"{type(model).__module__}.{type(model).__qualname__}",
    }
    for name in ("nq", "nv", "nu", "nbody", "ngeom", "njnt"):
        value = getattr(model, name, None)
        if value is not None:
            try:
                descriptor[name] = int(value)
            except (TypeError, ValueError):
                descriptor[name] = str(value)
    getter = getattr(model, "get_xml", None)
    if callable(getter):
        try:
            descriptor["xml_sha256"] = hashlib.sha256(str(getter()).encode()).hexdigest()
        except Exception:
            descriptor["xml_sha256"] = None
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _event_geometry_names(event: dict[str, Any]) -> dict[str, list[str]]:
    geometry = event.get("object_geometry") or {}
    result = {"object": [], "target": [], "robot": []}
    for key, source in (("object", "manipulated"), ("target", "target")):
        item = geometry.get(source) or {}
        result[key] = sorted({
            str(geom.get("geom")) for geom in item.get("geoms", [])
            if isinstance(geom, dict) and geom.get("geom")
        })
    result["robot"] = sorted({str(item) for item in event.get("robot_geometry_names", []) if item})
    return result


def _attach_geometry_metadata(env: Any, event: dict[str, Any]) -> None:
    names = _event_geometry_names(event)
    env._r16_object_geometry_names = names["object"]
    env._r16_target_geometry_names = names["target"]
    env._r16_robot_geometry_names = names["robot"]


def _normalize_contacts(value: Any) -> list[list[str]]:
    if value is None:
        return []
    pairs = set()
    for item in value:
        pair = (
            item.get("normalized_pair") or (item.get("geom1_name"), item.get("geom2_name"))
            if isinstance(item, dict) else item
        )
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            pairs.add(tuple(sorted((str(pair[0]), str(pair[1])))))
    return [list(pair) for pair in sorted(pairs)]


def _actor_call(bundle: ActorBundle, env: Any, history: ActorHistory, task: str) -> tuple[np.ndarray, bool]:
    before = (
        state_sha256(np.asarray(env.get_sim_state(), dtype=np.float64)),
        raw_array_hash(history.state_array(), np.float32),
        raw_array_hash(history.action_array(), np.float32),
    )
    chunk = np.ascontiguousarray(bundle.predict(history.state_array(), history.action_array(), task), dtype=np.float32)
    after = (
        state_sha256(np.asarray(env.get_sim_state(), dtype=np.float64)),
        raw_array_hash(history.state_array(), np.float32),
        raw_array_hash(history.action_array(), np.float32),
    )
    side_effect_free = before == after
    if not side_effect_free:
        raise RuntimeBlocked("recovery actor mutated environment or history")
    if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM or not np.isfinite(chunk).all():
        raise RuntimeBlocked(f"invalid recovery actor chunk shape={chunk.shape}")
    return chunk, side_effect_free


def _frozen_signature(env: Any, history: ActorHistory, event: dict[str, Any], tracker: TraceTracker, executed_prefix: list[np.ndarray]) -> dict[str, Any]:
    if _frozen_branch_signature is None:
        raise RuntimeBlocked("frozen Stage2D runtime is unavailable")
    try:
        return _frozen_branch_signature(env, history, event, tracker, executed_prefix)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeBlocked(f"branch signature unavailable: {type(exc).__name__}: {exc}") from exc


def _execute_impl(
    event: dict[str, Any],
    recovery_actor_seed: int,
    behavior_operator: str,
    result_operator: str,
    prefix_k: int,
    tail_horizon: int,
    action_budget: int,
    policy_call_cap: int,
    device: str,
    trace_path: str | None,
) -> dict[str, Any]:
    _validate_event_shape(event)
    if behavior_operator not in OPERATORS:
        raise ValueError(f"unsupported operator: {behavior_operator}")
    if not 2 <= int(prefix_k) <= 16:
        raise ValueError("prefix_k must be in [2,16]")
    contract = execution_contract(
        event["task"], behavior_operator, int(prefix_k), int(tail_horizon),
        int(action_budget), int(policy_call_cap), int(event["anchor_global_step"]),
    )
    event_instance_id = _event_id(event)
    branch_start = time.perf_counter()
    generator_seed = int(event["actor_seed"])
    if _frozen_reconstruct_anchor is None:
        raise RuntimeBlocked("frozen Stage2D runtime is unavailable")
    _configure_local_libero_assets()
    generator_bundle = ActorBundle.load(generator_seed, device)
    env, history, reconstruction = _frozen_reconstruct_anchor(event, generator_bundle)
    recorder: SimulationStepRecorder | None = None
    recovery_chunks: list[str] = []
    inference_side_effects: list[bool] = []
    executed_prefix: list[np.ndarray] = []
    recovery_action_hashes: list[str] = []
    tracker: TraceTracker | None = None
    error: BaseException | None = None
    try:
        if not all(bool(value) for key, value in reconstruction.items() if key != "max_anchor_state_error"):
            raise RuntimeBlocked(f"anchor reconstruction checks failed: {reconstruction}")
        if float(reconstruction.get("max_anchor_state_error", float("inf"))) > 1e-9:
            raise RuntimeBlocked("anchor reconstruction error exceeds 1e-9")
        _attach_geometry_metadata(env, event)
        env_hash = _environment_hash(env, event["task"])
        if not env_hash:
            raise RuntimeBlocked("environment hash missing")
        recovery_bundle = ActorBundle.load(int(recovery_actor_seed), device)
        recovery_checkpoint_sha256 = getattr(recovery_bundle, "checkpoint_sha256", None)
        initial = np.asarray(event.get("initial_manipulated_qpos"), dtype=np.float64)
        if initial.ndim != 1 or initial.size < 3:
            raise RuntimeBlocked("initial_manipulated_qpos provenance missing")
        tracker = TraceTracker(event["task"], float(initial[2]), float(history.actions[-1][-1]), bool(event["task_phase"]["stable_lift_two_steps"]))
        recorder = SimulationStepRecorder(
            env, event["task"], event_instance_id=event_instance_id, prefix_k=int(prefix_k),
            operator=result_operator, actor_seed=int(recovery_actor_seed),
            initial_object_z=float(initial[2]), trace_path=trace_path,
            baseline_contacts=_normalize_contacts(event.get("anchor_contacts")),
        )
        step_index = 0

        def run_action(action: Any, new_action: bool) -> None:
            nonlocal step_index
            action_array = np.asarray(action, dtype=np.float32).reshape(-1)
            if action_array.size != ACTION_DIM:
                raise RuntimeBlocked(f"invalid action dimension {action_array.size}")
            previous = float(tracker.previous_gripper)
            recorder.set_action(step_index, action_array, previous, tracker.ever_stably_lifted)
            observation, _, _, _ = env.step(action_array)
            history.update(observation, action_array)
            row = recorder.finish_action()
            tracker.observe(row, action_array, new_action)
            if new_action:
                recovery_action_hashes.append(raw_array_hash(action_array, np.float32))
            else:
                executed_prefix.append(action_array.copy())
            step_index += 1

        old = np.asarray(event["original_chunk"], dtype=np.float32).reshape(-1, ACTION_DIM)
        if old.shape[0] < int(prefix_k):
            raise RuntimeBlocked("original chunk shorter than requested prefix")
        detection_signature = _frozen_signature(env, history, event, tracker, [])
        for action in old[: int(prefix_k)]:
            run_action(action, False)
        pre_tail_signature = _frozen_signature(env, history, event, tracker, executed_prefix)

        configured_operator_horizon = int(contract["configured_operator_horizon"])
        effective_execution_horizon = int(contract["effective_execution_horizon"])
        configured_new_action_budget = int(contract["configured_action_budget"])
        effective_new_action_budget = int(contract["effective_action_budget"])
        actual_new_actions = 0
        actual_policy_calls = 0
        prelude = _OPERATOR_PRELUDE[behavior_operator]
        if prelude is not None and actual_new_actions < effective_new_action_budget:
            action = _hold_action(old[int(prefix_k) - 1]) if prelude == "hold" else _rollback_action(old[int(prefix_k) - 1])
            run_action(action, True)
            actual_new_actions += 1

        while actual_new_actions < effective_new_action_budget and actual_policy_calls < int(policy_call_cap) and not bool(env.check_success()):
            chunk, side_effect_free = _actor_call(recovery_bundle, env, history, event["task"])
            inference_side_effects.append(side_effect_free)
            recovery_chunks.append(chunk_hash(chunk))
            actual_policy_calls += 1
            execution_count = min(effective_execution_horizon, effective_new_action_budget - actual_new_actions, int(chunk.shape[0]))
            if execution_count <= 0:
                break
            for action in chunk[:execution_count]:
                run_action(action, True)
                actual_new_actions += 1
                if bool(env.check_success()) or actual_new_actions >= effective_new_action_budget:
                    break

        recorder.close()
        labels = label_trace(recorder.records, event["task"])
        task_success = bool(env.check_success())
        safe_label = labels["release_based"]
        safe_success = bool(task_success and not safe_label["violation"])
        required_missing = set(labels.get("missing", []))
        for row in recorder.records:
            required_missing.update(row.get("missing", []))
        if not recorder.physics_instrumented:
            required_missing.add("physics_step_instrumentation")
        if recorder.physics_step_count <= 0:
            required_missing.add("physics_step_count")
        if trace_path is not None and recorder.trace_sha256 is None:
            required_missing.add("trace_output")
        status = "OK" if not required_missing else "BLOCKED_BY_MISSING_PROVENANCE"
        final_signature = _frozen_signature(env, history, event, tracker, executed_prefix)
        return {
            "schema_version": 2, "event_id": event.get("event_id"), "event_instance_id": event_instance_id,
            "task": event["task"], "split": event.get("split"), "init_state_id": event.get("init_state_id"),
            "generator_actor_seed": generator_seed, "recovery_actor_seed": int(recovery_actor_seed),
            "operator": result_operator, "behavior_operator": behavior_operator,
            "prefix_k": int(prefix_k), "configured_prefix_actions": int(prefix_k), "actual_prefix_actions": int(prefix_k),
            "anchor_global_step": int(contract["anchor_global_step"]),
            "task_horizon": int(contract["task_horizon"]),
            "task_horizon_remaining_after_prefix": int(contract["task_horizon_remaining_after_prefix"]),
            "task_horizon_budget_clipped": bool(contract["task_horizon_budget_clipped"]),
            "configured_tail_horizon": int(contract["configured_tail_horizon"]),
            "configured_operator_horizon": configured_operator_horizon,
            "effective_execution_horizon": effective_execution_horizon,
            "configured_action_budget": configured_new_action_budget,
            "effective_new_action_budget": effective_new_action_budget,
            "actual_new_recovery_actions": int(actual_new_actions),
            "actual_total_action_steps": int(prefix_k + actual_new_actions),
            "actual_global_step": int(contract["anchor_global_step"] + prefix_k + actual_new_actions),
            "configured_policy_call_cap": int(contract["configured_policy_call_cap"]),
            "validation_policy_calls": 1,
            "recovery_policy_calls": int(actual_policy_calls),
            "total_policy_calls": int(1 + actual_policy_calls),
            "actual_policy_calls": int(actual_policy_calls),
            "policy_call_cap_respected": actual_policy_calls <= int(policy_call_cap) <= 8,
            "action_budget_respected": actual_new_actions <= configured_new_action_budget, "prefix_budget_separate": True,
            "generator_actor_calls": 1, "validation_policy_calls": 1,
            "generator_checkpoint_sha256": event.get("checkpoint_sha256"),
            "recovery_checkpoint_sha256": recovery_checkpoint_sha256,
            "generator_chunk_hash": event["original_chunk_hash"],
            "chunk_hash": event["original_chunk_hash"], "recovery_chunk_hashes": recovery_chunks,
            "recovery_action_hashes": recovery_action_hashes,
            "pid": os.getpid(), "parent_pid": os.getppid(), "process_start_method": "spawn",
            "fresh_environment_created": True, "env_hash": env_hash, "zero_injection": True, "injection_calls": 0,
            "reconstruction": reconstruction,
            "d4_signatures": {"detection": detection_signature, "pre_tail": pre_tail_signature, "final": final_signature},
            "trace_records": int(len(recorder.records)), "physics_instrumented": bool(recorder.physics_instrumented),
            "physics_step_count": int(recorder.physics_step_count), "physics_probe_error": recorder.physics_probe_error,
            "trace_sha256": recorder.trace_sha256, "trace_content_sha256": trace_content_sha256(recorder.records),
            "labels": labels, "release_based_violation": bool(safe_label["violation"]),
            "topology_candidate": labels["prerelease_topology_candidate"], "task_success": task_success,
            "safe_success": bool(safe_success and status == "OK"), "safe_success_unblocked_observation": safe_success,
            "actor_inference_side_effect_free": all(inference_side_effects),
            "diagnostic_only": True, "formal_positive_evidence_allowed": False, "new_idea_generated": False,
            "status": status, "blocked": status != "OK", "missing_provenance": sorted(required_missing),
            "error": None, "total_branch_wall_time_s": float(time.perf_counter() - branch_start),
        }
    except BaseException as exc:
        error = exc
        raise
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except BaseException:
                if error is None:
                    raise
        env.close()


def execute_branch(event: dict[str, Any], recovery_actor_seed: int, operator: str, prefix_k: int, tail_horizon: int, action_budget: int, policy_call_cap: int = 8, device: str = "cpu", trace_path: str | None = None) -> dict[str, Any]:
    """Execute one bounded zero-injection recovery branch."""
    if operator not in OPERATORS:
        raise ValueError(f"operator must be one of {OPERATORS}, got {operator!r}")
    return _execute_impl(event, recovery_actor_seed, operator, operator, int(prefix_k), int(tail_horizon), int(action_budget), int(policy_call_cap), device, trace_path)


def execute_reference_branch(event: dict[str, Any], recovery_actor_seed: int, reference_operator: str, tail_horizon: int, action_budget: int, policy_call_cap: int = 8, device: str = "cpu", trace_path: str | None = None) -> dict[str, Any]:
    """Run a reference arm outside the four recovery operators."""
    if reference_operator not in REFERENCE_OPERATORS:
        raise ValueError(f"unknown reference operator: {reference_operator}")
    if reference_operator == "immediate_fresh":
        prefix_k = 2
    elif reference_operator.startswith("fixed_delay_"):
        prefix_k = 2 + int(reference_operator.rsplit("_", 1)[1])
    else:
        raise ValueError(f"unknown reference operator: {reference_operator}")
    if prefix_k > 16:
        raise ValueError("reference prefix exceeds valid chunk")
    return _execute_impl(event, recovery_actor_seed, "fresh_h16", reference_operator, prefix_k, int(tail_horizon), int(action_budget), int(policy_call_cap), device, trace_path)


__all__ = [
    "OPERATORS", "REFERENCE_OPERATORS", "RuntimeBlocked", "PrefixOutsideTaskHorizon",
    "execution_contract",
    "execute_branch", "execute_reference_branch",
]
