from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from r16_p14_stage2a.envs import (
    contact_pairs,
    current_observation,
    feature_from_obs,
    joint_qpos,
    make_env,
    restore_state,
    set_joint_qpos,
    state_sha256,
)
from r16_p14_stage2a.mechanics import cause_contact
from r16_p14_stage2a.settings import ACTION_DIM, TASK_SPECS
from r16_p14_stage2b.io_utils import sha256_array, sha256_json
from r16_p14_stage2b.runtime import ActorBundle, ActorHistory, chunk_hash, gripper_release_index

from .goal_geometry import goal_geometry, target_distance
from .settings import DETECTION_PREFIX, H_VALID


def _jsonable(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return type(value).__name__
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item, depth + 1) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    return repr(value)


def controller_snapshot(env) -> dict[str, Any]:
    controllers = []
    for robot_index, robot in enumerate(env.robots):
        items = robot.controller.values() if isinstance(robot.controller, dict) else [robot.controller]
        for controller_index, controller in enumerate(items):
            values = {}
            for key, value in sorted(vars(controller).items()):
                if key.startswith("_") or callable(value):
                    continue
                if isinstance(value, (np.ndarray, np.generic, int, float, bool, str, type(None), list, tuple)):
                    values[key] = _jsonable(value)
            controllers.append({"robot": robot_index, "controller": controller_index, "class": type(controller).__name__, "values": values})
    return {"controllers": controllers}


def rng_snapshot(env) -> dict[str, Any]:
    result = {}
    for name, owner in (("wrapper", env), ("environment", env.env)):
        for attr in ("np_random", "_np_random"):
            rng = getattr(owner, attr, None)
            if rng is None:
                continue
            if hasattr(rng, "get_state"):
                result[f"{name}.{attr}"] = _jsonable(rng.get_state())
            elif hasattr(rng, "bit_generator"):
                result[f"{name}.{attr}"] = _jsonable(rng.bit_generator.state)
    return result


def runtime_trace_row(env, history: ActorHistory, *, global_step: int, action: np.ndarray | None) -> dict[str, Any]:
    state = np.ascontiguousarray(env.get_sim_state(), dtype=np.float64)
    feature = np.ascontiguousarray(feature_from_obs(current_observation(env)), dtype=np.float32)
    states = np.ascontiguousarray(history.state_array(), dtype=np.float32)
    actions = np.ascontiguousarray(history.action_array(), dtype=np.float32)
    controller = controller_snapshot(env)
    rng = rng_snapshot(env)
    return {
        "global_step": int(global_step),
        "action_hash": None if action is None else sha256_array(action, np.float32),
        "simulator_state_hash": state_sha256(state),
        "observation_feature_hash": sha256_array(feature, np.float32),
        "state_history_hash": sha256_array(states, np.float32),
        "action_history_hash": sha256_array(actions, np.float32),
        "controller_hash": sha256_json(controller),
        "rng_hash": sha256_json(rng),
        "wrapper": {"timestep": int(env.env.timestep)},
    }


def nominal_chunk_trace(env, task: str, anchor_state: np.ndarray, chunk: np.ndarray) -> dict[str, Any]:
    spec = TASK_SPECS[task]
    assert spec.manipulated_joint
    env.seed(0)
    env.reset()
    restore_state(env, anchor_state)
    positions = [joint_qpos(env, spec.manipulated_joint)]
    distances = [target_distance(env, task)]
    contacts = [list(contact_pairs(env))]
    for action in np.asarray(chunk, dtype=np.float32)[:H_VALID]:
        env.step(action)
        positions.append(joint_qpos(env, spec.manipulated_joint))
        distances.append(target_distance(env, task))
        contacts.append(list(contact_pairs(env)))
    return {
        "positions": np.asarray(positions, dtype=np.float64),
        "distances": np.asarray(distances, dtype=np.float64),
        "contacts": contacts,
        "displacement_m": float(np.linalg.norm(positions[-1][:3] - positions[0][:3])),
        "minimum_goal_distance": float(min(distances)),
    }


def validate_event_bytes(event: dict[str, Any], bundle: ActorBundle | None = None) -> None:
    arrays = {
        "init_state": (np.float64, "init_state_hash"),
        "pre_anchor_actions": (np.float32, "pre_anchor_actions_hash"),
        "anchor_state": (np.float64, "anchor_state_hash"),
        "state_history": (np.float32, "state_history_hash"),
        "action_history": (np.float32, "action_history_hash"),
        "original_chunk": (np.float32, "original_chunk_hash"),
    }
    failed = []
    for name, (dtype, hash_name) in arrays.items():
        value = np.ascontiguousarray(event[name], dtype=dtype)
        digest = state_sha256(value) if dtype == np.float64 and name == "anchor_state" else sha256_array(value, dtype)
        if digest != event[hash_name]:
            failed.append(name)
    if not event.get("source_is_actor_generated_chunk") or event.get("source_is_demonstration_chunk"):
        failed.append("actor_generated_source")
    if bundle is not None and (bundle.checkpoint_sha256 != event["checkpoint_sha256"] or bundle.seed != int(event["actor_seed"])):
        failed.append("checkpoint")
    if failed:
        raise ValueError(f"event byte contract rejected: {failed}")


def reconstruct_anchor(event: dict[str, Any], bundle: ActorBundle, *, capture_trace: bool = True):
    validate_event_bytes(event, bundle)
    env, _ = make_env(event["task"], seed=0)
    try:
        env.seed(0)
        env.reset()
        observation = restore_state(env, np.asarray(event["init_state"], dtype=np.float64))
        history = ActorHistory.initial(observation)
        initial_trace = runtime_trace_row(env, history, global_step=0, action=None)
        trace = [initial_trace] if capture_trace else []
        for global_step, action in enumerate(np.asarray(event["pre_anchor_actions"], dtype=np.float32), start=1):
            observation, _, _, _ = env.step(action)
            history.update(observation, action)
            trace_row = runtime_trace_row(env, history, global_step=global_step, action=action)
            if capture_trace:
                trace.append(trace_row)
        state = np.ascontiguousarray(env.get_sim_state(), dtype=np.float64)
        states = np.ascontiguousarray(history.state_array(), dtype=np.float32)
        actions = np.ascontiguousarray(history.action_array(), dtype=np.float32)
        chunk = np.ascontiguousarray(bundle.predict(states, actions, event["task"]), dtype=np.float32)
        checks = {
            "anchor_state_exact": state_sha256(state) == event["anchor_state_hash"],
            "state_history_exact": sha256_array(states, np.float32) == event["state_history_hash"],
            "action_history_exact": sha256_array(actions, np.float32) == event["action_history_hash"],
            "original_chunk_exact": chunk_hash(chunk) == event["original_chunk_hash"],
        }
        return env, history, {
            "state": state,
            "states": states,
            "actions": actions,
            "chunk": chunk,
            "trace": trace,
            "checks": checks,
            "passed": all(checks.values()),
        }
    except Exception:
        env.close()
        raise


def first_trace_divergence(saved: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> dict[str, Any] | None:
    keys = (
        "action_hash", "simulator_state_hash", "observation_feature_hash",
        "state_history_hash", "action_history_hash", "controller_hash", "rng_hash", "wrapper",
    )
    for index in range(max(len(saved), len(fresh))):
        if index >= len(saved) or index >= len(fresh):
            return {"global_step": index, "key": "missing_trace_row", "saved_rows": len(saved), "fresh_rows": len(fresh)}
        for key in keys:
            if saved[index].get(key) != fresh[index].get(key):
                return {"global_step": int(fresh[index]["global_step"]), "key": key, "saved": saved[index].get(key), "fresh": fresh[index].get(key)}
    return None


def numeric_difference(saved: Any, fresh: Any) -> dict[str, Any]:
    """Return a compact, JSON-safe exactness audit for two numeric tensors."""
    left = np.asarray(saved)
    right = np.asarray(fresh)
    result: dict[str, Any] = {
        "saved_shape": list(left.shape),
        "fresh_shape": list(right.shape),
        "shape_exact": left.shape == right.shape,
    }
    if left.shape != right.shape:
        result.update({"byte_exact": False, "nonzero_count": None, "max_abs_difference": None, "first_difference": None})
        return result
    byte_exact = (
        left.dtype == right.dtype
        and np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes()
    )
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    indices = np.argwhere(difference != 0)
    first = None
    if len(indices):
        index = tuple(int(item) for item in indices[0])
        first = {
            "index": list(index),
            "saved": float(left[index]),
            "fresh": float(right[index]),
            "absolute_difference": float(difference[index]),
        }
    result.update({
        "byte_exact": bool(byte_exact),
        "saved_dtype": str(left.dtype),
        "fresh_dtype": str(right.dtype),
        "nonzero_count": int(len(indices)),
        "max_abs_difference": float(difference.max()) if difference.size else 0.0,
        "first_difference": first,
    })
    return result


def replay_failure_diagnostic(event: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    """Explain a strict replay rejection without changing any admission gate."""
    return {
        "checks": dict(audit["checks"]),
        "first_trace_divergence": first_trace_divergence(event["pre_anchor_trace"], audit.get("trace", [])),
        "anchor_state": numeric_difference(np.asarray(event["anchor_state"], dtype=np.float64), audit["state"]),
        "state_history": numeric_difference(np.asarray(event["state_history"], dtype=np.float32), audit["states"]),
        "action_history": numeric_difference(np.asarray(event["action_history"], dtype=np.float32), audit["actions"]),
        "original_chunk": numeric_difference(np.asarray(event["original_chunk"], dtype=np.float32), audit["chunk"]),
    }


@dataclass
class CauseTracker:
    task: str
    initial_object_z: float
    previous_gripper: float
    ever_lifted: bool
    injected: bool = False
    injection_contact: bool = False
    violation: bool = False
    immediate_violation: bool = False
    violation_type: str | None = None
    first_violation_offset: int | None = None
    post_detection_steps: int = 0

    def mark(self, kind: str, *, immediate: bool = False) -> None:
        if not self.violation:
            self.violation = True
            self.violation_type = kind
            self.first_violation_offset = 0 if immediate else self.post_detection_steps
        self.immediate_violation |= immediate

    def observe_injection(self, env, before_contacts: tuple[tuple[str, str], ...]) -> None:
        self.injected = True
        introduced = set(contact_pairs(env)).difference(before_contacts)
        if self.task != "put_the_cream_cheese_in_the_bowl" and cause_contact(env, TASK_SPECS[self.task]) and introduced:
            self.injection_contact = True
            self.mark("future_path_obstacle_contact_at_injection", immediate=True)

    def observe_action(self, env, action: np.ndarray) -> None:
        spec = TASK_SPECS[self.task]
        assert spec.manipulated_joint
        if self.injected:
            self.post_detection_steps += 1
        qpos = joint_qpos(env, spec.manipulated_joint)
        if qpos[2] - self.initial_object_z >= spec.lift_delta:
            self.ever_lifted = True
        if self.injected and self.task == "put_the_cream_cheese_in_the_bowl":
            release = self.previous_gripper > 0.0 and float(action[-1]) < 0.0
            if release and self.ever_lifted:
                assert spec.target_joint
                target = joint_qpos(env, spec.target_joint)
                if np.linalg.norm(qpos[:2] - target[:2]) > spec.placement_xy_tolerance:
                    self.mark("misaligned_release_after_target_shift")
        elif self.injected and cause_contact(env, spec):
            self.mark("manipulated_object_future_path_obstacle_contact")
        self.previous_gripper = float(action[-1])

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def lateral_unit(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction = np.asarray(end[:2] - start[:2], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    direction /= norm
    return np.asarray([-direction[1], direction[0]], dtype=np.float64)


def apply_perturbation(env, event: dict[str, Any], parameter: dict[str, Any]) -> dict[str, Any]:
    task = event["task"]
    spec = TASK_SPECS[task]
    assert spec.manipulated_joint
    manipulated_before = joint_qpos(env, spec.manipulated_joint)
    before_contacts = contact_pairs(env)
    if task == "put_the_cream_cheese_in_the_bowl":
        assert spec.target_joint
        joint = spec.target_joint
        before = joint_qpos(env, joint)
        after = before.copy()
        magnitude = float(parameter["magnitude_m"])
        direction = -1.0 if manipulated_before[0] - before[0] >= 0 else 1.0
        after[0] += direction * magnitude
        kind = "target_object_shift"
    else:
        assert spec.obstacle_joint
        joint = spec.obstacle_joint
        before = joint_qpos(env, joint)
        after = before.copy()
        path_index = int(parameter["future_index"])
        positions = np.asarray(event["nominal_trace"]["positions"], dtype=np.float64)
        future = positions[path_index]
        prior = positions[max(0, path_index - 1)]
        lateral = lateral_unit(prior, future)
        after[:2] = future[:2] + lateral * float(parameter["lateral_clearance_m"])
        kind = "future_swept_path_obstacle"
    set_joint_qpos(env, joint, after)
    manipulated_after = joint_qpos(env, spec.manipulated_joint)
    manipulated_error = float(np.max(np.abs(manipulated_after - manipulated_before)))
    if manipulated_error != 0.0:
        raise RuntimeError(f"manipulated object teleported by injection: {manipulated_error}")
    return {
        "kind": kind,
        "parameter": parameter,
        "joint": joint,
        "before_qpos": before.tolist(),
        "after_qpos": after.tolist(),
        "before_contacts": list(before_contacts),
        "after_contacts": list(contact_pairs(env)),
        "manipulated_qpos_max_error": manipulated_error,
    }


@dataclass
class PrefixContext:
    env: Any
    history: ActorHistory
    tracker: CauseTracker
    perturbation: dict[str, Any]
    prefix_k: int

    def close(self) -> None:
        self.env.close()


def reconstruct_to_prefix(event: dict[str, Any], parameter: dict[str, Any], prefix_k: int, bundle: ActorBundle) -> PrefixContext:
    if prefix_k not in range(DETECTION_PREFIX, H_VALID + 1):
        raise ValueError(prefix_k)
    env, history, audit = reconstruct_anchor(event, bundle, capture_trace=False)
    try:
        if not audit["passed"]:
            raise ValueError(f"anchor replay rejected: {audit['checks']}")
        initial = np.asarray(event["initial_manipulated_qpos"], dtype=np.float64)
        tracker = CauseTracker(
            task=event["task"], initial_object_z=float(initial[2]),
            previous_gripper=float(history.actions[-1][-1]),
            ever_lifted=bool(event["phase"]["ever_lifted"]),
        )
        perturbation = None
        for index, action in enumerate(np.asarray(event["original_chunk"], dtype=np.float32)[:prefix_k]):
            observation, _, _, _ = env.step(action)
            history.update(observation, action)
            tracker.observe_action(env, action)
            if index + 1 == DETECTION_PREFIX:
                perturbation = apply_perturbation(env, event, parameter)
                tracker.observe_injection(env, tuple(tuple(item) for item in perturbation["before_contacts"]))
                history.refresh_latest_observation(current_observation(env))
        if perturbation is None:
            raise RuntimeError("perturbation was not injected")
        return PrefixContext(env=env, history=history, tracker=tracker, perturbation=perturbation, prefix_k=prefix_k)
    except Exception:
        env.close()
        raise


def rollback_action(last_action: np.ndarray) -> np.ndarray:
    action = np.asarray(last_action, dtype=np.float32).copy()
    action[:6] *= -1.0
    return action


def hold_action(last_action: np.ndarray) -> np.ndarray:
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[-1] = float(np.asarray(last_action, dtype=np.float32)[-1])
    return action
