from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r16_p14_stage2a.actor_eval import load_actor
from r16_p14_stage2a.actor_model import predict_chunk
from r16_p14_stage2a.envs import (
    array_sha256,
    contact_pairs,
    contact_sha256,
    current_observation,
    feature_from_obs,
    joint_qpos,
    make_env,
    restore_state,
    set_joint_qpos,
    state_sha256,
)
from r16_p14_stage2a.mechanics import cause_contact, joint_trajectory, load_demo
from r16_p14_stage2a.settings import TASK_SPECS, TASK_TO_INDEX

from .io_utils import sha256_array, sha256_file, sha256_json
from .settings import (
    ACTION_DIM,
    ACTION_HISTORY,
    CHECKPOINTS,
    CHUNK_LENGTH,
    DETECTION_PREFIX,
    LIBERO_CONFIG,
    OBS_HISTORY,
)


@dataclass
class ActorBundle:
    seed: int
    checkpoint: Path
    checkpoint_sha256: str
    payload: dict[str, Any]
    model: Any
    normalization: Any
    device: torch.device

    @classmethod
    def load(cls, seed: int, device: str | torch.device) -> "ActorBundle":
        target = torch.device(device)
        checkpoint = CHECKPOINTS[seed]
        payload, model, normalization = load_actor(checkpoint, target)
        if int(payload["seed"]) != seed:
            raise ValueError(f"checkpoint seed mismatch: expected {seed}")
        return cls(
            seed=seed,
            checkpoint=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            payload=payload,
            model=model,
            normalization=normalization,
            device=target,
        )

    def predict(self, state_history: Any, action_history: Any, task: str) -> np.ndarray:
        return predict_chunk(
            self.model,
            self.normalization,
            np.asarray(state_history, dtype=np.float32),
            np.asarray(action_history, dtype=np.float32),
            TASK_TO_INDEX[task],
            self.device,
        )


@dataclass
class ActorHistory:
    states: list[np.ndarray]
    actions: list[np.ndarray]

    @classmethod
    def initial(cls, observation: dict[str, Any]) -> "ActorHistory":
        feature = feature_from_obs(observation)
        return cls(
            states=[feature.copy() for _ in range(OBS_HISTORY)],
            actions=[np.zeros(ACTION_DIM, dtype=np.float32) for _ in range(ACTION_HISTORY)],
        )

    def update(self, observation: dict[str, Any], action: Any) -> None:
        action_array = np.asarray(action, dtype=np.float32).copy()
        self.states = self.states[1:] + [feature_from_obs(observation)]
        self.actions = self.actions[1:] + [action_array]

    def refresh_latest_observation(self, observation: dict[str, Any]) -> None:
        """Replace the current-time feature after an action-free world change."""
        self.states[-1] = feature_from_obs(observation)

    def state_array(self) -> np.ndarray:
        return np.asarray(self.states, dtype=np.float32)

    def action_array(self) -> np.ndarray:
        return np.asarray(self.actions, dtype=np.float32)

    def hashes(self) -> dict[str, Any]:
        states = self.state_array()
        actions = self.action_array()
        return {
            "state_history_hash": sha256_array(states, np.float32),
            "state_feature_hashes": [sha256_array(item, np.float32) for item in states],
            "action_history_hash": sha256_array(actions, np.float32),
            "action_hashes": [sha256_array(item, np.float32) for item in actions],
        }


def chunk_hash(chunk: Any) -> str:
    return sha256_array(chunk, np.float32)


def validate_event_contract(event: dict[str, Any], bundle: ActorBundle | None = None) -> dict[str, bool]:
    """Validate immutable event bytes before opening a simulator branch.

    Keeping this validation independent of MuJoCo makes corruption fail closed
    before any outcome can be generated and gives the metamorphic tests a pure
    contract surface.
    """
    state_history = np.asarray(event["state_history"], dtype=np.float32)
    action_history = np.asarray(event["action_history"], dtype=np.float32)
    original_chunk = np.asarray(event["original_chunk"], dtype=np.float32)
    pre_anchor = np.asarray(event["pre_anchor_actions"], dtype=np.float32).reshape(-1, ACTION_DIM)
    checks = {
        "actor_generated_nominal": bool(event.get("source_is_actor_generated_chunk")),
        "demonstration_nominal_forbidden": not bool(event.get("source_is_demonstration_chunk")),
        "state_history_shape": state_history.ndim == 2 and state_history.shape[0] == OBS_HISTORY,
        "action_history_shape": action_history.shape == (ACTION_HISTORY, ACTION_DIM),
        "chunk_shape": original_chunk.shape == (CHUNK_LENGTH, ACTION_DIM),
        "state_history_bytes": sha256_array(state_history, np.float32) == event["state_history_hash"],
        "action_history_bytes": sha256_array(action_history, np.float32) == event["action_history_hash"],
        "original_chunk_bytes": chunk_hash(original_chunk) == event["original_chunk_hash"],
        "pre_anchor_action_bytes": sha256_array(pre_anchor, np.float32) == event["pre_anchor_actions_hash"],
        "init_state_bytes": sha256_array(event["init_state"], np.float64) == event["init_state_hash"],
        "anchor_state_bytes": state_sha256(np.asarray(event["anchor_state"], dtype=np.float64))
        == event["anchor_state_hash"],
    }
    if bundle is not None:
        checks["checkpoint_hash"] = bundle.checkpoint_sha256 == event["checkpoint_sha256"]
        checks["actor_seed"] = int(bundle.seed) == int(event["actor_seed"])
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"event contract rejected: {failed}")
    return checks


def normalized_disagreement(old_action: Any, fresh_action: Any, bundle: ActorBundle) -> float:
    old = np.asarray(old_action, dtype=np.float32)
    fresh = np.asarray(fresh_action, dtype=np.float32)
    scale = np.maximum(np.asarray(bundle.normalization.action_std, dtype=np.float32), 1e-4)
    return float(np.sqrt(np.mean(((old - fresh) / scale) ** 2)))


def task_goal(env, task: str) -> np.ndarray:
    spec = TASK_SPECS[task]
    if task == "put_the_cream_cheese_in_the_bowl":
        assert spec.target_joint
        return joint_qpos(env, spec.target_joint)
    states, _ = load_demo(task, 0)
    assert spec.manipulated_joint
    return np.asarray(joint_trajectory(env, states, spec.manipulated_joint)[-1], dtype=np.float64)


def target_distance(env, task: str, goal: np.ndarray | None = None) -> float:
    spec = TASK_SPECS[task]
    assert spec.manipulated_joint
    manipulated = joint_qpos(env, spec.manipulated_joint)
    target = task_goal(env, task) if goal is None else np.asarray(goal, dtype=np.float64)
    if task == "put_the_cream_cheese_in_the_bowl":
        return float(np.linalg.norm(manipulated[:2] - target[:2]))
    return float(np.linalg.norm(manipulated[:3] - target[:3]))


def gripper_release_index(chunk: Any, previous_gripper: float, horizon: int) -> int | None:
    previous = float(previous_gripper)
    for index, action in enumerate(np.asarray(chunk, dtype=np.float32)[:horizon]):
        current = float(action[-1])
        if previous > 0.0 and current < 0.0:
            return index
        previous = current
    return None


@dataclass
class TaskMonitor:
    task: str
    initial_object: np.ndarray
    goal: np.ndarray
    previous_gripper: float = -1.0
    stable_lift_count: int = 0
    ever_stably_lifted: bool = False
    late_phase_reached: bool = False
    object_drop: bool = False
    wrong_release: bool = False
    phase_regression: bool = False
    valid_release: bool = False
    best_distance: float = math.inf

    @classmethod
    def create(cls, env, task: str) -> "TaskMonitor":
        spec = TASK_SPECS[task]
        assert spec.manipulated_joint
        initial = joint_qpos(env, spec.manipulated_joint)
        goal = task_goal(env, task)
        instance = cls(task=task, initial_object=initial, goal=goal)
        instance.best_distance = target_distance(env, task, goal)
        return instance

    def observe(self, env, action: Any, success: bool) -> None:
        spec = TASK_SPECS[self.task]
        assert spec.manipulated_joint
        action = np.asarray(action, dtype=np.float32)
        manipulated = joint_qpos(env, spec.manipulated_joint)
        lifted = float(manipulated[2] - self.initial_object[2]) >= spec.lift_delta
        self.stable_lift_count = self.stable_lift_count + 1 if lifted else 0
        self.ever_stably_lifted |= self.stable_lift_count >= 3
        distance = target_distance(env, self.task, self.goal)
        late_bound = 0.16 if self.task == "put_the_cream_cheese_in_the_bowl" else 0.18
        self.late_phase_reached |= self.ever_stably_lifted and distance <= late_bound
        release = self.previous_gripper > 0.0 and float(action[-1]) < 0.0
        if release and self.ever_stably_lifted:
            tolerance = spec.placement_xy_tolerance if self.task == "put_the_cream_cheese_in_the_bowl" else 0.12
            if distance <= tolerance or success:
                self.valid_release = True
            else:
                self.wrong_release = True
        if self.ever_stably_lifted and not success and not self.valid_release:
            drop_threshold = self.initial_object[2] + max(0.005, spec.lift_delta - 0.02)
            if manipulated[2] < drop_threshold:
                self.object_drop = True
        if self.late_phase_reached and distance > self.best_distance + 0.05:
            self.phase_regression = True
        self.best_distance = min(self.best_distance, distance)
        self.previous_gripper = float(action[-1])

    def late_anchor_preconditions(self, env, chunk: Any, horizon: int) -> dict[str, bool]:
        spec = TASK_SPECS[self.task]
        distance = target_distance(env, self.task, self.goal)
        return {
            "stable_lift": self.ever_stably_lifted,
            "gripper_closed": self.previous_gripper > 0.0,
            "plausible_release": gripper_release_index(chunk, self.previous_gripper, horizon) is not None,
            "target_distance": distance <= (0.16 if self.task == "put_the_cream_cheese_in_the_bowl" else 0.18),
            "task_not_successful": not bool(env.check_success()),
            "no_current_cause_contact": (
                True if self.task == "put_the_cream_cheese_in_the_bowl" else not cause_contact(env, spec)
            ),
        }

    def faithful(self, progress_delta: float, success: bool) -> bool:
        return bool(
            not self.object_drop
            and not self.wrong_release
            and not self.phase_regression
            and progress_delta >= -0.05
            and (self.ever_stably_lifted or success)
        )


def nominal_chunk_trace(
    env,
    *,
    task: str,
    anchor_state: Any,
    chunk: Any,
    horizon: int,
    goal: Any,
) -> dict[str, Any]:
    spec = TASK_SPECS[task]
    assert spec.manipulated_joint
    env.reset()
    restore_state(env, np.asarray(anchor_state, dtype=np.float64))
    positions = [joint_qpos(env, spec.manipulated_joint)]
    distances = [target_distance(env, task, np.asarray(goal, dtype=np.float64))]
    success = bool(env.check_success())
    for action in np.asarray(chunk, dtype=np.float32)[:horizon]:
        if success:
            break
        env.step(action)
        positions.append(joint_qpos(env, spec.manipulated_joint))
        distances.append(target_distance(env, task, np.asarray(goal, dtype=np.float64)))
        success = bool(env.check_success())
    best_index = int(np.argmin(distances))
    return {
        "positions": np.asarray(positions, dtype=np.float64),
        "distances": np.asarray(distances, dtype=np.float64),
        "best_index": best_index,
        "best_position": np.asarray(positions[best_index], dtype=np.float64),
        "minimum_distance": float(distances[best_index]),
        "final_distance": float(distances[-1]),
        "success": success,
    }


@dataclass
class CauseTracker:
    task: str
    initial_object_z: float
    previous_gripper: float
    ever_lifted: bool = False
    injected: bool = False
    immediate_violation: bool = False
    violation: bool = False
    violation_type: str | None = None
    first_violation_offset: int | None = None
    post_detection_steps: int = 0

    def mark(self, kind: str, immediate: bool = False) -> None:
        if not self.violation:
            self.violation = True
            self.violation_type = kind
            self.first_violation_offset = 0 if immediate else self.post_detection_steps
        self.immediate_violation |= immediate

    def observe_injection(self, env, before_contacts: tuple[tuple[str, str], ...]) -> None:
        self.injected = True
        if self.task == "put_the_bowl_on_the_stove":
            introduced = set(contact_pairs(env)).difference(before_contacts)
            spec = TASK_SPECS[self.task]
            if cause_contact(env, spec) and introduced:
                self.mark("injection_instant_bowl_blocker_contact", immediate=True)

    def observe_action(self, env, action: Any) -> None:
        spec = TASK_SPECS[self.task]
        assert spec.manipulated_joint
        action = np.asarray(action, dtype=np.float32)
        if self.injected:
            self.post_detection_steps += 1
        position = joint_qpos(env, spec.manipulated_joint)
        if position[2] - self.initial_object_z >= spec.lift_delta:
            self.ever_lifted = True
        if self.injected and self.task == "put_the_bowl_on_the_stove" and cause_contact(env, spec):
            self.mark("post_injection_bowl_blocker_contact")
        if self.injected and self.task == "put_the_cream_cheese_in_the_bowl":
            release = self.previous_gripper > 0.0 and float(action[-1]) < 0.0
            if release and self.ever_lifted:
                assert spec.target_joint
                target = joint_qpos(env, spec.target_joint)
                if np.linalg.norm(position[:2] - target[:2]) > spec.placement_xy_tolerance:
                    self.mark("misaligned_release_after_target_shift")
        self.previous_gripper = float(action[-1])

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _lateral_unit(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction = np.asarray(end[:2] - start[:2], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    direction /= norm
    return np.asarray([-direction[1], direction[0]], dtype=np.float64)


def apply_actor_perturbation(env, event: dict[str, Any], severity_m: float) -> dict[str, Any]:
    task = event["task"]
    spec = TASK_SPECS[task]
    assert spec.manipulated_joint
    manipulated_before = joint_qpos(env, spec.manipulated_joint)
    if task == "put_the_cream_cheese_in_the_bowl":
        assert spec.target_joint
        joint = spec.target_joint
        before = joint_qpos(env, joint)
        after = before.copy()
        direction = -1.0 if manipulated_before[0] - before[0] >= 0.0 else 1.0
        after[0] += direction * float(severity_m)
        kind = "target_free_joint_shift"
    else:
        assert spec.obstacle_joint
        joint = spec.obstacle_joint
        before = joint_qpos(env, joint)
        after = before.copy()
        future = np.asarray(event["nominal_future_object_qpos"], dtype=np.float64)
        lateral = _lateral_unit(manipulated_before, future)
        after[:2] = future[:2] + lateral * float(severity_m)
        kind = "actor_nominal_future_target_region_blocker"
    before_contacts = contact_pairs(env)
    set_joint_qpos(env, joint, after)
    manipulated_after = joint_qpos(env, spec.manipulated_joint)
    error = float(np.max(np.abs(manipulated_after - manipulated_before)))
    if error != 0.0:
        raise RuntimeError(f"manipulated object changed at injection: {error}")
    return {
        "type": kind,
        "severity_m": float(severity_m),
        "joint": joint,
        "before_qpos": before.tolist(),
        "after_qpos": after.tolist(),
        "before_contacts": before_contacts,
        "after_contacts": contact_pairs(env),
        "manipulated_qpos_max_error": error,
    }


def init_state_and_suite(task: str, seed: int = 0):
    env, suite = make_env(task, config_dir=LIBERO_CONFIG, seed=seed)
    spec = TASK_SPECS[task]
    states = np.asarray(suite.get_task_init_states(spec.task_id), dtype=np.float64)
    return env, suite, states


def reconstruct_anchor(env, event: dict[str, Any], bundle: ActorBundle) -> tuple[ActorHistory, dict[str, Any]]:
    validate_event_contract(event, bundle)
    env.seed(0)
    env.reset()
    observation = restore_state(env, np.asarray(event["init_state"], dtype=np.float64))
    history = ActorHistory.initial(observation)
    for action in np.asarray(event["pre_anchor_actions"], dtype=np.float32):
        observation, _, _, _ = env.step(action)
        history.update(observation, action)
    state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
    state_history = history.state_array()
    action_history = history.action_array()
    chunk = bundle.predict(state_history, action_history, event["task"])
    saved_state = np.asarray(event["anchor_state"], dtype=np.float64)
    checks = {
        "checkpoint_hash_match": bundle.checkpoint_sha256 == event["checkpoint_sha256"],
        "anchor_state_hash_match": state_sha256(state) == event["anchor_state_hash"],
        "state_history_hash_match": sha256_array(state_history, np.float32) == event["state_history_hash"],
        "action_history_hash_match": sha256_array(action_history, np.float32) == event["action_history_hash"],
        "original_chunk_hash_match": chunk_hash(chunk) == event["original_chunk_hash"],
        "max_anchor_state_error_le_1e_9": float(np.max(np.abs(state - saved_state))) <= 1e-9,
    }
    return history, {
        "state": state,
        "chunk": chunk,
        "checks": checks,
        "passed": all(checks.values()),
        "max_anchor_state_error": float(np.max(np.abs(state - saved_state))),
    }


@dataclass
class BranchContext:
    env: Any
    history: ActorHistory
    tracker: CauseTracker
    perturbation: dict[str, Any]
    anchor_replay: dict[str, Any]
    prefix_k: int
    state: np.ndarray
    contacts: tuple[tuple[str, str], ...]
    task_success: bool

    def close(self) -> None:
        self.env.close()


def reconstruct_to_prefix(
    event: dict[str, Any],
    *,
    severity_m: float,
    prefix_k: int,
    bundle: ActorBundle,
) -> BranchContext:
    if not DETECTION_PREFIX <= prefix_k <= int(event["effective_horizon"]):
        raise ValueError(prefix_k)
    env, _ = make_env(event["task"], config_dir=LIBERO_CONFIG, seed=0)
    try:
        history, anchor = reconstruct_anchor(env, event, bundle)
        if not anchor["passed"]:
            raise ValueError(f"anchor reconstruction rejected: {anchor['checks']}")
        spec = TASK_SPECS[event["task"]]
        assert spec.manipulated_joint
        initial = np.asarray(event["initial_manipulated_qpos"], dtype=np.float64)
        tracker = CauseTracker(
            task=event["task"],
            initial_object_z=float(initial[2]),
            previous_gripper=float(history.actions[-1][-1]),
            ever_lifted=bool(event["phase"]["ever_stably_lifted"]),
        )
        chunk = np.asarray(event["original_chunk"], dtype=np.float32)
        perturbation: dict[str, Any] | None = None
        for index, action in enumerate(chunk[:prefix_k]):
            observation, _, _, _ = env.step(action)
            history.update(observation, action)
            tracker.observe_action(env, action)
            if index + 1 == DETECTION_PREFIX:
                perturbation = apply_actor_perturbation(env, event, severity_m)
                tracker.observe_injection(env, tuple(perturbation["before_contacts"]))
                # Injection changes object state without consuming a control
                # step. The actor at k=d must see s_d after that change while
                # retaining the same four-time-step history length.
                history.refresh_latest_observation(current_observation(env))
        if perturbation is None:
            raise RuntimeError("perturbation not applied")
        state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        return BranchContext(
            env=env,
            history=history,
            tracker=tracker,
            perturbation=perturbation,
            anchor_replay=anchor,
            prefix_k=prefix_k,
            state=state,
            contacts=contact_pairs(env),
            task_success=bool(env.check_success()),
        )
    except Exception:
        env.close()
        raise


def branch_snapshot(context: BranchContext, bundle: ActorBundle) -> dict[str, Any]:
    state_history = context.history.state_array()
    action_history = context.history.action_array()
    replanned = bundle.predict(state_history, action_history, context.tracker.task)
    state = np.asarray(context.env.get_sim_state(), dtype=np.float64).copy()
    contacts = contact_pairs(context.env)
    return {
        "state": state,
        "state_hash": state_sha256(state),
        "state_history": state_history,
        "state_history_hash": sha256_array(state_history, np.float32),
        "action_history": action_history,
        "action_history_hash": sha256_array(action_history, np.float32),
        "contacts": contacts,
        "contacts_hash": contact_sha256(contacts),
        "task_success": bool(context.env.check_success()),
        "cause_tracker": context.tracker.as_dict(),
        "cause_tracker_hash": sha256_json(context.tracker.as_dict()),
        "replanned_chunk": replanned,
        "replanned_chunk_hash": chunk_hash(replanned),
    }


def severity_id(task: str, severity_m: float) -> str:
    prefix = "shift" if task == "put_the_cream_cheese_in_the_bowl" else "lateral"
    return f"{prefix}{round(float(severity_m) * 1000):03d}mm"
