from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, field
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
from r16_p14_stage2b.io_utils import sha256_array as raw_array_hash
from r16_p14_stage2b.runtime import ActorBundle, ActorHistory, chunk_hash, normalized_disagreement
from r16_p14_stage2c.goal_geometry import target_distance

from .geometry import lateral_unit, live_bounding_sphere
from .io_utils import sha256_array, sha256_json
from .settings import DETECTION_PREFIX, H_VALID, TAIL_HORIZON, TARGET_SHIFT_TASK


def hold_action(previous: Any) -> np.ndarray:
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[-1] = float(np.asarray(previous, dtype=np.float32)[-1])
    return action


@dataclass
class CauseTracker:
    task: str
    initial_object_z: float
    previous_gripper: float
    ever_stably_lifted: bool
    injected: bool = False
    injection_contact: bool = False
    violation: bool = False
    immediate_violation: bool = False
    violation_type: str | None = None
    first_violation_offset: int | None = None
    post_detection_steps: int = 0
    shifted_target_qpos: list[float] | None = None
    gripper_transition_indices: list[int] = field(default_factory=list)
    contact_event_indices: list[int] = field(default_factory=list)

    def mark(self, kind: str, immediate: bool = False) -> None:
        if not self.violation:
            self.violation = True
            self.violation_type = kind
            self.first_violation_offset = 0 if immediate else self.post_detection_steps
        self.immediate_violation |= bool(immediate)

    def observe_injection(self, env, before: tuple[tuple[str, str], ...]) -> None:
        self.injected = True
        if self.task != TARGET_SHIFT_TASK:
            after = set(contact_pairs(env))
            introduced = after.difference(before)
            if introduced and cause_contact(env, TASK_SPECS[self.task]):
                self.injection_contact = True
                self.mark("bowl_blocker_contact_at_injection", immediate=True)

    def observe_action(self, env, action: np.ndarray) -> None:
        spec = TASK_SPECS[self.task]
        assert spec.manipulated_joint
        if self.injected:
            self.post_detection_steps += 1
        qpos = joint_qpos(env, spec.manipulated_joint)
        if float(qpos[2] - self.initial_object_z) >= float(spec.lift_delta):
            self.ever_stably_lifted = True
        gripper = float(np.asarray(action, dtype=np.float32)[-1])
        release = self.previous_gripper > 0.0 and gripper < 0.0
        if release:
            self.gripper_transition_indices.append(self.post_detection_steps)
        if self.injected and self.task == TARGET_SHIFT_TASK and release and self.ever_stably_lifted:
            if self.shifted_target_qpos is None:
                raise RuntimeError("target-shift tracker is missing shifted target pose")
            target = np.asarray(self.shifted_target_qpos, dtype=np.float64)
            if float(np.linalg.norm(qpos[:2] - target[:2])) > float(spec.placement_xy_tolerance):
                self.mark("release_outside_shifted_target_region")
        elif self.injected and self.task != TARGET_SHIFT_TASK and cause_contact(env, spec):
            self.contact_event_indices.append(self.post_detection_steps)
            self.mark("bowl_registered_blocker_contact")
        self.previous_gripper = gripper

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def validate_event(event: dict[str, Any], bundle: ActorBundle) -> None:
    checks = {
        "actor_generated": bool(event.get("source_is_actor_generated_chunk")),
        "not_demonstration": not bool(event.get("source_is_demonstration_chunk")),
        "no_fallback": not bool(event.get("global_step_fallback_used")),
        "checkpoint": event["checkpoint_sha256"] == bundle.checkpoint_sha256,
        "actor_seed": int(event["actor_seed"]) == int(bundle.seed),
        "init_state": sha256_array(event["init_state"], np.float64) == event["init_state_hash"],
        "pre_anchor_actions": sha256_array(event["pre_anchor_actions"], np.float32)
        == event["pre_anchor_actions_hash"],
        "anchor_state": state_sha256(np.asarray(event["anchor_state"], dtype=np.float64))
        == event["anchor_state_hash"],
        "state_history": sha256_array(event["state_history"], np.float32)
        == event["state_history_hash"],
        "action_history": sha256_array(event["action_history"], np.float32)
        == event["action_history_hash"],
        "chunk": chunk_hash(event["original_chunk"]) == event["original_chunk_hash"],
    }
    if not all(checks.values()):
        raise ValueError(f"event contract rejected: {[key for key, value in checks.items() if not value]}")


def robot_state(env) -> tuple[np.ndarray, np.ndarray]:
    qpos, qvel = [], []
    for robot in env.robots:
        qpos.extend(np.asarray(env.sim.data.qpos[robot._ref_joint_pos_indexes], dtype=np.float64))
        qvel.extend(np.asarray(env.sim.data.qvel[robot._ref_joint_vel_indexes], dtype=np.float64))
    return np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)


def phase_state(env, event: dict[str, Any], tracker: CauseTracker) -> dict[str, Any]:
    spec = TASK_SPECS[event["task"]]
    assert spec.manipulated_joint
    manipulated = joint_qpos(env, spec.manipulated_joint)
    return {
        "ever_stably_lifted": tracker.ever_stably_lifted,
        "object_height_delta_m": float(manipulated[2] - tracker.initial_object_z),
        "task_success": bool(env.check_success()),
        "post_detection_steps": tracker.post_detection_steps,
        "gripper_closed": tracker.previous_gripper > 0.0,
    }


def branch_signature(
    env,
    history: ActorHistory,
    event: dict[str, Any],
    tracker: CauseTracker,
    executed_prefix: list[np.ndarray],
) -> dict[str, Any]:
    spec = TASK_SPECS[event["task"]]
    assert spec.manipulated_joint
    state = np.ascontiguousarray(env.get_sim_state(), dtype=np.float64)
    feature = np.ascontiguousarray(feature_from_obs(current_observation(env)), dtype=np.float32)
    states = np.ascontiguousarray(history.state_array(), dtype=np.float32)
    actions = np.ascontiguousarray(history.action_array(), dtype=np.float32)
    robot_qpos, robot_qvel = robot_state(env)
    contacts = list(contact_pairs(env))
    manipulated = joint_qpos(env, spec.manipulated_joint)
    target = joint_qpos(env, spec.target_joint) if spec.target_joint else None
    obstacle = joint_qpos(env, spec.obstacle_joint) if spec.obstacle_joint else None
    phase = phase_state(env, event, tracker)
    tracker_state = tracker.as_dict()
    prefix_array = np.asarray(executed_prefix, dtype=np.float32).reshape(-1, ACTION_DIM)
    signature = {
        "simulator_state_hash": state_sha256(state),
        "observation_feature_hash": raw_array_hash(feature, np.float32),
        "four_state_history_hash": raw_array_hash(states, np.float32),
        "three_action_history_hash": raw_array_hash(actions, np.float32),
        "robot_qpos_hash": raw_array_hash(robot_qpos, np.float64),
        "robot_qvel_hash": raw_array_hash(robot_qvel, np.float64),
        "contact_pairs": contacts,
        "contact_pairs_hash": sha256_json(contacts),
        "manipulated_qpos_hash": raw_array_hash(manipulated, np.float64),
        "target_qpos_hash": None if target is None else raw_array_hash(target, np.float64),
        "obstacle_qpos_hash": None if obstacle is None else raw_array_hash(obstacle, np.float64),
        "gripper_state": float(tracker.previous_gripper),
        "task_phase": phase,
        "task_phase_hash": sha256_json(phase),
        "cause_tracker": tracker_state,
        "cause_tracker_hash": sha256_json(tracker_state),
        "executed_prefix_action_hash": raw_array_hash(prefix_array, np.float32),
        "executed_prefix_length": int(len(prefix_array)),
    }
    signature["complete_signature_hash"] = sha256_json(signature)
    return signature


def reconstruct_anchor(event: dict[str, Any], bundle: ActorBundle):
    validate_event(event, bundle)
    env, _ = make_env(event["task"], seed=0)
    try:
        observation = restore_state(env, np.asarray(event["init_state"], dtype=np.float64))
        history = ActorHistory.initial(observation)
        for action in np.asarray(event["pre_anchor_actions"], dtype=np.float32).reshape(-1, ACTION_DIM):
            observation, _, _, _ = env.step(action)
            history.update(observation, action)
        state = np.ascontiguousarray(env.get_sim_state(), dtype=np.float64)
        states = np.ascontiguousarray(history.state_array(), dtype=np.float32)
        actions = np.ascontiguousarray(history.action_array(), dtype=np.float32)
        before = (
            state_sha256(state),
            raw_array_hash(states, np.float32),
            raw_array_hash(actions, np.float32),
        )
        chunk = np.ascontiguousarray(bundle.predict(states, actions, event["task"]), dtype=np.float32)
        after = (
            state_sha256(np.asarray(env.get_sim_state(), dtype=np.float64)),
            raw_array_hash(history.state_array(), np.float32),
            raw_array_hash(history.action_array(), np.float32),
        )
        checks = {
            "anchor_state_exact": state_sha256(state) == event["anchor_state_hash"],
            "state_history_exact": sha256_array(states, np.float32) == event["state_history_hash"],
            "action_history_exact": sha256_array(actions, np.float32) == event["action_history_hash"],
            "event_chunk_exact": chunk_hash(chunk) == event["original_chunk_hash"],
            "actor_inference_side_effect_free": before == after,
        }
        if not all(checks.values()):
            raise ValueError(f"fresh anchor reconstruction failed: {checks}")
        checks["max_anchor_state_error"] = float(
            np.max(np.abs(state - np.asarray(event["anchor_state"], dtype=np.float64)))
        )
        return env, history, checks
    except BaseException:
        env.close()
        raise


def apply_perturbation(env, event: dict[str, Any], parameter: dict[str, Any]) -> dict[str, Any]:
    spec = TASK_SPECS[event["task"]]
    assert spec.manipulated_joint
    manipulated_before = joint_qpos(env, spec.manipulated_joint)
    contacts_before = contact_pairs(env)
    if event["task"] == TARGET_SHIFT_TASK:
        assert spec.target_joint
        before = joint_qpos(env, spec.target_joint)
        approach = before[:2] - manipulated_before[:2]
        lateral = lateral_unit(np.zeros(2), approach)
        after = before.copy()
        after[:2] += lateral * float(parameter["magnitude_m"])
        moved_joint = spec.target_joint
        kind = "lateral_target_shift"
        geometry = None
    else:
        assert spec.obstacle_joint
        before = joint_qpos(env, spec.obstacle_joint)
        positions = np.asarray(event["nominal_object_trajectory"]["positions"], dtype=np.float64)
        future_index = int(parameter["future_index"])
        future = positions[future_index]
        prior = positions[max(DETECTION_PREFIX, future_index - 1)]
        lateral = lateral_unit(prior, future)
        object_geometry = live_bounding_sphere(env, spec.manipulated_joint)
        obstacle_geometry = live_bounding_sphere(env, spec.obstacle_joint)
        clearance = (
            float(object_geometry["radius_m"])
            + float(obstacle_geometry["radius_m"])
            + float(parameter["clearance_delta_m"])
        )
        after = before.copy()
        after[:2] = future[:2] + lateral * clearance
        moved_joint = spec.obstacle_joint
        kind = "future_path_live_geometry_blocker"
        geometry = {
            "manipulated": object_geometry,
            "obstacle": obstacle_geometry,
            "center_clearance_m": clearance,
            "future_point": future.tolist(),
            "path_tangent_source": [prior.tolist(), future.tolist()],
        }
    set_joint_qpos(env, moved_joint, after)
    manipulated_after = joint_qpos(env, spec.manipulated_joint)
    teleport_error = float(np.max(np.abs(manipulated_after - manipulated_before)))
    if teleport_error != 0.0:
        raise RuntimeError(f"manipulated object was teleported: {teleport_error}")
    return {
        "kind": kind,
        "parameter": parameter,
        "moved_joint": moved_joint,
        "before_qpos": before.tolist(),
        "after_qpos": after.tolist(),
        "lateral_unit": lateral.tolist(),
        "geometry": geometry,
        "contacts_before": list(contacts_before),
        "contacts_after": list(contact_pairs(env)),
        "manipulated_qpos_max_error": teleport_error,
    }


def actor_call(
    bundle: ActorBundle,
    env,
    history: ActorHistory,
    task: str,
    timings: list[float],
    side_effect_checks: list[bool],
) -> np.ndarray:
    before_state = state_sha256(np.asarray(env.get_sim_state(), dtype=np.float64))
    before_states = raw_array_hash(history.state_array(), np.float32)
    before_actions = raw_array_hash(history.action_array(), np.float32)
    start = time.perf_counter()
    chunk = np.ascontiguousarray(
        bundle.predict(history.state_array(), history.action_array(), task), dtype=np.float32
    )
    timings.append(time.perf_counter() - start)
    after = (
        state_sha256(np.asarray(env.get_sim_state(), dtype=np.float64)),
        raw_array_hash(history.state_array(), np.float32),
        raw_array_hash(history.action_array(), np.float32),
    )
    passed = after == (before_state, before_states, before_actions)
    side_effect_checks.append(passed)
    if not passed:
        raise RuntimeError("actor inference mutated environment or history")
    return chunk


def metric_state(env, task: str) -> dict[str, Any]:
    observation = current_observation(env)
    spec = TASK_SPECS[task]
    assert spec.manipulated_joint
    return {
        "last_eef": np.asarray(observation["robot0_eef_pos"], dtype=np.float64),
        "last_object": joint_qpos(env, spec.manipulated_joint),
        "eef_path_length_m": 0.0,
        "object_path_length_m": 0.0,
        "contact_count": 0,
        "post_detection_actions": 0,
        "first_success_step": None,
    }


def step_action(env, history: ActorHistory, tracker: CauseTracker, metrics: dict[str, Any], action: np.ndarray) -> None:
    observation, _, _, _ = env.step(np.asarray(action, dtype=np.float32))
    history.update(observation, action)
    tracker.observe_action(env, np.asarray(action, dtype=np.float32))
    eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
    manipulated = joint_qpos(env, TASK_SPECS[tracker.task].manipulated_joint)
    metrics["eef_path_length_m"] += float(np.linalg.norm(eef - metrics["last_eef"]))
    metrics["object_path_length_m"] += float(
        np.linalg.norm(manipulated[:3] - metrics["last_object"][:3])
    )
    metrics["last_eef"] = eef
    metrics["last_object"] = manipulated
    metrics["contact_count"] += len(contact_pairs(env))
    if tracker.injected:
        metrics["post_detection_actions"] += 1
        if metrics["first_success_step"] is None and env.check_success():
            metrics["first_success_step"] = metrics["post_detection_actions"]


def arm_plan(arm: str, prefix_k: int, event: dict[str, Any]) -> tuple[str, int]:
    if arm == "IMMEDIATE_FRESH":
        return "IMMEDIATE_FRESH", DETECTION_PREFIX
    if arm.startswith("FIXED_DELAY_"):
        delay = int(arm.rsplit("_", 1)[1])
        return "CACHED_MATCHED", min(H_VALID, DETECTION_PREFIX + delay)
    mapping = {
        "EVENT_ALIGNED_CACHED": "CACHED_MATCHED",
        "FRESH_MATCHED_AT_RULE_K": "FRESH_MATCHED",
        "HOLD_MATCHED_AT_RULE_K": "HOLD_MATCHED",
        "CACHED_NOQUERY_AT_RULE_K": "CACHED_NOQUERY",
    }
    return mapping.get(arm, arm), prefix_k


def execute_branch(
    event: dict[str, Any],
    parameter: dict[str, Any],
    prefix_k: int,
    arm: str,
    repeat: int,
    device: str = "cpu",
) -> dict[str, Any]:
    branch_start = time.perf_counter()
    bundle = ActorBundle.load(int(event["actor_seed"]), device)
    env, history, reconstruction = reconstruct_anchor(event, bundle)
    try:
        requested_prefix_k = int(prefix_k)
        mode, prefix_k = arm_plan(arm, prefix_k, event)
        if not DETECTION_PREFIX <= prefix_k <= H_VALID:
            raise ValueError(prefix_k)
        old = np.asarray(event["original_chunk"], dtype=np.float32)
        tracker = CauseTracker(
            task=event["task"],
            initial_object_z=float(event["initial_manipulated_qpos"][2]),
            previous_gripper=float(history.actions[-1][-1]),
            ever_stably_lifted=bool(event["task_phase"]["stable_lift_two_steps"]),
        )
        metrics = metric_state(env, event["task"])
        for action in old[:DETECTION_PREFIX]:
            step_action(env, history, tracker, metrics, action)
        perturbation = apply_perturbation(env, event, parameter)
        if event["task"] == TARGET_SHIFT_TASK:
            tracker.shifted_target_qpos = perturbation["after_qpos"]
        tracker.observe_injection(
            env, tuple(tuple(item) for item in perturbation["contacts_before"])
        )
        history.refresh_latest_observation(current_observation(env))
        detection_signature = branch_signature(env, history, event, tracker, [])
        detection_distance = target_distance(env, event["task"])

        inference_times: list[float] = []
        side_effect_checks: list[bool] = []
        actor_calls = 0
        fresh_at_d: np.ndarray | None = None
        if mode in {"IMMEDIATE_FRESH", "CACHED_MATCHED", "FRESH_MATCHED", "HOLD_MATCHED"}:
            fresh_at_d = actor_call(
                bundle, env, history, event["task"], inference_times, side_effect_checks
            )
            actor_calls += 1

        prefix_length = prefix_k - DETECTION_PREFIX
        if mode in {"CACHED_MATCHED", "CACHED_NOQUERY"}:
            prefix = old[DETECTION_PREFIX:prefix_k]
            cached_retained = prefix_length
        elif mode == "FRESH_MATCHED":
            assert fresh_at_d is not None
            prefix = fresh_at_d[:prefix_length]
            cached_retained = 0
        elif mode == "HOLD_MATCHED":
            prefix = np.asarray(
                [hold_action(old[DETECTION_PREFIX - 1]) for _ in range(prefix_length)],
                dtype=np.float32,
            ).reshape(-1, ACTION_DIM)
            cached_retained = 0
        elif mode == "IMMEDIATE_FRESH":
            prefix = np.empty((0, ACTION_DIM), dtype=np.float32)
            cached_retained = 0
        elif mode == "FULL_OLD_CHUNK":
            prefix_k = H_VALID
            prefix_length = H_VALID - DETECTION_PREFIX
            prefix = old[DETECTION_PREFIX:H_VALID]
            cached_retained = prefix_length
        elif mode == "RECONSTRUCT_ONLY":
            prefix = np.empty((0, ACTION_DIM), dtype=np.float32)
            cached_retained = 0
        else:
            raise ValueError(f"unknown arm: {arm}")

        executed_prefix: list[np.ndarray] = []
        for action in prefix:
            step_action(env, history, tracker, metrics, action)
            executed_prefix.append(np.asarray(action, dtype=np.float32).copy())
        s_obs_k = not tracker.violation
        progress_at_k = float(detection_distance - target_distance(env, event["task"]))
        pre_tail_signature = branch_signature(env, history, event, tracker, executed_prefix)

        disagreement = None
        if fresh_at_d is not None and prefix_length:
            disagreement = float(
                np.mean(
                    [
                        normalized_disagreement(old[DETECTION_PREFIX + index], fresh_at_d[index], bundle)
                        for index in range(prefix_length)
                    ]
                )
            )

        if mode == "IMMEDIATE_FRESH":
            tail = fresh_at_d
        elif mode not in {"FULL_OLD_CHUNK", "RECONSTRUCT_ONLY"} and not env.check_success():
            tail = actor_call(
                bundle, env, history, event["task"], inference_times, side_effect_checks
            )
            actor_calls += 1
        else:
            tail = None
        if tail is not None and mode not in {"FULL_OLD_CHUNK", "RECONSTRUCT_ONLY"}:
            for action in tail[:TAIL_HORIZON]:
                if env.check_success():
                    break
                step_action(env, history, tracker, metrics, action)

        success = bool(env.check_success())
        end_distance = target_distance(env, event["task"])
        final_signature = branch_signature(env, history, event, tracker, executed_prefix)
        maximum_action_budget = H_VALID - DETECTION_PREFIX + TAIL_HORIZON
        maximum_call_budget = 2
        result = {
            "schema_version": 1,
            "event_id": event["event_id"],
            "task": event["task"],
            "split": event["split"],
            "init_state_id": int(event["init_state_id"]),
            "actor_seed": int(event["actor_seed"]),
            "parameter": parameter,
            "parameter_id": parameter["parameter_id"],
            "requested_arm": arm,
            "effective_arm": mode,
            "requested_prefix_k": requested_prefix_k,
            "prefix_k": int(prefix_k),
            "repeat": int(repeat),
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "process_start_method": "spawn",
            "fresh_environment_created": True,
            "reconstruction": reconstruction,
            "detection_signature": detection_signature,
            "pre_tail_signature": pre_tail_signature,
            "final_signature": final_signature,
            "perturbation": perturbation,
            "S_obs_at_k": bool(s_obs_k),
            "cause_violation": bool(tracker.violation),
            "cause_violation_type": tracker.violation_type,
            "first_violation_offset": tracker.first_violation_offset,
            "immediate_violation": tracker.immediate_violation,
            "injection_contact": tracker.injection_contact,
            "task_success": success,
            "safe_success": bool(success and not tracker.violation),
            "actual_post_detection_actions": int(metrics["post_detection_actions"]),
            "completion_steps": metrics["first_success_step"],
            "actual_actor_calls": actor_calls,
            "actual_inference_wall_time_s": float(sum(inference_times)),
            "individual_inference_wall_times_s": inference_times,
            "total_branch_wall_time_s": float(time.perf_counter() - branch_start),
            "eef_path_length_m": float(metrics["eef_path_length_m"]),
            "manipulated_object_path_length_m": float(metrics["object_path_length_m"]),
            "contact_count": int(metrics["contact_count"]),
            "cached_actions_retained": int(cached_retained),
            "cached_fresh_action_disagreement": disagreement,
            "gripper_event_indices": tracker.gripper_transition_indices,
            "contact_event_indices": tracker.contact_event_indices,
            "task_progress_at_k": progress_at_k,
            "progress_retained_end_m": float(detection_distance - end_distance),
            "progress_regression_m": float(max(0.0, end_distance - detection_distance)),
            "executed_prefix_length": int(len(executed_prefix)),
            "executed_prefix_action_hash": pre_tail_signature["executed_prefix_action_hash"],
            "tail_horizon": TAIL_HORIZON,
            "maximum_action_budget": maximum_action_budget,
            "maximum_policy_call_budget": maximum_call_budget,
            "actor_inference_side_effect_free": all(side_effect_checks),
            "tracker": tracker.as_dict(),
            "error": None,
        }
        return result
    finally:
        env.close()
