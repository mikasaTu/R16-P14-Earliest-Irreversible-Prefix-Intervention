from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from .envs import (
    array_sha256,
    contact_pairs,
    contact_sha256,
    feature_from_obs,
    joint_qpos,
    restore_state,
    set_joint_qpos,
    state_sha256,
)
from .settings import (
    CHUNK_LENGTH,
    INSERTION_PREFIX,
    TASK_SPECS,
    TaskSpec,
    config_id,
)


@dataclass(frozen=True)
class Candidate:
    task: str
    demo_id: int
    anchor_step: int
    parameters: dict[str, float | int]
    release_step: int | None = None
    future_step: int | None = None

    @property
    def config_id(self) -> str:
        return config_id(self.task, self.parameters)

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.task}__demo{self.demo_id:02d}__t{self.anchor_step:04d}__"
            f"{self.config_id}"
        )


@dataclass
class CauseTracker:
    spec: TaskSpec
    initial_object_z: float | None
    previous_gripper: float
    ever_lifted: bool = False
    injected: bool = False
    immediate_violation: bool = False
    violation: bool = False
    violation_type: str | None = None
    first_violation_post_detection_step: int | None = None
    post_detection_steps: int = 0

    def mark(self, violation_type: str, *, immediate: bool = False) -> None:
        if not self.violation:
            self.violation = True
            self.violation_type = violation_type
            self.first_violation_post_detection_step = (
                0 if immediate else self.post_detection_steps
            )
        self.immediate_violation = self.immediate_violation or immediate

    def observe_injection(
        self, env, before_contacts: tuple[tuple[str, str], ...]
    ) -> None:
        self.injected = True
        introduced = set(contact_pairs(env)).difference(before_contacts)
        if any(pair_is_cause(env, self.spec, pair) for pair in introduced):
            self.mark("injection_instant_target_cause_contact", immediate=True)

    def observe_action(self, env, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        if self.injected:
            self.post_detection_steps += 1
        if self.spec.manipulated_joint and self.initial_object_z is not None:
            object_position = joint_qpos(env, self.spec.manipulated_joint)
            if object_position[2] - self.initial_object_z >= self.spec.lift_delta:
                self.ever_lifted = True
        if self.injected and self.spec.family in {
            "drawer_obstacle",
            "swept_path_blocker",
            "target_region_blocker",
        }:
            if cause_contact(env, self.spec):
                self.mark(f"delayed_{self.spec.family}_contact")
        if self.injected and self.spec.family == "target_shift":
            release = self.previous_gripper > 0 and float(action[-1]) < 0
            if release and self.ever_lifted:
                assert self.spec.manipulated_joint and self.spec.target_joint
                object_position = joint_qpos(env, self.spec.manipulated_joint)
                target_position = joint_qpos(env, self.spec.target_joint)
                if (
                    np.linalg.norm(object_position[:2] - target_position[:2])
                    > self.spec.placement_xy_tolerance
                ):
                    self.mark("misaligned_release_after_target_shift")
        self.previous_gripper = float(action[-1])


def load_demo(task_name: str, demo_id: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(TASK_SPECS[task_name].hdf5_path, "r") as dataset:
        demo = dataset[f"data/demo_{demo_id}"]
        return (
            np.asarray(demo["states"], dtype=np.float64),
            np.asarray(demo["actions"], dtype=np.float32),
        )


def joint_trajectory(env, states: np.ndarray, joint_name: str) -> np.ndarray:
    address = env.sim.model.get_joint_qpos_addr(joint_name)
    qpos = states[:, 1 : 1 + env.sim.model.nq]
    if isinstance(address, (int, np.integer)):
        return qpos[:, int(address)]
    return qpos[:, int(address[0]) : int(address[1])]


def release_step(env, spec: TaskSpec, states: np.ndarray, actions: np.ndarray) -> int:
    if not spec.manipulated_joint:
        raise ValueError(f"{spec.name} has no manipulated joint")
    trajectory = joint_trajectory(env, states, spec.manipulated_joint)
    lifted = np.flatnonzero(trajectory[:, 2] - trajectory[0, 2] >= spec.lift_delta)
    if not len(lifted):
        raise ValueError("demonstration has no stable lift")
    releases = np.flatnonzero(
        (np.arange(len(actions)) >= int(lifted[0]))
        & (actions[:, -1] < 0)
        & np.r_[False, actions[:-1, -1] > 0]
    )
    if not len(releases):
        raise ValueError("demonstration has no post-lift release")
    return int(releases[-1])


def first_motion_step(env, states: np.ndarray, joint_name: str, threshold: float) -> int:
    trajectory = np.asarray(joint_trajectory(env, states, joint_name))
    if trajectory.ndim == 2:
        delta = np.linalg.norm(trajectory[:, :3] - trajectory[0, :3], axis=1)
    else:
        delta = np.abs(trajectory - trajectory[0])
    moved = np.flatnonzero(delta > threshold)
    if not len(moved):
        raise ValueError(f"demonstration has no motion for {joint_name}")
    return int(moved[0])


def build_candidate(
    env,
    task_name: str,
    demo_id: int,
    parameters: dict[str, float | int],
) -> Candidate:
    spec = TASK_SPECS[task_name]
    states, actions = load_demo(task_name, demo_id)
    release: int | None = None
    future: int | None = None
    if spec.family in {"target_shift", "target_region_blocker"}:
        release = release_step(env, spec, states, actions)
        lead = int(parameters["release_lead_actions"])
        anchor = release - INSERTION_PREFIX - lead + 1
        future = min(len(states) - 1, release + 2)
    elif spec.family == "drawer_obstacle":
        assert spec.mechanism_joint
        first = first_motion_step(env, states, spec.mechanism_joint, 0.005)
        injection_global = first + int(parameters["insertion_offset"])
        anchor = injection_global - INSERTION_PREFIX
    else:
        assert spec.manipulated_joint
        trajectory = np.asarray(
            joint_trajectory(env, states, spec.manipulated_joint), dtype=np.float64
        )
        displacement = np.linalg.norm(
            trajectory[:, :3] - trajectory[0, :3], axis=1
        )
        total = float(displacement.max())
        if total <= 0.01:
            raise ValueError("demonstration has no stable plate path")
        progress = float(parameters["target_path_progress_fraction"])
        approach = np.flatnonzero(displacement >= progress * total)
        if not len(approach):
            raise ValueError(
                f"demonstration never reaches {progress:.2f} path progress"
            )
        future = int(approach[0])
        first = first_motion_step(env, states, spec.manipulated_joint, 0.005)
        anchor = first - INSERTION_PREFIX
    if anchor < 0 or anchor + CHUNK_LENGTH > len(actions):
        raise ValueError(
            f"invalid anchor={anchor} action_count={len(actions)} for {task_name}"
        )
    return Candidate(
        task=task_name,
        demo_id=demo_id,
        anchor_step=int(anchor),
        parameters=dict(parameters),
        release_step=release,
        future_step=future,
    )


def _object_key(joint_name: str | None) -> str:
    return "" if joint_name is None else joint_name.removesuffix("_joint0")


def _pair_has(pair: tuple[str, str], left: str, right: str) -> bool:
    return any(left in item for item in pair) and any(right in item for item in pair)


def mechanism_geoms(env, spec: TaskSpec) -> set[str]:
    if spec.mechanism_joint is None:
        return set()
    suffix = "cabinet_top" if "top" in spec.mechanism_joint else "cabinet_middle"
    output: set[str] = set()
    for geom_id in range(env.sim.model.ngeom):
        body_id = int(env.sim.model.geom_bodyid[geom_id])
        body_name = env.sim.model.body_id2name(body_id) or ""
        geom_name = env.sim.model.geom_id2name(geom_id)
        if geom_name and suffix in body_name:
            output.add(geom_name)
    return output


def pair_is_cause(env, spec: TaskSpec, pair: tuple[str, str]) -> bool:
    if spec.family == "target_shift":
        return _pair_has(
            pair,
            _object_key(spec.manipulated_joint),
            _object_key(spec.target_joint),
        )
    if spec.family == "drawer_obstacle":
        obstacle = _object_key(spec.obstacle_joint)
        target_geoms = mechanism_geoms(env, spec)
        return bool(
            any(obstacle in item for item in pair)
            and any(item in target_geoms for item in pair)
        )
    return _pair_has(
        pair,
        _object_key(spec.manipulated_joint),
        _object_key(spec.obstacle_joint),
    )


def cause_contact(env, spec: TaskSpec) -> bool:
    return any(pair_is_cause(env, spec, pair) for pair in contact_pairs(env))


def _lateral_unit(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction = np.asarray(end[:2] - start[:2], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    direction /= norm
    return np.asarray([-direction[1], direction[0]], dtype=np.float64)


def apply_perturbation(
    env,
    candidate: Candidate,
    states: np.ndarray,
) -> dict[str, Any]:
    spec = TASK_SPECS[candidate.task]
    manipulated_before = (
        joint_qpos(env, spec.manipulated_joint)
        if spec.manipulated_joint
        else None
    )
    if spec.family == "target_shift":
        assert spec.target_joint and spec.manipulated_joint
        joint = spec.target_joint
        before = joint_qpos(env, joint)
        after = before.copy()
        object_position = joint_qpos(env, spec.manipulated_joint)
        direction = -1.0 if object_position[0] - before[0] >= 0 else 1.0
        after[0] += direction * float(candidate.parameters["shift_magnitude_m"])
        perturbation_type = "target_free_joint_shift"
    elif spec.family == "drawer_obstacle":
        assert spec.obstacle_joint and spec.handle_geom
        joint = spec.obstacle_joint
        before = joint_qpos(env, joint)
        after = before.copy()
        handle = np.asarray(env.sim.data.get_geom_xpos(spec.handle_geom), dtype=np.float64)
        after[:3] = np.asarray(
            [
                handle[0],
                handle[1] + float(candidate.parameters["obstacle_clearance_m"]),
                before[2],
            ]
        )
        perturbation_type = "existing_object_drawer_path_obstacle"
    else:
        assert spec.obstacle_joint and spec.manipulated_joint and candidate.future_step is not None
        joint = spec.obstacle_joint
        before = joint_qpos(env, joint)
        after = before.copy()
        trajectory = joint_trajectory(env, states, spec.manipulated_joint)
        future_position = np.asarray(trajectory[candidate.future_step], dtype=np.float64)
        current_position = joint_qpos(env, spec.manipulated_joint)
        lateral = _lateral_unit(current_position, future_position)
        offset = float(candidate.parameters["blocker_lateral_offset_m"])
        after[:2] = future_position[:2] + lateral * offset
        perturbation_type = (
            "target_region_blocker"
            if spec.family == "target_region_blocker"
            else "future_swept_path_blocker"
        )
    set_joint_qpos(env, joint, after)
    manipulated_after = (
        joint_qpos(env, spec.manipulated_joint)
        if spec.manipulated_joint
        else None
    )
    manipulated_error = (
        float(np.max(np.abs(manipulated_after - manipulated_before)))
        if manipulated_before is not None
        else 0.0
    )
    if manipulated_error != 0.0:
        raise RuntimeError(f"manipulated object changed at injection: {manipulated_error}")
    return {
        "type": perturbation_type,
        "joint": joint,
        "before_qpos": before.tolist(),
        "after_qpos": after.tolist(),
        "manipulated_joint": spec.manipulated_joint,
        "manipulated_qpos_max_error": manipulated_error,
    }


def initial_tracker(env, candidate: Candidate, states: np.ndarray, actions: np.ndarray) -> CauseTracker:
    spec = TASK_SPECS[candidate.task]
    initial_z: float | None = None
    ever_lifted = False
    if spec.manipulated_joint and spec.family != "swept_path_blocker":
        trajectory = joint_trajectory(env, states, spec.manipulated_joint)
        initial_z = float(trajectory[0, 2])
        ever_lifted = bool(
            joint_qpos(env, spec.manipulated_joint)[2] - initial_z >= spec.lift_delta
        )
    previous_gripper = (
        float(actions[candidate.anchor_step - 1, -1])
        if candidate.anchor_step > 0
        else 1.0
    )
    return CauseTracker(
        spec=spec,
        initial_object_z=initial_z,
        previous_gripper=previous_gripper,
        ever_lifted=ever_lifted,
    )


def reconstruct_prefix(
    env,
    candidate: Candidate,
    *,
    prefix_k: int,
) -> tuple[CauseTracker, dict[str, Any]]:
    if not INSERTION_PREFIX <= prefix_k <= CHUNK_LENGTH:
        raise ValueError(prefix_k)
    states, actions = load_demo(candidate.task, candidate.demo_id)
    chunk = np.asarray(
        actions[candidate.anchor_step : candidate.anchor_step + CHUNK_LENGTH],
        dtype=np.float32,
    )
    env.seed(0)
    env.reset()
    obs = restore_state(env, states[candidate.anchor_step])
    anchor_feature_hash = array_sha256(feature_from_obs(obs))
    tracker = initial_tracker(env, candidate, states, actions)
    perturbation: dict[str, Any] | None = None
    phase_contract: dict[str, bool] | None = None
    contacts: list[tuple[tuple[str, str], ...]] = []
    for index, action in enumerate(chunk[:prefix_k]):
        env.step(action)
        tracker.observe_action(env, action)
        if index + 1 == INSERTION_PREFIX:
            before_contacts = contact_pairs(env)
            perturbation = apply_perturbation(env, candidate, states)
            tracker.observe_injection(env, before_contacts)
            release_pending = (
                candidate.release_step is None
                or candidate.anchor_step + index < candidate.release_step
            )
            phase_contract = {
                "environment_only": perturbation["manipulated_qpos_max_error"] == 0.0,
                "no_injection_cause_violation": not tracker.immediate_violation,
                "at_least_8_chunk_actions_remaining": CHUNK_LENGTH - INSERTION_PREFIX >= 8,
                "release_pending_when_applicable": release_pending,
                "task_not_already_successful": not bool(env.check_success()),
                "stable_phase_anchor": (
                    tracker.ever_lifted
                    if tracker.spec.family in {"target_shift", "target_region_blocker"}
                    else True
                ),
            }
        contacts.append(contact_pairs(env))
    if perturbation is None or phase_contract is None:
        raise RuntimeError("perturbation was not applied")
    state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
    return tracker, {
        "candidate_id": candidate.candidate_id,
        "task": candidate.task,
        "demo_id": candidate.demo_id,
        "config_id": candidate.config_id,
        "anchor_step": candidate.anchor_step,
        "prefix_k": prefix_k,
        "chunk": chunk,
        "action_chunk_hash": array_sha256(chunk),
        "anchor_feature_hash": anchor_feature_hash,
        "state": state,
        "state_hash": state_sha256(state),
        "contacts": contacts,
        "contacts_hash": contact_sha256(contacts),
        "outcome": {
            "cause_violation": tracker.violation,
            "violation_type": tracker.violation_type,
            "task_success": bool(env.check_success()),
        },
        "perturbation": perturbation,
        "phase_contract": phase_contract,
        "phase_contract_valid": all(phase_contract.values()),
    }


def nominal_record(env, candidate: Candidate) -> dict[str, Any]:
    tracker, info = reconstruct_prefix(env, candidate, prefix_k=CHUNK_LENGTH)
    return {
        "record_type": "calibration_nominal",
        "candidate_id": candidate.candidate_id,
        "task": candidate.task,
        "demo_id": candidate.demo_id,
        "config_id": candidate.config_id,
        "parameters": candidate.parameters,
        "anchor_step": candidate.anchor_step,
        "immediate_violation": tracker.immediate_violation,
        "cause_violation": tracker.violation,
        "violation_type": tracker.violation_type,
        "first_violation_post_detection_step": tracker.first_violation_post_detection_step,
        "task_success": bool(env.check_success()),
        "safe_success": bool(env.check_success() and not tracker.violation),
        "phase_contract": info["phase_contract"],
        "phase_contract_valid": info["phase_contract_valid"],
        "action_chunk_hash": info["action_chunk_hash"],
        "final_state_hash": info["state_hash"],
        "contact_trace_hash": info["contacts_hash"],
        "perturbation": info["perturbation"],
    }


def replay_equal(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    state_error = float(np.max(np.abs(left["state"] - right["state"])))
    checks = {
        "state_hash_match": left["state_hash"] == right["state_hash"],
        "contacts_match": left["contacts_hash"] == right["contacts_hash"],
        "outcome_match": left["outcome"] == right["outcome"],
        "chunk_hash_match": left["action_chunk_hash"] == right["action_chunk_hash"],
        "max_state_error_le_1e_9": state_error <= 1e-9,
    }
    return {
        **checks,
        "max_state_error": state_error,
        "passed": all(checks.values()),
    }
