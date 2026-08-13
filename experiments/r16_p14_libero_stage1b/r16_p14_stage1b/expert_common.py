from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from r16_p14_stage1.envs import contact_pairs, restore_state, state_sha256


CHUNK_LENGTH = 16
INSERTION_PREFIX = 2
BRANCH_BUDGET = 48
DATASET_ROOT = Path("/mnt/cpfs/zbl-cpfs-new/dataset/leon/libero/libero_goal")


@dataclass(frozen=True)
class ExpertTaskSpec:
    task_id: int
    name: str
    hdf5_name: str
    family: str
    cause: str
    object_joint: str | None = None
    target_joint: str | None = None
    lift_delta: float = 0.03
    alignment_xy: float = 0.06
    placement_z_offset: float = 0.035
    horizon: int = 360

    @property
    def hdf5_path(self) -> Path:
        return DATASET_ROOT / self.hdf5_name


EXPERT_TASKS: dict[str, ExpertTaskSpec] = {
    "open_the_middle_drawer_of_the_cabinet": ExpertTaskSpec(
        task_id=0,
        name="open_the_middle_drawer_of_the_cabinet",
        hdf5_name="open_the_middle_drawer_of_the_cabinet_demo.hdf5",
        family="drawer_obstacle",
        cause="fixture_obstacle_contact",
        horizon=360,
    ),
    "put_the_bowl_on_the_plate": ExpertTaskSpec(
        task_id=8,
        name="put_the_bowl_on_the_plate",
        hdf5_name="put_the_bowl_on_the_plate_demo.hdf5",
        family="target_shift",
        cause="premature_release_after_target_shift",
        object_joint="akita_black_bowl_1_joint0",
        target_joint="plate_1_joint0",
        lift_delta=0.03,
        alignment_xy=0.075,
        placement_z_offset=0.035,
        horizon=320,
    ),
    "put_the_cream_cheese_in_the_bowl": ExpertTaskSpec(
        task_id=6,
        name="put_the_cream_cheese_in_the_bowl",
        hdf5_name="put_the_cream_cheese_in_the_bowl_demo.hdf5",
        family="target_shift",
        cause="misaligned_release_after_target_shift",
        object_joint="cream_cheese_1_joint0",
        target_joint="akita_black_bowl_1_joint0",
        lift_delta=0.025,
        alignment_xy=0.055,
        placement_z_offset=0.04,
        horizon=360,
    ),
}


@dataclass(frozen=True)
class ExpertCandidate:
    task: str
    demo_id: int
    anchor_step: int
    release_step: int | None
    insertion_prefix: int
    config_id: str
    parameters: dict[str, float | int]

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.task}__demo{self.demo_id:02d}__t{self.anchor_step:04d}"
            f"__{self.config_id}"
        )


@dataclass
class CauseTracker:
    spec: ExpertTaskSpec
    object_initial_z: float | None = None
    ever_lifted: bool = False
    perturbation_applied: bool = False
    immediate_violation: bool = False
    violation: bool = False
    violation_type: str | None = None
    first_violation_post_detection_step: int | None = None
    post_detection_steps: int = 0
    previous_gripper: float = 1.0
    contact_trace: list[tuple[tuple[str, str], ...]] = field(default_factory=list)

    def copy(self) -> "CauseTracker":
        return CauseTracker(
            spec=self.spec,
            object_initial_z=self.object_initial_z,
            ever_lifted=self.ever_lifted,
            perturbation_applied=self.perturbation_applied,
            immediate_violation=self.immediate_violation,
            violation=self.violation,
            violation_type=self.violation_type,
            first_violation_post_detection_step=self.first_violation_post_detection_step,
            post_detection_steps=self.post_detection_steps,
            previous_gripper=self.previous_gripper,
            contact_trace=list(self.contact_trace),
        )

    def _set_violation(self, kind: str, *, immediate: bool = False) -> None:
        if not self.violation:
            self.violation = True
            self.violation_type = kind
            self.first_violation_post_detection_step = (
                0 if immediate else self.post_detection_steps
            )
        if immediate:
            self.immediate_violation = True

    def observe_injection(self, env, before_contacts: tuple[tuple[str, str], ...]) -> None:
        self.perturbation_applied = True
        after_contacts = contact_pairs(env)
        self.contact_trace.append(after_contacts)
        if self.spec.family == "drawer_obstacle" and _drawer_obstacle_contact(after_contacts):
            self._set_violation("fixture_obstacle_contact", immediate=True)
            return
        if self.spec.family == "target_shift":
            introduced = set(after_contacts).difference(before_contacts)
            object_key = _joint_object_key(self.spec.object_joint)
            target_key = _joint_object_key(self.spec.target_joint)
            if any(
                any(object_key in geom for geom in pair)
                and any(target_key in geom for geom in pair)
                for pair in introduced
            ):
                self._set_violation("target_shift_instant_contact", immediate=True)

    def observe_action(self, env, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        contacts = contact_pairs(env)
        self.contact_trace.append(contacts)
        if self.perturbation_applied:
            self.post_detection_steps += 1

        if self.spec.family == "drawer_obstacle":
            if self.perturbation_applied and _drawer_obstacle_contact(contacts):
                self._set_violation("fixture_obstacle_contact")
            self.previous_gripper = float(action[-1])
            return

        assert self.spec.object_joint is not None
        assert self.spec.target_joint is not None
        object_qpos = _joint_qpos(env, self.spec.object_joint)
        target_qpos = _joint_qpos(env, self.spec.target_joint)
        if self.object_initial_z is not None:
            if object_qpos[2] - self.object_initial_z >= self.spec.lift_delta:
                self.ever_lifted = True
            if (
                self.perturbation_applied
                and self.ever_lifted
                and object_qpos[2] < self.object_initial_z - 0.025
            ):
                self._set_violation("dropped_object")
        release_transition = self.previous_gripper > 0 and float(action[-1]) < 0
        if self.perturbation_applied and self.ever_lifted and release_transition:
            aligned = bool(
                np.linalg.norm(object_qpos[:2] - target_qpos[:2])
                <= self.spec.alignment_xy
            )
            if not aligned:
                self._set_violation(self.spec.cause)
        self.previous_gripper = float(action[-1])


def _joint_object_key(joint_name: str | None) -> str:
    if not joint_name:
        return ""
    return joint_name.removesuffix("_joint0")


def _drawer_obstacle_contact(pairs: Iterable[tuple[str, str]]) -> bool:
    for pair in pairs:
        has_obstacle = any("wine_bottle_1" in name for name in pair)
        has_mechanism = any("wooden_cabinet_1_g" in name for name in pair)
        if has_obstacle and has_mechanism:
            return True
    return False


def _joint_qpos(env, joint_name: str) -> np.ndarray:
    return np.asarray(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()


def configure_libero(config_dir: Path) -> None:
    import os

    config_dir = config_dir.resolve()
    if not (config_dir / "config.yaml").is_file():
        raise FileNotFoundError(config_dir / "config.yaml")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)


def make_expert_env(task_name: str, config_dir: Path, *, seed: int = 0):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    configure_libero(config_dir)
    spec = EXPERT_TASKS[task_name]
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(spec.task_id)
    if task.name != spec.name:
        raise RuntimeError(f"LIBERO task order mismatch: {task.name} != {spec.name}")
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = ControlEnv(
        bddl_file_name=str(bddl_file),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        horizon=spec.horizon,
        ignore_done=True,
        hard_reset=True,
    )
    env.seed(seed)
    env.reset()
    return env


def load_demo(task_name: str, demo_id: int) -> tuple[np.ndarray, np.ndarray]:
    spec = EXPERT_TASKS[task_name]
    with h5py.File(spec.hdf5_path, "r") as dataset:
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


def release_step(env, task_name: str, states: np.ndarray, actions: np.ndarray) -> int:
    spec = EXPERT_TASKS[task_name]
    if spec.object_joint is None:
        raise ValueError(f"{task_name} has no release event")
    trajectory = joint_trajectory(env, states, spec.object_joint)
    lifted = np.flatnonzero(trajectory[:, 2] - trajectory[0, 2] >= spec.lift_delta)
    if not len(lifted):
        raise ValueError(f"demo has no stable lift for {task_name}")
    releases = np.flatnonzero(
        (np.arange(len(actions)) >= int(lifted[0]))
        & (actions[:, -1] < 0)
        & np.r_[False, actions[:-1, -1] > 0]
    )
    if not len(releases):
        raise ValueError(f"demo has no post-lift release for {task_name}")
    return int(releases[-1])


def first_drawer_motion_step(env, states: np.ndarray) -> int:
    drawer = np.asarray(
        joint_trajectory(env, states, "wooden_cabinet_1_middle_level")
    ).reshape(-1)
    moved = np.flatnonzero(np.abs(drawer - drawer[0]) > 0.005)
    if not len(moved):
        raise ValueError("demo has no middle-drawer motion")
    return int(moved[0])


def target_shift_grid() -> list[dict[str, float | int]]:
    return [
        {
            "release_lead_actions": lead,
            "target_shift_x_magnitude_meters": magnitude,
        }
        for lead in (8, 10, 12, 14)
        for magnitude in (0.04, 0.06, 0.08, 0.10)
    ]


def drawer_grid() -> list[dict[str, float | int]]:
    return [
        {
            "insertion_offset_from_first_drawer_motion": offset,
            "obstacle_clearance_meters": clearance,
        }
        for offset in (-2, 0, 2)
        for clearance in (0.035, 0.05, 0.065, 0.08)
    ]


def config_id(task_name: str, parameters: dict[str, float | int]) -> str:
    if EXPERT_TASKS[task_name].family == "target_shift":
        return (
            f"lead{int(parameters['release_lead_actions']):02d}"
            f"_shift{int(round(float(parameters['target_shift_x_magnitude_meters']) * 1000)):03d}mm"
        )
    return (
        f"offset{int(parameters['insertion_offset_from_first_drawer_motion']):+03d}"
        f"_clear{int(round(float(parameters['obstacle_clearance_meters']) * 1000)):03d}mm"
    )


def build_candidate(
    env,
    task_name: str,
    demo_id: int,
    parameters: dict[str, float | int],
) -> ExpertCandidate:
    states, actions = load_demo(task_name, demo_id)
    spec = EXPERT_TASKS[task_name]
    release: int | None = None
    if spec.family == "target_shift":
        release = release_step(env, task_name, states, actions)
        lead = int(parameters["release_lead_actions"])
        anchor = release - INSERTION_PREFIX - lead + 1
    else:
        first_motion = first_drawer_motion_step(env, states)
        injection_global = first_motion + int(
            parameters["insertion_offset_from_first_drawer_motion"]
        )
        anchor = injection_global - INSERTION_PREFIX
    if anchor < 0 or anchor + CHUNK_LENGTH > len(actions):
        raise ValueError(
            f"invalid chunk anchor task={task_name} demo={demo_id} anchor={anchor} len={len(actions)}"
        )
    return ExpertCandidate(
        task=task_name,
        demo_id=demo_id,
        anchor_step=int(anchor),
        release_step=release,
        insertion_prefix=INSERTION_PREFIX,
        config_id=config_id(task_name, parameters),
        parameters=dict(parameters),
    )


def initial_tracker(env, candidate: ExpertCandidate, states: np.ndarray, actions: np.ndarray) -> CauseTracker:
    spec = EXPERT_TASKS[candidate.task]
    initial_z: float | None = None
    ever_lifted = False
    if spec.object_joint is not None:
        trajectory = joint_trajectory(env, states, spec.object_joint)
        initial_z = float(trajectory[0, 2])
        current_z = float(_joint_qpos(env, spec.object_joint)[2])
        ever_lifted = current_z - initial_z >= spec.lift_delta
    previous_gripper = (
        float(actions[candidate.anchor_step - 1, -1])
        if candidate.anchor_step > 0
        else 1.0
    )
    return CauseTracker(
        spec=spec,
        object_initial_z=initial_z,
        ever_lifted=ever_lifted,
        previous_gripper=previous_gripper,
    )


def apply_expert_perturbation(env, candidate: ExpertCandidate) -> dict[str, Any]:
    spec = EXPERT_TASKS[candidate.task]
    if spec.family == "target_shift":
        assert spec.object_joint is not None and spec.target_joint is not None
        before = _joint_qpos(env, spec.target_joint)
        object_qpos = _joint_qpos(env, spec.object_joint)
        after = before.copy()
        magnitude = float(candidate.parameters["target_shift_x_magnitude_meters"])
        relative_x = float(object_qpos[0] - before[0])
        direction = -1.0 if relative_x >= 0 else 1.0
        signed_delta = direction * magnitude
        after[0] += signed_delta
        env.sim.data.set_joint_qpos(spec.target_joint, after)
        env.sim.forward()
        env._post_process()
        env._update_observables(force=True)
        return {
            "type": "target_free_joint_shift",
            "joint": spec.target_joint,
            "axis": 0,
            "magnitude_m": magnitude,
            "signed_delta_m": signed_delta,
            "direction_rule": "target_x_away_from_held_object_x",
            "before_qpos": before.tolist(),
            "after_qpos": after.tolist(),
        }

    before = _joint_qpos(env, "wine_bottle_1_joint0")
    handle = np.asarray(
        env.sim.data.get_geom_xpos("wooden_cabinet_1_g29"), dtype=np.float64
    )
    after = before.copy()
    after[:3] = np.asarray(
        [
            handle[0],
            handle[1] + float(candidate.parameters["obstacle_clearance_meters"]),
            before[2],
        ],
        dtype=np.float64,
    )
    env.sim.data.set_joint_qpos("wine_bottle_1_joint0", after)
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return {
        "type": "existing_obstacle_placement",
        "joint": "wine_bottle_1_joint0",
        "clearance_m": float(candidate.parameters["obstacle_clearance_meters"]),
        "before_qpos": before.tolist(),
        "after_qpos": after.tolist(),
    }


def reconstruct_prefix(
    env,
    candidate: ExpertCandidate,
    *,
    prefix_k: int,
) -> tuple[CauseTracker, dict[str, Any]]:
    if not candidate.insertion_prefix <= prefix_k <= CHUNK_LENGTH:
        raise ValueError(prefix_k)
    states, actions = load_demo(candidate.task, candidate.demo_id)
    chunk = np.asarray(
        actions[candidate.anchor_step : candidate.anchor_step + CHUNK_LENGTH],
        dtype=np.float32,
    )
    # Recovery branches mutate controller and robot history that is absent from
    # MuJoCo's flattened state. A hard reset is therefore part of every branch
    # initializer before restoring the demonstration anchor and replaying the
    # exact expert prefix.
    env.seed(0)
    env.reset()
    restore_state(env, states[candidate.anchor_step])
    tracker = initial_tracker(env, candidate, states, actions)
    applied: dict[str, Any] | None = None
    phase_contract: dict[str, bool] | None = None
    prefix_contacts: list[tuple[tuple[str, str], ...]] = []
    for index, action in enumerate(chunk[:prefix_k]):
        env.step(action)
        tracker.observe_action(env, action)
        if index + 1 == candidate.insertion_prefix:
            before_contacts = contact_pairs(env)
            applied = apply_expert_perturbation(env, candidate)
            tracker.observe_injection(env, before_contacts)
            if tracker.spec.family == "target_shift":
                object_key = _joint_object_key(tracker.spec.object_joint)
                target_key = _joint_object_key(tracker.spec.target_joint)
                target_clear = not any(
                    any(object_key in geom for geom in pair)
                    and any(target_key in geom for geom in pair)
                    for pair in before_contacts
                )
                release_not_occurred = bool(
                    candidate.release_step is not None
                    and candidate.anchor_step + index < candidate.release_step
                )
                phase_contract = {
                    "stable_lift": bool(tracker.ever_lifted),
                    "at_least_two_transport_actions": candidate.insertion_prefix >= 2,
                    "release_not_occurred": release_not_occurred,
                    "gripper_closed": bool(float(action[-1]) > 0),
                    "target_clear_before_injection": target_clear,
                }
            else:
                phase_contract = {
                    "no_contact_at_injection": not tracker.immediate_violation,
                }
        prefix_contacts.append(contact_pairs(env))
    if applied is None:
        raise RuntimeError("perturbation was not applied")
    state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
    return tracker, {
        "state_hash": state_sha256(state),
        "state": state,
        "contacts_hash": hashlib.sha256(repr(prefix_contacts).encode("utf-8")).hexdigest(),
        "contacts": prefix_contacts,
        "success": bool(env.check_success()),
        "perturbation": applied,
        "phase_contract": phase_contract,
        "chunk": chunk,
        "last_original_action": chunk[prefix_k - 1].copy(),
    }


def _eef_position(env) -> np.ndarray:
    return np.asarray(env.sim.data.site_xpos[env.robots[0].eef_site_id], dtype=np.float64).copy()


def _controller_action(delta_position: np.ndarray, gripper: float) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[:3] = np.clip(np.asarray(delta_position, dtype=np.float64) / 0.05, -1.0, 1.0)
    action[-1] = float(gripper)
    return action


def _step_branch(
    env,
    tracker: CauseTracker,
    action: np.ndarray,
    executed: list[np.ndarray],
    *,
    budget: int,
) -> bool:
    if len(executed) >= budget or tracker.violation:
        return False
    env.step(np.asarray(action, dtype=np.float32))
    tracker.observe_action(env, action)
    executed.append(np.asarray(action, dtype=np.float32).copy())
    return not tracker.violation


def _move_object_delta(
    env,
    tracker: CauseTracker,
    executed: list[np.ndarray],
    delta_fn,
    *,
    tolerance: float,
    maximum_steps: int,
    budget: int,
    gripper: float,
) -> None:
    for _ in range(maximum_steps):
        delta = np.asarray(delta_fn(), dtype=np.float64)
        if float(np.linalg.norm(delta)) <= tolerance:
            break
        if not _step_branch(
            env,
            tracker,
            _controller_action(delta, gripper),
            executed,
            budget=budget,
        ):
            break


def _branch_result(
    env,
    tracker: CauseTracker,
    executed: list[np.ndarray],
    *,
    operator: str,
    retained_nominal_actions: int,
    discarded_nominal_actions: int,
) -> dict[str, Any]:
    success = bool(env.check_success())
    path_length = float(sum(np.linalg.norm(action[:3]) for action in executed))
    result = {
        "operator": operator,
        "cause_violation": bool(tracker.violation),
        "violation_type": tracker.violation_type,
        "task_success": success,
        "safe_success": bool(success and not tracker.violation),
        "new_non_nominal_actions": len(executed) - retained_nominal_actions,
        "retained_nominal_actions": retained_nominal_actions,
        "discarded_nominal_actions": discarded_nominal_actions,
        "policy_calls": 0,
        "completion_steps": len(executed),
        "path_length_action_space": path_length,
        "executed_action_hash": hashlib.sha256(
            np.ascontiguousarray(np.asarray(executed, dtype=np.float32)).tobytes()
        ).hexdigest(),
        "final_state_hash": state_sha256(env.get_sim_state()),
    }
    if tracker.spec.object_joint and tracker.spec.target_joint:
        object_qpos = _joint_qpos(env, tracker.spec.object_joint)
        target_qpos = _joint_qpos(env, tracker.spec.target_joint)
        result["final_object_qpos"] = object_qpos.tolist()
        result["final_target_qpos"] = target_qpos.tolist()
        result["final_object_target_xy_error"] = float(
            np.linalg.norm(object_qpos[:2] - target_qpos[:2])
        )
        result["final_eef_position"] = _eef_position(env).tolist()
    return result


def run_full_replan(
    env,
    tracker: CauseTracker,
    branch_info: dict[str, Any],
    *,
    prefix_k: int,
    budget: int = BRANCH_BUDGET,
) -> dict[str, Any]:
    spec = tracker.spec
    executed: list[np.ndarray] = []
    discarded = CHUNK_LENGTH - prefix_k
    if spec.family == "drawer_obstacle":
        # Fixed generic stop/retract controller. It deliberately does not move
        # the environment obstacle; failure is a valid calibration outcome.
        recent = np.asarray(branch_info["chunk"][:prefix_k], dtype=np.float32)
        for source in reversed(recent[-4:]):
            action = source.copy()
            action[:6] *= -1.0
            action[-1] = -1.0
            if not _step_branch(env, tracker, action, executed, budget=budget):
                break
        for _ in range(4):
            if not _step_branch(
                env,
                tracker,
                np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
                executed,
                budget=budget,
            ):
                break
        return _branch_result(
            env,
            tracker,
            executed,
            operator="full_replan",
            retained_nominal_actions=0,
            discarded_nominal_actions=discarded,
        )

    assert spec.object_joint is not None and spec.target_joint is not None
    if float(branch_info["last_original_action"][-1]) < 0:
        return _branch_result(
            env,
            tracker,
            executed,
            operator="full_replan",
            retained_nominal_actions=0,
            discarded_nominal_actions=discarded,
        )

    for _ in range(2):
        if not _step_branch(
            env,
            tracker,
            np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            executed,
            budget=budget,
        ):
            break

    _move_object_delta(
        env,
        tracker,
        executed,
        lambda: np.asarray([0.0, 0.0, max(0.0, _joint_qpos(env, spec.target_joint)[2] + 0.18 - _joint_qpos(env, spec.object_joint)[2])]),
        tolerance=0.012,
        maximum_steps=6,
        budget=budget,
        gripper=1.0,
    )
    _move_object_delta(
        env,
        tracker,
        executed,
        lambda: np.r_[_joint_qpos(env, spec.target_joint)[:2] - _joint_qpos(env, spec.object_joint)[:2], 0.0],
        tolerance=0.010,
        maximum_steps=10,
        budget=budget,
        gripper=1.0,
    )
    _move_object_delta(
        env,
        tracker,
        executed,
        lambda: np.asarray([0.0, 0.0, _joint_qpos(env, spec.target_joint)[2] + spec.placement_z_offset - _joint_qpos(env, spec.object_joint)[2]]),
        tolerance=0.010,
        maximum_steps=10,
        budget=budget,
        gripper=1.0,
    )
    for _ in range(4):
        if not _step_branch(
            env,
            tracker,
            np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
            executed,
            budget=budget,
        ):
            break
    _move_object_delta(
        env,
        tracker,
        executed,
        lambda: np.asarray([0.0, 0.0, 0.10]),
        tolerance=0.012,
        maximum_steps=4,
        budget=budget,
        gripper=-1.0,
    )
    for _ in range(4):
        if bool(env.check_success()) or len(executed) >= budget or tracker.violation:
            break
        _step_branch(
            env,
            tracker,
            np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
            executed,
            budget=budget,
        )
    return _branch_result(
        env,
        tracker,
        executed,
        operator="full_replan",
        retained_nominal_actions=0,
        discarded_nominal_actions=discarded,
    )


def run_local_repair(
    env,
    tracker: CauseTracker,
    branch_info: dict[str, Any],
    *,
    prefix_k: int,
    budget: int = BRANCH_BUDGET,
) -> dict[str, Any]:
    spec = tracker.spec
    executed: list[np.ndarray] = []
    chunk = np.asarray(branch_info["chunk"], dtype=np.float32)
    if spec.family == "drawer_obstacle":
        recent = chunk[:prefix_k]
        for source in reversed(recent[-4:]):
            action = source.copy()
            action[:6] *= -1.0
            action[-1] = -1.0
            if not _step_branch(env, tracker, action, executed, budget=budget):
                break
        return _branch_result(
            env,
            tracker,
            executed,
            operator="cause_specific_local_repair",
            retained_nominal_actions=0,
            discarded_nominal_actions=CHUNK_LENGTH - prefix_k,
        )

    assert spec.object_joint is not None and spec.target_joint is not None
    if float(branch_info["last_original_action"][-1]) < 0:
        return _branch_result(
            env,
            tracker,
            executed,
            operator="cause_specific_local_repair",
            retained_nominal_actions=0,
            discarded_nominal_actions=CHUNK_LENGTH - prefix_k,
        )
    before_repair = len(executed)
    _move_object_delta(
        env,
        tracker,
        executed,
        lambda: np.r_[_joint_qpos(env, spec.target_joint)[:2] - _joint_qpos(env, spec.object_joint)[:2], 0.0],
        tolerance=0.010,
        maximum_steps=6,
        budget=budget,
        gripper=1.0,
    )
    repair_count = len(executed) - before_repair
    retained = 0
    for action in chunk[prefix_k:]:
        if not _step_branch(env, tracker, action, executed, budget=budget):
            break
        retained += 1
        if bool(env.check_success()):
            break
    for _ in range(4):
        if bool(env.check_success()) or len(executed) >= budget or tracker.violation:
            break
        _step_branch(
            env,
            tracker,
            np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
            executed,
            budget=budget,
        )
    result = _branch_result(
        env,
        tracker,
        executed,
        operator="cause_specific_local_repair",
        retained_nominal_actions=retained,
        discarded_nominal_actions=(CHUNK_LENGTH - prefix_k) - retained,
    )
    result["repair_action_count"] = repair_count
    return result


def run_nominal(env, candidate: ExpertCandidate) -> dict[str, Any]:
    tracker, info = reconstruct_prefix(env, candidate, prefix_k=CHUNK_LENGTH)
    return {
        "candidate_id": candidate.candidate_id,
        "task": candidate.task,
        "demo_id": candidate.demo_id,
        "config_id": candidate.config_id,
        "parameters": candidate.parameters,
        "anchor_step": candidate.anchor_step,
        "release_step": candidate.release_step,
        "insertion_prefix": candidate.insertion_prefix,
        "immediate_violation": tracker.immediate_violation,
        "cause_violation": tracker.violation,
        "violation_type": tracker.violation_type,
        "first_violation_post_detection_step": tracker.first_violation_post_detection_step,
        "task_success": bool(env.check_success()),
        "safe_success": bool(env.check_success() and not tracker.violation),
        "final_state_hash": info["state_hash"],
        "contact_trace_hash": hashlib.sha256(repr(tracker.contact_trace).encode("utf-8")).hexdigest(),
        "perturbation": info["perturbation"],
        "phase_contract": info["phase_contract"],
        "phase_contract_valid": bool(
            info["phase_contract"] and all(info["phase_contract"].values())
        ),
    }


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
