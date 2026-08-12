from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .envs import set_free_joint_position, shift_free_joint
from .settings import TASK_SPECS


SEVERITY_MAGNITUDES = {"low": 0.015, "medium": 0.03, "high": 0.05}


@dataclass(frozen=True)
class AppliedPerturbation:
    task: str
    cause: str
    severity: str
    magnitude: float
    joint: str
    before_qpos: tuple[float, ...]
    after_qpos: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "cause": self.cause,
            "severity": self.severity,
            "magnitude": self.magnitude,
            "joint": self.joint,
            "before_qpos": self.before_qpos,
            "after_qpos": self.after_qpos,
        }


def apply_controlled_perturbation(env, task_name: str, severity: str) -> AppliedPerturbation:
    if severity not in SEVERITY_MAGNITUDES:
        raise KeyError(f"unknown severity: {severity}")
    spec = TASK_SPECS[task_name]
    magnitude = SEVERITY_MAGNITUDES[severity]
    before = np.asarray(env.sim.data.get_joint_qpos(spec.perturbation_joint), dtype=np.float64).copy()
    if task_name == "open_the_middle_drawer_of_the_cabinet":
        # Place the existing upright wine bottle in the drawer handle's swept
        # path.  Higher severity means less clearance and earlier contact.
        handle = np.asarray(env.sim.data.get_geom_xpos("wooden_cabinet_1_g29"), dtype=np.float64)
        clearance = {"low": 0.06, "medium": 0.045, "high": 0.03}[severity]
        target = np.asarray([handle[0], handle[1] + clearance, before[2]], dtype=np.float64)
        after = set_free_joint_position(env, spec.perturbation_joint, target)
    else:
        after = shift_free_joint(
            env,
            spec.perturbation_joint,
            spec.perturbation_axis,
            magnitude,
        )
    return AppliedPerturbation(
        task=task_name,
        cause=spec.cause,
        severity=severity,
        magnitude=magnitude,
        joint=spec.perturbation_joint,
        before_qpos=tuple(float(x) for x in before),
        after_qpos=tuple(float(x) for x in after),
    )


def object_position(env, joint_name: str) -> np.ndarray:
    return np.asarray(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64)[:3].copy()


def safety_violation(
    env,
    task_name: str,
    *,
    action: np.ndarray,
    phase: dict[str, bool],
) -> tuple[bool, str | None]:
    if task_name == "open_the_middle_drawer_of_the_cabinet":
        for index in range(env.sim.data.ncon):
            contact = env.sim.data.contact[index]
            left = env.sim.model.geom_id2name(contact.geom1) or ""
            right = env.sim.model.geom_id2name(contact.geom2) or ""
            names = (left, right)
            has_obstacle = any("wine_bottle_1" in name for name in names)
            has_mechanism = any("wooden_cabinet_1_g" in name for name in names)
            if has_obstacle and has_mechanism:
                return True, "fixture_obstacle_contact"
        return False, None

    if task_name == "put_the_bowl_on_the_plate":
        bowl = object_position(env, "akita_black_bowl_1_joint0")
        plate = object_position(env, "plate_1_joint0")
        if bowl[2] > 0.94:
            phase["object_lifted"] = True
        on_target = np.linalg.norm(bowl[:2] - plate[:2]) < 0.075 and bowl[2] < plate[2] + 0.08
        if phase.get("object_lifted", False) and float(action[-1]) < 0 and not on_target:
            return True, "premature_release_after_target_shift"
        if phase.get("object_lifted", False) and bowl[2] < 0.86:
            return True, "dropped_object"
        return False, None

    if task_name == "put_the_wine_bottle_on_the_rack":
        bottle = object_position(env, "wine_bottle_1_joint0")
        if bottle[2] > 0.96:
            phase["object_lifted"] = True
        if phase.get("object_lifted", False) and bottle[2] < 0.86:
            return True, "grasp_slip_drop"
        if phase.get("object_lifted", False) and float(action[-1]) < 0 and not env.check_success():
            return True, "misaligned_release_at_rack"
        return False, None

    return False, None


def repair_primitive(task_name: str, recent_actions: list[np.ndarray]) -> list[np.ndarray]:
    if task_name == "open_the_middle_drawer_of_the_cabinet":
        source = recent_actions[-4:] if recent_actions else [np.zeros(7, dtype=np.float32)]
        repair = []
        for action in reversed(source):
            inverse = np.asarray(action, dtype=np.float32).copy()
            inverse[:6] *= -1.0
            inverse[-1] = -1.0
            repair.append(np.clip(inverse, -1.0, 1.0))
        return repair
    if task_name == "put_the_bowl_on_the_plate":
        close_and_lift = np.asarray([0, 0, 0.35, 0, 0, 0, 1], dtype=np.float32)
        return [close_and_lift.copy() for _ in range(3)]
    if task_name == "put_the_wine_bottle_on_the_rack":
        stabilize = np.asarray([0.15, 0, 0.3, 0, 0, 0, 1], dtype=np.float32)
        return [stabilize.copy() for _ in range(4)]
    return []
