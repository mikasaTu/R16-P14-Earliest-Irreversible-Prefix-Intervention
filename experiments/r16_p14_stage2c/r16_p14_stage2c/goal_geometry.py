from __future__ import annotations

from typing import Any

import numpy as np

from r16_p14_stage2a.envs import joint_qpos
from r16_p14_stage2a.settings import TASK_SPECS


def bddl_goal_contract(env, task: str) -> dict[str, Any]:
    goals = list(env.env.parsed_problem["goal_state"])
    if len(goals) != 1 or len(goals[0]) != 3:
        raise ValueError(f"expected one binary goal predicate, got {goals!r}")
    relation, manipulated_name, target_name = (str(value) for value in goals[0])
    spec = TASK_SPECS[task]
    assert spec.manipulated_joint
    expected_manipulated = spec.manipulated_joint.removesuffix("_joint0")
    if manipulated_name != expected_manipulated:
        raise ValueError(f"BDDL manipulated object mismatch: {manipulated_name} != {expected_manipulated}")
    if relation not in {"on", "in"}:
        raise ValueError(f"unsupported goal relation: {relation}")
    target_kind = "site" if target_name in env.env.object_sites_dict else "object"
    if target_kind == "object" and target_name not in env.env.objects_dict:
        raise ValueError(f"unknown BDDL target: {target_name}")
    return {
        "task": task,
        "relation": relation,
        "manipulated_name": manipulated_name,
        "manipulated_joint": spec.manipulated_joint,
        "target_name": target_name,
        "target_kind": target_kind,
        "source": "env.env.parsed_problem.goal_state",
        "demo_endpoint_used": False,
    }


def _point_box_distance(point: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    below = np.maximum(lower - point, 0.0)
    above = np.maximum(point - upper, 0.0)
    return float(np.linalg.norm(below + above))


def goal_geometry(env, task: str) -> dict[str, Any]:
    contract = bddl_goal_contract(env, task)
    target = contract["target_name"]
    if contract["target_kind"] == "object":
        body_id = int(env.env.obj_body_id[target])
        center = np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64)
        return {
            **contract,
            "center": center.tolist(),
            "horizontal_support_radius_m": 0.03,
            "predicate_source": "ObjectState.check_ontop",
        }
    site = env.env.object_sites_dict[target]
    center = np.asarray(env.sim.data.get_site_xpos(target), dtype=np.float64)
    rotation = np.asarray(env.sim.data.get_site_xmat(target), dtype=np.float64).reshape(3, 3)
    size = np.asarray(site.size, dtype=np.float64)
    return {
        **contract,
        "center": center.tolist(),
        "rotation": rotation.tolist(),
        "size": size.tolist(),
        "parent_name": site.parent_name,
        "predicate_source": "SiteObject.in_box" if contract["relation"] == "in" else "SiteObject.under",
    }


def target_distance(env, task: str) -> float:
    contract = bddl_goal_contract(env, task)
    spec = TASK_SPECS[task]
    assert spec.manipulated_joint
    point = np.asarray(joint_qpos(env, spec.manipulated_joint)[:3], dtype=np.float64)
    target = contract["target_name"]
    if contract["target_kind"] == "object":
        target_id = int(env.env.obj_body_id[target])
        center = np.asarray(env.sim.data.body_xpos[target_id], dtype=np.float64)
        return max(0.0, float(np.linalg.norm(point[:2] - center[:2])) - 0.03)
    site = env.env.object_sites_dict[target]
    center = np.asarray(env.sim.data.get_site_xpos(target), dtype=np.float64)
    rotation = np.asarray(env.sim.data.get_site_xmat(target), dtype=np.float64).reshape(3, 3)
    size = np.asarray(site.size, dtype=np.float64)
    if contract["relation"] == "in":
        # Mirror LIBERO SiteObject.in_box, including its axis-aligned transformed extent.
        total_size = np.abs(rotation @ size)
        lower, upper = center - total_size, center + total_size
        lower[2] -= 0.01
        return _point_box_distance(point, lower, upper)
    local = rotation @ (point - center)
    lower = np.asarray([-size[0], -size[1], size[2] - 0.005], dtype=np.float64)
    upper = np.asarray([size[0], size[1], size[2] + 0.10], dtype=np.float64)
    return _point_box_distance(local, lower, upper)

