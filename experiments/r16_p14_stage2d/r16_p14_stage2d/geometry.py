from __future__ import annotations

from typing import Any

import numpy as np

from r16_p14_stage2a.envs import joint_qpos


def object_name_from_joint(joint_name: str) -> str:
    suffix = "_joint0"
    if not joint_name.endswith(suffix):
        raise ValueError(f"unsupported free joint name: {joint_name}")
    return joint_name[: -len(suffix)]


def live_bounding_sphere(env, joint_name: str) -> dict[str, Any]:
    """Return a conservative live XY bounding sphere from MuJoCo geoms.

    Each contact geom contributes its MuJoCo ``geom_rbound`` plus its current
    horizontal offset from the object's free-joint origin. The maximum is a
    documented conservative sphere radius and is invariant to rigid rotation.
    """

    object_name = object_name_from_joint(joint_name)
    object_model = env.env.objects_dict[object_name]
    origin = np.asarray(joint_qpos(env, joint_name)[:3], dtype=np.float64)
    entries: list[dict[str, Any]] = []
    for geom_name in sorted(object_model.contact_geoms):
        geom_id = int(env.sim.model.geom_name2id(geom_name))
        center = np.asarray(env.sim.data.geom_xpos[geom_id], dtype=np.float64)
        rbound = float(env.sim.model.geom_rbound[geom_id])
        horizontal_offset = float(np.linalg.norm(center[:2] - origin[:2]))
        entries.append(
            {
                "geom": geom_name,
                "geom_id": geom_id,
                "center": center.tolist(),
                "geom_rbound_m": rbound,
                "horizontal_offset_m": horizontal_offset,
                "candidate_radius_m": horizontal_offset + rbound,
            }
        )
    if not entries:
        raise RuntimeError(f"object has no contact geoms: {object_name}")
    radius = max(item["candidate_radius_m"] for item in entries)
    return {
        "object_name": object_name,
        "joint_name": joint_name,
        "origin": origin.tolist(),
        "radius_m": float(radius),
        "method": "max_xy_geom_offset_plus_mujoco_geom_rbound",
        "conservative_bounding_sphere": True,
        "geoms": entries,
    }


def lateral_unit(start_xy: Any, end_xy: Any) -> np.ndarray:
    direction = np.asarray(end_xy, dtype=np.float64)[:2] - np.asarray(
        start_xy, dtype=np.float64
    )[:2]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("cannot define a lateral direction for a zero-length path")
    direction /= norm
    return np.asarray([-direction[1], direction[0]], dtype=np.float64)
