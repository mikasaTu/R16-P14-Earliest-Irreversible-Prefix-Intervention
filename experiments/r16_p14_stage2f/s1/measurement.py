"""D1-D3 measurements and pure labels for the zero-injection backend."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from r16_p14_stage2a.envs import joint_qpos
from r16_p14_stage2a.settings import TASK_SPECS

SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _array_hash(value: Any, dtype: Any = np.float32) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _as_pair(left: Any, right: Any) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def _maybe_int(value: Any) -> int | None:
    """Keep numeric MuJoCo ids numeric while accepting name-only stubs."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _name_from_geom_id(env: Any, geom_id: Any) -> str:
    model = getattr(getattr(env, "sim", None), "model", None)
    resolver = getattr(model, "geom_id2name", None)
    if callable(resolver):
        try:
            name = resolver(int(geom_id))
            if name:
                return str(name)
        except (TypeError, ValueError, IndexError, AttributeError):
            pass
    return str(geom_id)


def _contact_snapshot(env: Any) -> dict[str, Any]:
    """Return normalized unique topology plus every raw contact row."""
    data = getattr(getattr(env, "sim", None), "data", None)
    contacts = getattr(data, "contact", None)
    ncon = getattr(data, "ncon", None)
    raw: list[dict[str, Any]] = []
    available = contacts is not None and ncon is not None
    if available:
        try:
            count = max(0, int(ncon))
        except (TypeError, ValueError):
            count = 0
            available = False
        for index in range(count):
            contact = contacts[index]
            left_id = getattr(contact, "geom1", None)
            right_id = getattr(contact, "geom2", None)
            left_name = right_name = None
            if isinstance(contact, dict):
                left_id = contact.get("geom1", contact.get("geom1_id", left_id))
                right_id = contact.get("geom2", contact.get("geom2_id", right_id))
                left_name = contact.get("geom1_name")
                right_name = contact.get("geom2_name")
            elif isinstance(contact, (list, tuple)) and len(contact) >= 2:
                left_id, right_id = contact[0], contact[1]
            left_name = left_name or _name_from_geom_id(env, left_id)
            right_name = right_name or _name_from_geom_id(env, right_id)
            raw.append(
                {
                    "raw_index": int(index),
                    "geom1_id": _maybe_int(left_id),
                    "geom2_id": _maybe_int(right_id),
                    "geom1_name": str(left_name),
                    "geom2_name": str(right_name),
                    "normalized_pair": list(_as_pair(left_name, right_name)),
                }
            )
    if not available:
        alternate = getattr(env, "contacts", None)
        if alternate is None:
            alternate = getattr(getattr(env, "sim", None), "contacts", None)
        if alternate is not None:
            available = True
            for index, item in enumerate(list(alternate)):
                left_id = right_id = None
                left_name = right_name = None
                if isinstance(item, dict):
                    left_id = item.get("geom1_id", item.get("geom1"))
                    right_id = item.get("geom2_id", item.get("geom2"))
                    left_name = item.get("geom1_name")
                    right_name = item.get("geom2_name")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    left_name, right_name = item[0], item[1]
                left_name = left_name or _name_from_geom_id(env, left_id)
                right_name = right_name or _name_from_geom_id(env, right_id)
                raw.append(
                    {
                        "raw_index": int(index),
                        "geom1_id": _maybe_int(left_id),
                        "geom2_id": _maybe_int(right_id),
                        "geom1_name": str(left_name),
                        "geom2_name": str(right_name),
                        "normalized_pair": list(_as_pair(left_name, right_name)),
                    }
                )
    return {
        "available": bool(available),
        "normalized": [list(pair) for pair in sorted({tuple(item["normalized_pair"]) for item in raw})],
        "raw": raw,
    }


def _normalise_pairs(value: Any) -> list[list[str]]:
    if value is None:
        return []
    pairs: set[tuple[str, str]] = set()
    for item in value:
        if isinstance(item, dict):
            pair = item.get("normalized_pair") or (item.get("geom1_name"), item.get("geom2_name"))
        else:
            pair = item
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            pairs.add(_as_pair(pair[0], pair[1]))
    return [list(pair) for pair in sorted(pairs)]


def _model_geom_names(env: Any) -> list[str]:
    model = getattr(getattr(env, "sim", None), "model", None)
    count = getattr(model, "ngeom", None)
    resolver = getattr(model, "geom_id2name", None)
    if count is None or not callable(resolver):
        return []
    result = []
    for index in range(int(count)):
        try:
            name = resolver(index)
        except (TypeError, ValueError, IndexError, AttributeError):
            name = None
        if name:
            result.append(str(name))
    return sorted(set(result))


def _geometry_names(env: Any, task: str) -> dict[str, list[str]]:
    all_names = _model_geom_names(env)
    spec = TASK_SPECS[task]

    def attr_names(*keys: str) -> list[str]:
        for key in keys:
            value = getattr(env, key, None)
            if value is None:
                continue
            if isinstance(value, str):
                return [value]
            return sorted({str(item) for item in value if item is not None})
        return []

    object_names = attr_names("_r16_object_geometry_names", "object_geometry_names")
    target_names = attr_names("_r16_target_geometry_names", "target_geometry_names")
    robot_names = attr_names("_r16_robot_geometry_names", "robot_geometry_names")
    if not object_names and spec.manipulated_joint:
        prefix = spec.manipulated_joint.rsplit("_joint", 1)[0]
        object_names = [name for name in all_names if name.startswith(prefix + "_")]
    if not target_names and spec.target_joint:
        prefix = spec.target_joint.rsplit("_joint", 1)[0]
        target_names = [name for name in all_names if name.startswith(prefix + "_")]
    if not robot_names:
        robot_names = [
            name for name in all_names
            if any(token in name.lower() for token in ("robot", "gripper", "finger", "eef", "hand"))
        ]
    return {"object": sorted(set(object_names)), "target": sorted(set(target_names)), "robot": sorted(set(robot_names))}


def _safe_qpos(env: Any, joint_name: str | None) -> tuple[list[float] | None, str | None]:
    if not joint_name:
        return None, None
    try:
        value = np.asarray(joint_qpos(env, joint_name), dtype=np.float64)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        return None, f"qpos:{joint_name}:{type(exc).__name__}"
    if value.ndim != 1 or not np.isfinite(value).all():
        return None, f"qpos:{joint_name}:invalid"
    return value.tolist(), None


def record_step(
    env: Any,
    task: str,
    *,
    event_instance_id: str,
    prefix_k: int,
    operator: str,
    actor_seed: int,
    step: int,
    action: Any,
    previous_gripper: float,
    initial_object_z: float,
    ever_lifted: bool,
    baseline_contacts: Any = None,
    step_kind: str = "action_step",
    physics_step_index: int | None = None,
    control_step: int | None = None,
    substep: int | None = None,
) -> dict[str, Any]:
    """Capture one JSON-safe post-action record with complete D1 fields."""
    if task not in TASK_SPECS:
        raise KeyError(task)
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    control_step = int(step if control_step is None else control_step)
    if substep is None and str(step_kind) == "action_step":
        substep = 0
    contacts = _contact_snapshot(env)
    names = _geometry_names(env, task)
    spec = TASK_SPECS[task]
    object_qpos, object_error = _safe_qpos(env, spec.manipulated_joint)
    target_qpos, target_error = _safe_qpos(env, spec.target_joint)
    missing: list[str] = []
    if not contacts["available"]:
        missing.append("contact_stream")
    if object_qpos is None:
        missing.append(object_error or "object_qpos")
    if spec.target_joint and target_qpos is None:
        missing.append(target_error or "target_qpos")
    current_gripper = float(action_array[-1]) if action_array.size else None
    release_transition = bool(
        current_gripper is not None and float(previous_gripper) > 0.0 and current_gripper < 0.0
    )
    lifted_now = bool(
        object_qpos is not None
        and float(object_qpos[2]) - float(initial_object_z) >= float(spec.lift_delta)
    )
    ever_lifted_after = bool(ever_lifted or lifted_now)
    try:
        task_success = bool(env.check_success())
    except (AttributeError, TypeError, ValueError):
        task_success = False
        missing.append("task_success")
    baseline = None if baseline_contacts is None else _normalise_pairs(baseline_contacts)
    current_pairs = contacts["normalized"]
    new_pairs = None if baseline is None else sorted(
        set(tuple(item) for item in current_pairs) - set(tuple(item) for item in baseline)
    )
    phase = {
        "task": task,
        "task_success": task_success,
        "lifted_now": lifted_now,
        "ever_lifted": ever_lifted_after,
        "gripper_closed": bool(current_gripper is not None and current_gripper > 0.0),
        "release_transition": release_transition,
        "object_height_delta_m": None if object_qpos is None else float(object_qpos[2]) - float(initial_object_z),
    }
    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "event_instance_id": str(event_instance_id),
            "prefix_k": int(prefix_k),
            "operator": str(operator),
            "actor_seed": int(actor_seed),
            "step": int(step),
            "control_step": control_step,
            "substep": substep,
            "step_kind": str(step_kind),
            "physics_step_index": physics_step_index,
            "action": action_array.tolist(),
            "action_hash": _array_hash(action_array, np.float32),
            "previous_gripper": float(previous_gripper),
            "current_gripper": current_gripper,
            "initial_object_z": float(initial_object_z),
            "contact_stream_available": bool(contacts["available"]),
            "normalized_contact_pairs": contacts["normalized"],
            "contact_pairs": contacts["normalized"],
            "raw_contact_pairs": contacts["raw"],
            "contact_count": len(contacts["raw"]),
            "baseline_contact_pairs": baseline,
            "new_contact_pairs": None if new_pairs is None else [list(item) for item in new_pairs],
            "geometry_names": names,
            "object_geometry_names": names["object"],
            "target_geometry_names": names["target"],
            "robot_geometry_names": names["robot"],
            "object_qpos": object_qpos,
            "target_qpos": target_qpos,
            "task_phase": phase,
            "missing": sorted(set(missing)),
        }
    )


def _pair_sets(record: dict[str, Any]) -> tuple[set[tuple[str, str]] | None, set[tuple[str, str]] | None]:
    current = record.get("normalized_contact_pairs", record.get("contact_pairs"))
    baseline = record.get("baseline_contact_pairs")
    if current is None:
        return None, None
    current_set = {tuple(_as_pair(item[0], item[1])) for item in current if len(item) >= 2}
    if baseline is None:
        return current_set, None
    baseline_set = {tuple(_as_pair(item[0], item[1])) for item in baseline if len(item) >= 2}
    return current_set, baseline_set


def _is_pre_anchor_record(record: dict[str, Any]) -> bool:
    phase = record.get("phase")
    if phase is None:
        task_phase = record.get("task_phase") or {}
        phase = task_phase.get("phase")
    return str(phase).strip().lower() in {"pre_anchor", "before_anchor", "pre-anchor"}


def label_trace(records: Iterable[dict[str, Any]], task: str) -> dict[str, Any]:
    """Purely label release failure and a parallel pre-release topology candidate."""
    if task not in TASK_SPECS:
        raise KeyError(task)
    rows = [dict(record) for record in records]
    spec = TASK_SPECS[task]
    release_missing: set[str] = set()
    candidate_missing: set[str] = set()
    release_indices: list[int] = []
    release_violation_indices: list[int] = []
    candidate_indices: list[int] = []
    candidate_pairs: list[dict[str, Any]] = []
    first_release_index = None
    first_release_violation_index = None
    first_candidate_index = None
    release_seen = False
    lifted_closed_indices: list[int] = []
    ignored_pre_lift_release_indices: list[int] = []
    ignored_pre_anchor_candidate_indices: list[int] = []
    for index, record in enumerate(rows):
        phase = record.get("task_phase") or {}
        action = record.get("action")
        current_gripper = record.get("current_gripper")
        if current_gripper is None and isinstance(action, (list, tuple)) and action:
            current_gripper = action[-1]
        previous_gripper = record.get("previous_gripper")
        release_transition = record.get("release_transition")
        if release_transition is None:
            release_transition = bool(
                previous_gripper is not None and current_gripper is not None
                and float(previous_gripper) > 0.0 and float(current_gripper) < 0.0
            )
        if record.get("step_kind", "action_step") != "action_step":
            release_transition = False
        lifted = bool(phase.get("ever_lifted", phase.get("lifted_now", record.get("ever_lifted", False))))
        if release_transition:
            if not lifted:
                ignored_pre_lift_release_indices.append(index)
            else:
                release_seen = True
                release_indices.append(index)
                if first_release_index is None:
                    first_release_index = index
                object_qpos = record.get("object_qpos")
                target_qpos = record.get("target_qpos")
                if object_qpos is None:
                    release_missing.add("object_qpos")
                if target_qpos is None:
                    release_missing.add("target_qpos")
                if object_qpos is not None and target_qpos is not None:
                    try:
                        distance = float(np.linalg.norm(
                            np.asarray(object_qpos, dtype=np.float64)[:2]
                            - np.asarray(target_qpos, dtype=np.float64)[:2]
                        ))
                        if distance > float(spec.placement_xy_tolerance):
                            release_violation_indices.append(index)
                            if first_release_violation_index is None:
                                first_release_violation_index = index
                    except (TypeError, ValueError, IndexError):
                        release_missing.add("release_distance")
        gripper_closed = bool(
            phase.get("gripper_closed", current_gripper is not None and float(current_gripper) > 0.0)
        )
        if lifted and gripper_closed:
            lifted_closed_indices.append(index)
        if lifted and gripper_closed and not release_seen:
            if _is_pre_anchor_record(record):
                ignored_pre_anchor_candidate_indices.append(index)
                continue
            current_pairs, baseline_pairs = _pair_sets(record)
            names = record.get("geometry_names") or {}
            object_names = set(record.get("object_geometry_names", names.get("object", [])))
            target_names = set(record.get("target_geometry_names", names.get("target", [])))
            robot_names = set(record.get("robot_geometry_names", names.get("robot", [])))
            if current_pairs is None:
                candidate_missing.add("contact_pairs")
            if baseline_pairs is None:
                candidate_missing.add("baseline_contacts")
            if not object_names:
                candidate_missing.add("object_geometry_names")
            if not target_names:
                candidate_missing.add("target_geometry_names")
            if not robot_names:
                candidate_missing.add("robot_geometry_names")
            if current_pairs is not None and baseline_pairs is not None and object_names:
                for pair in sorted(current_pairs - baseline_pairs):
                    left, right = pair
                    if left in object_names:
                        other = right
                    elif right in object_names:
                        other = left
                    else:
                        continue
                    if other in object_names or other in target_names or other in robot_names:
                        continue
                    candidate_indices.append(index)
                    candidate_pairs.append({"index": index, "pair": list(pair)})
                    if first_candidate_index is None:
                        first_candidate_index = index
    release_missing_list = sorted(release_missing)
    candidate_missing_list = sorted(candidate_missing)
    candidate_before_release = (
        None if first_candidate_index is None or first_release_index is None
        else bool(first_candidate_index < first_release_index)
    )
    if first_candidate_index is None:
        candidate_order = "none"
    elif first_release_index is None:
        candidate_order = "candidate_without_release"
    elif first_candidate_index < first_release_index:
        candidate_order = "candidate_before_release"
    elif first_candidate_index == first_release_index:
        candidate_order = "same_step_as_release"
    else:
        candidate_order = "candidate_after_release"
    release_based = {
        "label": "release_outside_original_target",
        "violation": bool(release_violation_indices),
        "violation_type": "release_outside_original_target" if release_violation_indices else None,
        "first_index": first_release_violation_index,
        "release_indices": release_indices,
        "ignored_pre_lift_release_indices": ignored_pre_lift_release_indices,
        "release_violation_indices": release_violation_indices,
        "placement_xy_tolerance_m": float(spec.placement_xy_tolerance),
        "missing": release_missing_list,
        "time_order": {
            "first_release_index": first_release_index,
            "ignored_pre_lift_release_indices": ignored_pre_lift_release_indices,
            "first_violation_index": first_release_violation_index,
            "lift_before_release_required": True,
            "complete": not release_missing_list,
        },
    }
    topology_candidate = {
        "label": "prerelease_object_contact_topology_candidate",
        "candidate": bool(candidate_indices),
        "first_index": first_candidate_index,
        "candidate_indices": sorted(set(candidate_indices)),
        "new_pairs": candidate_pairs,
        "lifted_closed_indices": lifted_closed_indices,
        "ignored_pre_anchor_indices": ignored_pre_anchor_candidate_indices,
        "missing": candidate_missing_list,
        "time_order": {
            "first_candidate_index": first_candidate_index,
            "first_release_index": first_release_index,
            "candidate_before_release": candidate_before_release,
            "ordering": candidate_order,
            "complete": not candidate_missing_list,
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "record_count": len(rows),
        "first_release_index": first_release_index,
        "ignored_pre_lift_release_indices": ignored_pre_lift_release_indices,
        "first_candidate_index": first_candidate_index,
        "missing": sorted(set(release_missing_list + candidate_missing_list)),
        "release_based": release_based,
        "prerelease_topology_candidate": topology_candidate,
        "candidate_precedes_release": candidate_before_release,
        "time_order": candidate_order,
        "label_complete": not bool(release_missing_list),
    }


def _trace_lines(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for record in records
    )


def trace_content_sha256(records: Iterable[dict[str, Any]]) -> str:
    return hashlib.sha256(_trace_lines(records)).hexdigest()


def write_trace_gzip_atomic(path: str | Path, records: Iterable[dict[str, Any]]) -> str:
    """Atomically write gzip JSONL and return the published file hash."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [_json_safe(record) for record in records]
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                compressed.write(_trace_lines(rows))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SimulationStepRecorder:
    """Action/physics recorder for the Stage-2F collector."""

    def __init__(
        self,
        env: Any,
        task: str,
        *,
        event_instance_id: str,
        prefix_k: int = 0,
        operator: str = "clean_actor",
        actor_seed: int = 7,
        initial_object_z: float,
        trace_path: str | Path | None = None,
        baseline_contacts: Any = None,
    ) -> None:
        self.env = env
        self.task = task
        self.event_instance_id = str(event_instance_id)
        self.prefix_k = int(prefix_k)
        self.operator = str(operator)
        self.actor_seed = int(actor_seed)
        self.initial_object_z = float(initial_object_z)
        self.trace_path = None if trace_path is None else Path(trace_path)
        self.baseline_contacts = baseline_contacts
        self.records: list[dict[str, Any]] = []
        self.action_records: list[dict[str, Any]] = []
        self.physics_records: list[dict[str, Any]] = []
        self.physics_step_count = 0
        self.physics_instrumented = False
        self.physics_probe_error: str | None = None
        self.trace_sha256: str | None = None
        self._pending: dict[str, Any] | None = None
        self._local_substep = 0
        self._closed = False
        self._patches: list[tuple[Any, str, Any]] = []
        self._install_physics_probe()

    def _install_physics_probe(self) -> None:
        targets: list[Any] = []
        for candidate in (
            getattr(self.env, "sim", None),
            getattr(getattr(self.env, "env", None), "sim", None),
        ):
            if candidate is not None and all(candidate is not item for item in targets):
                targets.append(candidate)
        for target in targets:
            original = getattr(target, "step", None)
            if not callable(original):
                continue

            def wrapped(*args: Any, _original=original, **kwargs: Any) -> Any:
                result = _original(*args, **kwargs)
                if self._pending is not None:
                    self.physics_step_count += 1
                    self._local_substep += 1
                    pending = self._pending
                    row = record_step(
                        self.env,
                        self.task,
                        event_instance_id=self.event_instance_id,
                        prefix_k=self.prefix_k,
                        operator=self.operator,
                        actor_seed=self.actor_seed,
                        step=int(pending["step"]),
                        action=pending["action"],
                        previous_gripper=float(pending["previous_gripper"]),
                        initial_object_z=self.initial_object_z,
                        ever_lifted=bool(pending["ever_lifted"]),
                        baseline_contacts=self.baseline_contacts,
                        step_kind="physics_step",
                        physics_step_index=self.physics_step_count,
                        control_step=int(pending["step"]),
                        substep=int(self._local_substep),
                    )
                    self.physics_records.append(row)
                    self.records.append(row)
                return result

            try:
                setattr(target, "step", wrapped)
                self._patches.append((target, "step", original))
                self.physics_instrumented = True
            except (AttributeError, TypeError, RuntimeError) as exc:
                self.physics_probe_error = f"{type(exc).__name__}: {exc}"

    def set_action(self, step: int, action: Any, previous_gripper: float, ever_lifted: bool) -> None:
        self._local_substep = 0
        self._pending = {
            "step": int(step),
            "action": np.asarray(action, dtype=np.float32).copy(),
            "previous_gripper": float(previous_gripper),
            "ever_lifted": bool(ever_lifted),
        }

    def finish_action(self) -> dict[str, Any]:
        if self._pending is None:
            raise RuntimeError("finish_action called without set_action")
        pending = self._pending
        row = record_step(
            self.env,
            self.task,
            event_instance_id=self.event_instance_id,
            prefix_k=self.prefix_k,
            operator=self.operator,
            actor_seed=self.actor_seed,
            step=int(pending["step"]),
            action=pending["action"],
            previous_gripper=float(pending["previous_gripper"]),
            initial_object_z=self.initial_object_z,
            ever_lifted=bool(pending["ever_lifted"]),
            baseline_contacts=self.baseline_contacts,
            control_step=int(pending["step"]),
            substep=0,
        )
        row["physics_steps_before_action_record"] = int(
            sum(1 for item in self.physics_records if item["step"] == int(pending["step"]))
        )
        self.action_records.append(row)
        self.records.append(row)
        self._pending = None
        return row

    def close(self) -> str | None:
        if self._closed:
            return self.trace_sha256
        self._pending = None
        for target, name, original in reversed(self._patches):
            try:
                setattr(target, name, original)
            except (AttributeError, TypeError, RuntimeError):
                pass
        self._patches.clear()
        if self.trace_path is not None:
            self.trace_sha256 = write_trace_gzip_atomic(self.trace_path, self.records)
        self._closed = True
        return self.trace_sha256

    def __enter__(self) -> "SimulationStepRecorder":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


__all__ = [
    "SimulationStepRecorder", "label_trace", "record_step",
    "trace_content_sha256", "write_trace_gzip_atomic",
]

