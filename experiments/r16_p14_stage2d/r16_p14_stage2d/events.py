from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r16_p14_stage2a.envs import (
    contact_pairs,
    joint_qpos,
    make_env,
    restore_state,
    state_sha256,
)
from r16_p14_stage2a.mechanics import cause_contact
from r16_p14_stage2a.settings import ACTION_DIM, TASK_SPECS
from r16_p14_stage2b.runtime import ActorBundle, ActorHistory, chunk_hash, gripper_release_index
from r16_p14_stage2c.goal_geometry import goal_geometry, target_distance

from .geometry import live_bounding_sphere
from .io_utils import atomic_write_json, atomic_write_jsonl, sha256_array, sha256_file
from .settings import (
    ACTOR_SEEDS,
    ARTIFACT_ROOT,
    CALIBRATION_IDS,
    EVALUATION_IDS,
    EXPERIMENT_ROOT,
    H_VALID,
    INFRASTRUCTURE_IDS,
    MIRROR_EXPERIMENT_OUTPUTS,
    PATH_FUTURE_INDICES,
    RESERVE_IDS,
    TARGET_SHIFT_TASK,
    TASKS,
)


MEASURABLE_TRANSPORT_M = 0.010


def split_ids(split: str) -> tuple[int, ...]:
    if split == "infrastructure":
        return INFRASTRUCTURE_IDS
    if split == "calibration":
        return CALIBRATION_IDS
    if split == "evaluation":
        return EVALUATION_IDS
    if split == "reserve":
        raise PermissionError("Stage-2D may not read reserve IDs 80-99")
    raise ValueError(split)


def load_init_state(task: str, init_id: int) -> tuple[np.ndarray, str]:
    if init_id in RESERVE_IDS:
        raise PermissionError("Stage-2D may not read reserve IDs 80-99")
    path = ARTIFACT_ROOT / "init_pool/init_states.npz"
    with np.load(path, allow_pickle=False) as pool:
        state = np.ascontiguousarray(pool[task][init_id], dtype=np.float64)
    return state, sha256_file(path)


def stable_grasp(env, task: str) -> bool:
    spec = TASK_SPECS[task]
    assert spec.manipulated_joint
    object_name = spec.manipulated_joint.removesuffix("_joint0")
    return bool(
        env.env._check_grasp(
            env.env.robots[0].gripper,
            env.env.objects_dict[object_name],
        )
    )


def nominal_trace(env, task: str, anchor_state: np.ndarray, chunk: np.ndarray) -> dict[str, Any]:
    spec = TASK_SPECS[task]
    assert spec.manipulated_joint
    env.reset()
    restore_state(env, anchor_state)
    positions = [joint_qpos(env, spec.manipulated_joint)]
    distances = [target_distance(env, task)]
    contacts = [list(contact_pairs(env))]
    success = [bool(env.check_success())]
    for action in np.asarray(chunk, dtype=np.float32)[:H_VALID]:
        env.step(action)
        positions.append(joint_qpos(env, spec.manipulated_joint))
        distances.append(target_distance(env, task))
        contacts.append(list(contact_pairs(env)))
        success.append(bool(env.check_success()))
    positions_array = np.asarray(positions, dtype=np.float64)
    displacements = np.linalg.norm(positions_array[:, :3] - positions_array[0, :3], axis=1)
    return {
        "positions": positions_array.tolist(),
        "distances": np.asarray(distances, dtype=np.float64).tolist(),
        "contacts": contacts,
        "success": success,
        "displacements_m": displacements.tolist(),
        "maximum_displacement_m": float(displacements.max()),
        "final_displacement_m": float(displacements[-1]),
        "minimum_goal_distance_m": float(min(distances)),
    }


def eligible(
    env,
    task: str,
    chunk: np.ndarray,
    previous_gripper: float,
    stable_lift_count: int,
    anchor_object: np.ndarray,
    nominal: dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    spec = TASK_SPECS[task]
    grasp = stable_grasp(env, task)
    closed = previous_gripper > 0.0
    task_not_successful = not bool(env.check_success())
    release = gripper_release_index(chunk, previous_gripper, H_VALID)
    common = {
        "stable_grasp": grasp,
        "stable_lift_count": stable_lift_count,
        "stable_lift_two_steps": stable_lift_count >= 2,
        "gripper_closed": closed,
        "task_not_successful": task_not_successful,
        "release_index": release,
        "global_step_fallback_used": False,
    }
    if task == TARGET_SHIFT_TASK:
        release_in_range = release is not None and 6 <= release <= 13
        progress = (
            float(nominal["displacements_m"][release + 1])
            if release is not None and release + 1 < len(nominal["displacements_m"])
            else 0.0
        )
        checks = {
            **common,
            "release_in_preregistered_range": release_in_range,
            "actions_between_detection_and_release_min3": release is not None and release - 2 >= 3,
            "nominal_transport_progress_m": progress,
            "measurable_transport_progress": progress >= MEASURABLE_TRANSPORT_M,
            "remaining_chunk_length": int(len(chunk)),
        }
        passed = all(
            [
                grasp,
                stable_lift_count >= 2,
                closed,
                task_not_successful,
                release_in_range,
                checks["actions_between_detection_and_release_min3"],
                checks["measurable_transport_progress"],
                len(chunk) == H_VALID,
            ]
        )
        return passed, "strict_release_event" if passed else "strict_target_criteria_not_met", checks

    no_contact = not cause_contact(env, spec)
    current = np.asarray(anchor_object[:3], dtype=np.float64)
    positions = np.asarray(nominal["positions"], dtype=np.float64)
    future_distances = {
        str(index): float(np.linalg.norm(positions[index, :3] - current))
        for index in PATH_FUTURE_INDICES
        if index < len(positions)
    }
    valid_future = [
        index for index in PATH_FUTURE_INDICES if future_distances.get(str(index), 0.0) >= 0.05
    ]
    checks = {
        **common,
        "no_current_bowl_blocker_contact": no_contact,
        "nominal_displacement_m": float(nominal["maximum_displacement_m"]),
        "nominal_displacement_min_006": float(nominal["maximum_displacement_m"]) >= 0.06,
        "future_point_distances_m": future_distances,
        "valid_future_indices": valid_future,
        "future_path_point_exists": bool(valid_future),
        "actions_before_future_event_min3": bool(valid_future) and min(valid_future) - 2 >= 3,
        "remaining_chunk_length": int(len(chunk)),
    }
    passed = all(
        [
            grasp,
            stable_lift_count >= 2,
            closed,
            task_not_successful,
            no_contact,
            checks["nominal_displacement_min_006"],
            checks["future_path_point_exists"],
            checks["actions_before_future_event_min3"],
            len(chunk) == H_VALID,
        ]
    )
    return passed, "strict_future_path_event" if passed else "strict_path_criteria_not_met", checks


def build_event(task: str, actor_seed: int, init_id: int, split: str, device: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    torch.set_num_threads(1)
    bundle = ActorBundle.load(actor_seed, device)
    init_state, pool_file_hash = load_init_state(task, init_id)
    spec = TASK_SPECS[task]
    assert spec.manipulated_joint
    env, _ = make_env(task, seed=0)
    trace_env, _ = make_env(task, seed=0)
    try:
        observation = restore_state(env, init_state)
        history = ActorHistory.initial(observation)
        initial_object = joint_qpos(env, spec.manipulated_joint)
        pre_actions: list[np.ndarray] = []
        stable_lift_count = 0
        candidate_count = 0
        last_checks: dict[str, Any] = {}
        for global_step in range(spec.horizon - H_VALID):
            if env.check_success():
                return None, {
                    "task": task,
                    "actor_seed": actor_seed,
                    "init_state_id": init_id,
                    "split": split,
                    "eligible": False,
                    "reason": "task_success_before_strict_anchor",
                    "candidate_count": candidate_count,
                }
            states = np.ascontiguousarray(history.state_array(), dtype=np.float32)
            actions = np.ascontiguousarray(history.action_array(), dtype=np.float32)
            chunk = np.ascontiguousarray(bundle.predict(states, actions, task), dtype=np.float32)
            current_object = joint_qpos(env, spec.manipulated_joint)
            lifted = bool(current_object[2] - initial_object[2] >= spec.lift_delta)
            grasp_now = stable_grasp(env, task)
            stable_lift_count = stable_lift_count + 1 if lifted and grasp_now else 0
            previous_gripper = float(history.actions[-1][-1])
            plausible = stable_lift_count >= 2 and previous_gripper > 0.0
            if plausible:
                candidate_count += 1
                anchor_state = np.ascontiguousarray(env.get_sim_state(), dtype=np.float64)
                nominal = nominal_trace(trace_env, task, anchor_state, chunk)
                is_eligible, reason, checks = eligible(
                    env,
                    task,
                    chunk,
                    previous_gripper,
                    stable_lift_count,
                    current_object,
                    nominal,
                )
                last_checks = checks
                if is_eligible:
                    event_id = f"{task}__{split}__seed{actor_seed:02d}__init{init_id:02d}"
                    target_qpos = (
                        joint_qpos(env, spec.target_joint).tolist() if spec.target_joint else None
                    )
                    obstacle_qpos = (
                        joint_qpos(env, spec.obstacle_joint).tolist() if spec.obstacle_joint else None
                    )
                    geometry: dict[str, Any] = {
                        "manipulated": live_bounding_sphere(env, spec.manipulated_joint)
                    }
                    if spec.target_joint:
                        geometry["target"] = live_bounding_sphere(env, spec.target_joint)
                    if spec.obstacle_joint:
                        geometry["obstacle"] = live_bounding_sphere(env, spec.obstacle_joint)
                    event = {
                        "schema_version": 1,
                        "event_id": event_id,
                        "task": task,
                        "split": split,
                        "actor_seed": actor_seed,
                        "checkpoint": str(bundle.checkpoint),
                        "checkpoint_sha256": bundle.checkpoint_sha256,
                        "actor_run_id": bundle.payload["run_id"],
                        "init_state_id": init_id,
                        "init_pool_file_sha256": pool_file_hash,
                        "init_state": init_state.tolist(),
                        "init_state_hash": sha256_array(init_state, np.float64),
                        "pre_anchor_actions": np.asarray(pre_actions, dtype=np.float32).reshape(-1, ACTION_DIM).tolist(),
                        "pre_anchor_actions_hash": sha256_array(
                            np.asarray(pre_actions, dtype=np.float32).reshape(-1, ACTION_DIM), np.float32
                        ),
                        "anchor_global_step": global_step,
                        "anchor_state": anchor_state.tolist(),
                        "anchor_state_hash": state_sha256(anchor_state),
                        "state_history": states.tolist(),
                        "state_history_hash": sha256_array(states, np.float32),
                        "action_history": actions.tolist(),
                        "action_history_hash": sha256_array(actions, np.float32),
                        "original_chunk": chunk.tolist(),
                        "original_chunk_hash": chunk_hash(chunk),
                        "chunk_length": H_VALID,
                        "release_index": checks.get("release_index"),
                        "predicted_path_indices": checks.get("valid_future_indices", []),
                        "initial_manipulated_qpos": initial_object.tolist(),
                        "anchor_manipulated_qpos": current_object.tolist(),
                        "anchor_target_qpos": target_qpos,
                        "anchor_obstacle_qpos": obstacle_qpos,
                        "anchor_contacts": list(contact_pairs(env)),
                        "goal_geometry": goal_geometry(env, task),
                        "object_geometry": geometry,
                        "task_phase": checks,
                        "nominal_object_trajectory": nominal,
                        "source_is_actor_generated_chunk": True,
                        "source_is_demonstration_chunk": False,
                        "global_step_fallback_used": False,
                        "method_outcomes_read_during_construction": False,
                    }
                    return event, {
                        "task": task,
                        "actor_seed": actor_seed,
                        "init_state_id": init_id,
                        "split": split,
                        "eligible": True,
                        "reason": reason,
                        "anchor_global_step": global_step,
                        "candidate_count": candidate_count,
                    }
            action = np.ascontiguousarray(chunk[0], dtype=np.float32)
            observation, _, _, _ = env.step(action)
            history.update(observation, action)
            pre_actions.append(action.copy())
        return None, {
            "task": task,
            "actor_seed": actor_seed,
            "init_state_id": init_id,
            "split": split,
            "eligible": False,
            "reason": "no_strict_actor_event",
            "candidate_count": candidate_count,
            "last_checks": last_checks,
            "global_step_fallback_used": False,
        }
    finally:
        env.close()
        trace_env.close()


def shard_path(task: str, seed: int, split: str, init_id: int) -> Path:
    return (
        ARTIFACT_ROOT
        / "actor_events/shards"
        / split
        / f"seed_{seed}"
        / f"{task}__init{init_id:02d}.json"
    )


def run_shard(
    task: str,
    seed: int,
    split: str,
    device: str,
    init_ids: tuple[int, ...] | None = None,
) -> None:
    for init_id in init_ids or split_ids(split):
        if init_id not in split_ids(split):
            raise ValueError(f"init id {init_id} does not belong to {split}")
        path = shard_path(task, seed, split, init_id)
        if path.is_file():
            print(f"EVENT_RESUME task={task} seed={seed} split={split} id={init_id}", flush=True)
            continue
        try:
            event, attempt = build_event(task, seed, init_id, split, device)
            payload = {"complete": True, "event": event, "attempt": attempt}
        except Exception as exc:
            payload = {
                "complete": True,
                "event": None,
                "attempt": {
                    "task": task,
                    "actor_seed": seed,
                    "init_state_id": init_id,
                    "split": split,
                    "eligible": False,
                    "reason": "exception",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            }
        atomic_write_json(path, payload)
        print(
            f"EVENT_DONE task={task} seed={seed} split={split} id={init_id} "
            f"eligible={int(payload['event'] is not None)}",
            flush=True,
        )


def consolidate(splits: tuple[str, ...]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    missing: list[str] = []
    for split in splits:
        for task in TASKS:
            for seed in ACTOR_SEEDS:
                for init_id in split_ids(split):
                    path = shard_path(task, seed, split, init_id)
                    if not path.is_file():
                        missing.append(str(path.relative_to(ARTIFACT_ROOT)))
                        continue
                    payload = json.loads(path.read_text())
                    attempts.append(payload["attempt"])
                    if payload["event"] is not None:
                        events.append(payload["event"])
    output = ARTIFACT_ROOT / "actor_events"
    atomic_write_jsonl(output / "events.jsonl", events)
    atomic_write_jsonl(output / "attempts.jsonl", attempts)
    counts = {
        split: {
            task: {
                "eligible": sum(e["split"] == split and e["task"] == task for e in events),
                "by_seed": {
                    str(seed): sum(
                        e["split"] == split
                        and e["task"] == task
                        and int(e["actor_seed"]) == seed
                        for e in events
                    )
                    for seed in ACTOR_SEEDS
                },
                "distinct_init_ids": len(
                    {
                        int(e["init_state_id"])
                        for e in events
                        if e["split"] == split and e["task"] == task
                    }
                ),
            }
            for task in TASKS
        }
        for split in splits
    }
    summary = {
        "schema_version": 1,
        "splits": list(splits),
        "completed_attempts": len(attempts),
        "eligible_events": len(events),
        "missing_shards": missing,
        "error_count": sum(bool(row.get("error")) for row in attempts),
        "counts": counts,
        "availability_gate": {
            split: {
                task: (
                    counts[split][task]["eligible"] >= 40
                    and min(counts[split][task]["by_seed"].values()) >= 10
                    and counts[split][task]["distinct_init_ids"] >= 30
                )
                for task in TASKS
            }
            for split in splits
            if split == "calibration"
        },
        "global_step_fallback_events": sum(bool(e["global_step_fallback_used"]) for e in events),
        "demonstration_events": sum(bool(e["source_is_demonstration_chunk"]) for e in events),
        "method_outcomes_read": False,
    }
    atomic_write_json(output / "summary.json", summary)
    report = (
        "# Strict actor-generated events\n\n"
        "Events use the frozen ACT checkpoints and fresh reset-randomized init pool. No global-step "
        "fallback and no demonstration chunk is allowed. Eligibility reads opportunity geometry only, "
        "not cached/fresh/replan outcomes.\n\n"
        f"Eligible events: {len(events)}; errors: {summary['error_count']}; missing shards: {len(missing)}.\n"
    )
    (output / "report.md").write_text(report)
    if MIRROR_EXPERIMENT_OUTPUTS:
        mirror = EXPERIMENT_ROOT / "actor_events"
        mirror.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "report.md"):
            (mirror / name).write_bytes((output / name).read_bytes())
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--seed", type=int, choices=ACTOR_SEEDS)
    parser.add_argument("--split", choices=("infrastructure", "calibration", "evaluation"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init-id", action="append", type=int)
    parser.add_argument("--consolidate", action="store_true")
    parser.add_argument("--consolidate-splits", default="infrastructure,calibration,evaluation")
    args = parser.parse_args()
    if args.consolidate:
        consolidate(tuple(item for item in args.consolidate_splits.split(",") if item))
        return
    if args.task is None or args.seed is None or args.split is None:
        parser.error("--task, --seed and --split are required")
    run_shard(
        args.task,
        args.seed,
        args.split,
        args.device,
        None if args.init_id is None else tuple(args.init_id),
    )


if __name__ == "__main__":
    main()
