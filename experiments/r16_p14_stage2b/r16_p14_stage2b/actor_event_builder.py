from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from r16_p14_stage2a.envs import contact_pairs, contact_sha256, joint_qpos, restore_state, state_sha256
from r16_p14_stage2a.settings import TASK_SPECS, TASK_TO_INDEX

from .io_utils import (
    atomic_write_json,
    atomic_write_text,
    load_jsonl,
    sha256_array,
    write_once_json,
    write_once_jsonl,
)
from .runtime import (
    ActorBundle,
    ActorHistory,
    TaskMonitor,
    chunk_hash,
    gripper_release_index,
    init_state_and_suite,
    nominal_chunk_trace,
    target_distance,
)
from .settings import (
    ACTOR_QUALIFICATION_IDS,
    ACTOR_SEEDS,
    ARTIFACT_ROOT,
    ATLAS_EVALUATION_IDS,
    CHUNK_LENGTH,
    EXPERIMENT_ROOT,
    FALLBACK_ANALYSIS_HORIZON,
    PERTURBATION_CALIBRATION_IDS,
    PRIMARY_TASKS,
    RESERVED_IDS,
)


def split_name(init_id: int) -> str:
    if init_id in ACTOR_QUALIFICATION_IDS:
        return "actor_qualification"
    if init_id in PERTURBATION_CALIBRATION_IDS:
        return "perturbation_calibration"
    if init_id in ATLAS_EVALUATION_IDS:
        return "atlas_pilot_evaluation"
    if init_id in RESERVED_IDS:
        return "reserved_uninspected"
    raise ValueError(init_id)


def effective_horizon() -> tuple[int, bool, int | None]:
    summary_path = ARTIFACT_ROOT / "chunk_executability/summary.json"
    summary = json.loads(summary_path.read_text())
    h_valid = summary["H_valid"]
    if h_valid is not None and int(h_valid) >= 8:
        return int(h_valid), False, int(h_valid)
    return FALLBACK_ANALYSIS_HORIZON, True, h_valid


def build_event(
    env,
    nominal_env,
    *,
    task: str,
    init_id: int,
    init_state: np.ndarray,
    bundle: ActorBundle,
    horizon: int,
    upstream_override: bool,
    h_valid: int | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    spec = TASK_SPECS[task]
    assert spec.manipulated_joint
    env.reset()
    observation = restore_state(env, init_state)
    history = ActorHistory.initial(observation)
    monitor = TaskMonitor.create(env, task)
    pre_anchor_actions: list[np.ndarray] = []
    best_distance = target_distance(env, task, monitor.goal)
    last_reason = "late_phase_not_reached"
    success = bool(env.check_success())
    for global_step in range(spec.horizon):
        if success:
            last_reason = "task_succeeded_before_eligible_anchor"
            break
        chunk = bundle.predict(history.state_array(), history.action_array(), task)
        current_distance = target_distance(env, task, monitor.goal)
        best_distance = min(best_distance, current_distance)
        base = {
            "stable_lift": monitor.ever_stably_lifted,
            "gripper_closed": monitor.previous_gripper > 0.0,
            "at_least_effective_horizon_actions": len(chunk) >= horizon,
            "task_not_successful": not success,
            "target_distance": current_distance <= (0.16 if task == PRIMARY_TASKS[0] else 0.18),
            "no_current_blocker_contact": True,
            "plausible_late_phase": False,
        }
        if task == PRIMARY_TASKS[0]:
            base["plausible_late_phase"] = (
                gripper_release_index(chunk, monitor.previous_gripper, horizon) is not None
            )
        else:
            from r16_p14_stage2a.mechanics import cause_contact

            base["no_current_blocker_contact"] = not cause_contact(env, spec)
        trace = None
        if all(value for key, value in base.items() if key != "plausible_late_phase"):
            anchor_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
            trace = nominal_chunk_trace(
                nominal_env,
                task=task,
                anchor_state=anchor_state,
                chunk=chunk,
                horizon=horizon,
                goal=monitor.goal,
            )
            if task == PRIMARY_TASKS[1]:
                base["plausible_late_phase"] = trace["minimum_distance"] <= 0.12
        if all(base.values()):
            assert trace is not None
            anchor_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
            states = history.state_array()
            actions = history.action_array()
            contacts = contact_pairs(env)
            event_id = f"{task}__seed{bundle.seed:02d}__init{init_id:02d}"
            hashes = history.hashes()
            event = {
                "schema_version": 1,
                "event_id": event_id,
                "task": task,
                "actor_seed": bundle.seed,
                "checkpoint": str(bundle.checkpoint),
                "checkpoint_sha256": bundle.checkpoint_sha256,
                "actor_run_id": bundle.payload["run_id"],
                "actor_task_id": TASK_TO_INDEX[task],
                "init_state_id": init_id,
                "init_state_hash": sha256_array(init_state, np.float64),
                "init_state": np.asarray(init_state, dtype=np.float64).tolist(),
                "split": split_name(init_id),
                "split_label": "perturbation-selection-split",
                "policy_held_out": False,
                "pre_anchor_actions": np.asarray(pre_anchor_actions, dtype=np.float32).reshape(-1, 7).tolist(),
                "pre_anchor_actions_hash": sha256_array(
                    np.asarray(pre_anchor_actions, dtype=np.float32).reshape(-1, 7), np.float32
                ),
                "anchor_global_step": global_step,
                "anchor_state": anchor_state.tolist(),
                "anchor_state_hash": state_sha256(anchor_state),
                "state_history": states.tolist(),
                "action_history": actions.tolist(),
                **hashes,
                "original_chunk": np.asarray(chunk, dtype=np.float32).tolist(),
                "original_chunk_hash": chunk_hash(chunk),
                "original_chunk_length": CHUNK_LENGTH,
                "effective_horizon": horizon,
                "H_valid_at_build": h_valid,
                "upstream_gate_override": upstream_override,
                "phase": {
                    "ever_stably_lifted": bool(monitor.ever_stably_lifted),
                    "late_phase_reached": bool(monitor.late_phase_reached),
                    "current_gripper": float(monitor.previous_gripper),
                    "target_distance": current_distance,
                    "eligibility": base,
                },
                "initial_manipulated_qpos": monitor.initial_object.tolist(),
                "anchor_manipulated_qpos": np.asarray(
                    joint_qpos(env, spec.manipulated_joint), dtype=np.float64
                ).tolist(),
                "goal_qpos": np.asarray(monitor.goal, dtype=np.float64).tolist(),
                "contact_pairs": contacts,
                "contact_hash": contact_sha256(contacts),
                "nominal_trace_manipulated_qpos": trace["positions"].tolist(),
                "nominal_trace_target_distances": trace["distances"].tolist(),
                "nominal_future_object_qpos": trace["best_position"].tolist(),
                "nominal_minimum_target_distance": trace["minimum_distance"],
                "nominal_final_target_distance": trace["final_distance"],
                "nominal_chunk_success": bool(trace["success"]),
                "task_already_successful": False,
                "source_is_actor_generated_chunk": True,
                "source_is_demonstration_chunk": False,
            }
            return event, {
                "event_id": event_id,
                "task": task,
                "actor_seed": bundle.seed,
                "init_state_id": init_id,
                "split": split_name(init_id),
                "eligible": True,
                "reason": "eligible",
                "anchor_global_step": global_step,
                "best_target_distance": best_distance,
                "task_success_before_anchor": False,
            }
        failed = [key for key, value in base.items() if not value]
        last_reason = ",".join(failed)
        action = np.asarray(chunk[0], dtype=np.float32)
        observation, _, _, _ = env.step(action)
        history.update(observation, action)
        pre_anchor_actions.append(action.copy())
        success = bool(env.check_success())
        monitor.observe(env, action, success)
    return None, {
        "event_id": f"{task}__seed{bundle.seed:02d}__init{init_id:02d}",
        "task": task,
        "actor_seed": bundle.seed,
        "init_state_id": init_id,
        "split": split_name(init_id),
        "eligible": False,
        "reason": last_reason,
        "anchor_global_step": None,
        "best_target_distance": best_distance,
        "task_success_before_anchor": success,
        "grasp_lift_reached": bool(monitor.ever_stably_lifted),
        "late_phase_reached": bool(monitor.late_phase_reached),
        "object_drop": bool(monitor.object_drop),
        "wrong_release": bool(monitor.wrong_release),
        "phase_regression": bool(monitor.phase_regression),
    }


def event_checkpoint_path(output: Path, seed: int, task: str, init_id: int) -> Path:
    return output / "episode_checkpoints" / f"seed_{seed}" / f"{task}__init{init_id:02d}.json"


def consolidate_seed_checkpoints(
    output: Path, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    events: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for task in PRIMARY_TASKS:
        for init_id in range(30):
            checkpoint = event_checkpoint_path(output, seed, task, init_id)
            if not checkpoint.is_file():
                return None
            saved = json.loads(checkpoint.read_text())
            attempts.append(saved["attempt"])
            if saved["event"] is not None:
                events.append(saved["event"])
    return events, attempts


def run_seed(seed: int, device: str, tasks: tuple[str, ...] = PRIMARY_TASKS) -> None:
    output = ARTIFACT_ROOT / "actor_events/shards"
    events_path = output / f"seed_{seed}.events.jsonl"
    attempts_path = output / f"seed_{seed}.attempts.jsonl"
    if events_path.is_file() and attempts_path.is_file():
        print(f"EVENT_SEED_ALREADY_COMPLETE seed={seed}")
        return
    horizon, override, h_valid = effective_horizon()
    bundle = ActorBundle.load(seed, device)
    for task in tasks:
        env, _, init_states = init_state_and_suite(task)
        nominal_env, _, _ = init_state_and_suite(task)
        try:
            for init_id in range(30):
                checkpoint = event_checkpoint_path(output, seed, task, init_id)
                if checkpoint.is_file():
                    saved = json.loads(checkpoint.read_text())
                    print(
                        f"ACTOR_EVENT_RESUME seed={seed} task={task} init={init_id} "
                        f"eligible={int(saved['event'] is not None)}",
                        flush=True,
                    )
                    continue
                try:
                    event, attempt = build_event(
                        env,
                        nominal_env,
                        task=task,
                        init_id=init_id,
                        init_state=init_states[init_id],
                        bundle=bundle,
                        horizon=horizon,
                        upstream_override=override,
                        h_valid=h_valid,
                    )
                    write_once_json(
                        checkpoint,
                        {"event": event, "attempt": attempt, "complete": True},
                    )
                    print(
                        f"ACTOR_EVENT seed={seed} task={task} init={init_id} "
                        f"eligible={int(event is not None)} reason={attempt['reason']}",
                        flush=True,
                    )
                except Exception as exc:
                    attempt = {
                            "event_id": f"{task}__seed{seed:02d}__init{init_id:02d}",
                            "task": task,
                            "actor_seed": seed,
                            "init_state_id": init_id,
                            "split": split_name(init_id),
                            "eligible": False,
                            "reason": f"error:{type(exc).__name__}:{exc}",
                            "traceback": traceback.format_exc(),
                        }
                    write_once_json(
                        checkpoint,
                        {"event": None, "attempt": attempt, "complete": True},
                    )
        finally:
            env.close()
            nominal_env.close()
    consolidated = consolidate_seed_checkpoints(output, seed)
    if consolidated is None:
        print(f"EVENT_SEED_PARTIAL seed={seed} tasks={','.join(tasks)}", flush=True)
        return
    events, attempts = consolidated
    write_once_jsonl(events_path, events)
    write_once_jsonl(attempts_path, attempts)
    atomic_write_json(
        output / f"seed_{seed}.complete.json",
        {
            "seed": seed,
            "event_count": len(events),
            "attempt_count": len(attempts),
            "effective_horizon": horizon,
            "H_valid": h_valid,
            "upstream_gate_override": override,
        },
    )


def aggregate() -> dict[str, Any]:
    shard = ARTIFACT_ROOT / "actor_events/shards"
    events: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for seed in ACTOR_SEEDS:
        events.extend(load_jsonl(shard / f"seed_{seed}.events.jsonl"))
        attempts.extend(load_jsonl(shard / f"seed_{seed}.attempts.jsonl"))
    expected_attempts = len(PRIMARY_TASKS) * len(ACTOR_SEEDS) * 30
    if len(attempts) != expected_attempts:
        raise ValueError(f"expected {expected_attempts} attempts, found {len(attempts)}")
    if any(int(record["init_state_id"]) in RESERVED_IDS for record in attempts):
        raise ValueError("reserved init state was inspected")
    freeze = json.loads((ARTIFACT_ROOT / "source_freeze/task_init_states.json").read_text())
    split_manifest = {
        "schema_version": 1,
        "label": "perturbation-selection-splits",
        "policy_held_out": False,
        "splits": {
            "actor_qualification": list(ACTOR_QUALIFICATION_IDS),
            "perturbation_calibration": list(PERTURBATION_CALIBRATION_IDS),
            "atlas_pilot_evaluation": list(ATLAS_EVALUATION_IDS),
            "reserved_uninspected": list(RESERVED_IDS),
        },
        "pairwise_disjoint": True,
        "reserved_outcomes_inspected": False,
        "tasks": {
            task: {
                "available_init_states": freeze[task]["available_count"],
                "first_50_distinct": freeze[task]["first_50_distinct"],
                "first_50_hash": freeze[task]["combined_first_50_sha256"],
            }
            for task in PRIMARY_TASKS
        },
    }
    counts: dict[str, Any] = {}
    for task in PRIMARY_TASKS:
        task_events = [record for record in events if record["task"] == task]
        counts[task] = {
            "total": len(task_events),
            "by_split": dict(Counter(record["split"] for record in task_events)),
            "by_seed": dict(Counter(str(record["actor_seed"]) for record in task_events)),
            "attempts": len([record for record in attempts if record["task"] == task]),
            "eligible_rate": len(task_events) / 90.0,
        }
    summary = {
        "schema_version": 1,
        "event_count": len(events),
        "attempt_count": len(attempts),
        "reserved_outcomes_inspected": False,
        "source_actor_chunk_only": all(record["source_is_actor_generated_chunk"] for record in events),
        "demonstration_nominal_chunk_count": sum(record["source_is_demonstration_chunk"] for record in events),
        "tasks": counts,
        "event_split_blocked": any(not freeze[task]["first_50_distinct"] for task in PRIMARY_TASKS),
        "failure_label": (
            "BLOCKED_BY_EVENT_SPLIT"
            if any(not freeze[task]["first_50_distinct"] for task in PRIMARY_TASKS)
            else None
        ),
        "early_stop_applied": False,
    }
    output = ARTIFACT_ROOT / "actor_events"
    write_once_jsonl(output / "events.jsonl", events)
    write_once_jsonl(output / "attempts.jsonl", attempts)
    atomic_write_json(output / "split_manifest.json", split_manifest)
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(EXPERIMENT_ROOT / "actor_events/summary.json", summary)
    lines = [
        "# Phase B — actor-generated event pool",
        "",
        f"Eligible actor events: **{len(events)} / {len(attempts)}** attempted closed-loop rollouts.",
        "Reserved init IDs 30--49 were hashed during source freeze and never rolled out.",
        "",
        "| task | eligible | qualification | calibration | evaluation |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in PRIMARY_TASKS:
        row = counts[task]
        by_split = row["by_split"]
        lines.append(
            f"| {task} | {row['total']} | {by_split.get('actor_qualification', 0)} | "
            f"{by_split.get('perturbation_calibration', 0)} | {by_split.get('atlas_pilot_evaluation', 0)} |"
        )
    lines.append("")
    report = "\n".join(lines)
    atomic_write_text(output / "report.md", report)
    atomic_write_text(EXPERIMENT_ROOT / "actor_events/report.md", report)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=ACTOR_SEEDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", choices=PRIMARY_TASKS)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    elif args.seed is not None:
        run_seed(args.seed, args.device, (args.task,) if args.task else PRIMARY_TASKS)
    else:
        parser.error("provide --seed or --aggregate")


if __name__ == "__main__":
    main()
