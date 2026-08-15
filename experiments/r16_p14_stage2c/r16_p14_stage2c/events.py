from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from r16_p14_stage2a.envs import contact_pairs, joint_qpos, restore_state, state_sha256
from r16_p14_stage2a.settings import TASK_SPECS
from r16_p14_stage2b.io_utils import atomic_write_json, load_jsonl, sha256_array, write_once_jsonl
from r16_p14_stage2b.runtime import ActorBundle, ActorHistory, chunk_hash, gripper_release_index

from .contracts import admit_event_replay
from .goal_geometry import goal_geometry, target_distance
from .runtime import (
    controller_snapshot,
    first_trace_divergence,
    nominal_chunk_trace,
    reconstruct_anchor,
    rng_snapshot,
    runtime_trace_row,
)
from .settings import ACTOR_SEEDS, ALL_CANDIDATE_TASKS, ARTIFACT_ROOT, CALIBRATION_IDS, EVALUATION_IDS, H_VALID


def split_ids(split: str) -> tuple[int, ...]:
    if split == "calibration":
        return CALIBRATION_IDS
    if split == "evaluation":
        return EVALUATION_IDS
    raise ValueError(split)


def eligibility(task: str, global_step: int, ever_lifted: bool, chunk: np.ndarray, current_distance: float) -> tuple[bool, str]:
    release = gripper_release_index(chunk, 1.0, H_VALID)
    if task == "put_the_cream_cheese_in_the_bowl":
        if ever_lifted and (release is not None or current_distance <= 0.15):
            return True, "lifted_release_or_target_approach"
        return (global_step >= 120), "deterministic_fallback_step120"
    if task == "push_the_plate_to_the_front_of_the_stove":
        return (global_step >= 40), "deterministic_push_phase_step40"
    if ever_lifted:
        return True, "stable_lift_phase"
    return (global_step >= 120), "deterministic_fallback_step120"


def build_event(task: str, seed: int, init_id: int, split: str, bundle: ActorBundle) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    from r16_p14_stage2b.runtime import init_state_and_suite

    spec = TASK_SPECS[task]
    assert spec.manipulated_joint
    env, _, init_states = init_state_and_suite(task)
    nominal_env, _, _ = init_state_and_suite(task)
    try:
        init_state = np.ascontiguousarray(init_states[init_id], dtype=np.float64)
        env.seed(0)
        env.reset()
        observation = restore_state(env, init_state)
        history = ActorHistory.initial(observation)
        initial_qpos = joint_qpos(env, spec.manipulated_joint)
        ever_lifted = False
        pre_actions: list[np.ndarray] = []
        trace = [runtime_trace_row(env, history, global_step=0, action=None)]
        best_distance = target_distance(env, task)
        for global_step in range(spec.horizon - H_VALID):
            if env.check_success():
                return None, {"task": task, "actor_seed": seed, "init_state_id": init_id, "split": split, "eligible": False, "reason": "task_success_before_anchor", "best_goal_distance": best_distance}
            states = history.state_array()
            actions = history.action_array()
            chunk = np.ascontiguousarray(bundle.predict(states, actions, task), dtype=np.float32)
            current_qpos = joint_qpos(env, spec.manipulated_joint)
            ever_lifted |= bool(current_qpos[2] - initial_qpos[2] >= spec.lift_delta)
            distance = target_distance(env, task)
            best_distance = min(best_distance, distance)
            eligible, reason = eligibility(task, global_step, ever_lifted, chunk, distance)
            if eligible:
                anchor_state = np.ascontiguousarray(env.get_sim_state(), dtype=np.float64)
                nominal = nominal_chunk_trace(nominal_env, task, anchor_state, chunk)
                if nominal["displacement_m"] >= 0.004 or global_step >= 120:
                    event_id = f"{task}__seed{seed:02d}__init{init_id:02d}"
                    geometry = goal_geometry(env, task)
                    event = {
                        "schema_version": 1,
                        "event_id": event_id,
                        "task": task,
                        "actor_seed": seed,
                        "checkpoint": str(bundle.checkpoint),
                        "checkpoint_sha256": bundle.checkpoint_sha256,
                        "actor_run_id": bundle.payload["run_id"],
                        "init_state_id": init_id,
                        "split": split,
                        "init_state": init_state.tolist(),
                        "init_state_hash": sha256_array(init_state, np.float64),
                        "pre_anchor_actions": np.asarray(pre_actions, dtype=np.float32).reshape(-1, 7).tolist(),
                        "pre_anchor_actions_hash": sha256_array(np.asarray(pre_actions, dtype=np.float32).reshape(-1, 7), np.float32),
                        "pre_anchor_trace": trace,
                        "anchor_global_step": global_step,
                        "anchor_state": anchor_state.tolist(),
                        "anchor_state_hash": state_sha256(anchor_state),
                        "state_history": states.tolist(),
                        "state_history_hash": sha256_array(states, np.float32),
                        "action_history": actions.tolist(),
                        "action_history_hash": sha256_array(actions, np.float32),
                        "original_chunk": chunk.tolist(),
                        "original_chunk_hash": chunk_hash(chunk),
                        "effective_horizon": H_VALID,
                        "initial_manipulated_qpos": initial_qpos.tolist(),
                        "anchor_manipulated_qpos": current_qpos.tolist(),
                        "phase": {"ever_lifted": ever_lifted, "eligibility_reason": reason, "goal_distance": distance},
                        "goal_geometry": geometry,
                        "nominal_trace": {
                            "positions": nominal["positions"].tolist(),
                            "distances": nominal["distances"].tolist(),
                            "contacts": nominal["contacts"],
                            "displacement_m": nominal["displacement_m"],
                            "minimum_goal_distance": nominal["minimum_goal_distance"],
                        },
                        "anchor_contacts": list(contact_pairs(env)),
                        "anchor_controller": controller_snapshot(env),
                        "anchor_rng": rng_snapshot(env),
                        "source_is_actor_generated_chunk": True,
                        "source_is_demonstration_chunk": False,
                        "intervention_outcome_read_during_build": False,
                    }
                    return event, {"task": task, "actor_seed": seed, "init_state_id": init_id, "split": split, "eligible": True, "reason": reason, "anchor_global_step": global_step, "best_goal_distance": best_distance}
            action = np.asarray(chunk[0], dtype=np.float32)
            observation, _, _, _ = env.step(action)
            history.update(observation, action)
            pre_actions.append(action.copy())
            trace.append(runtime_trace_row(env, history, global_step=global_step + 1, action=action))
        return None, {"task": task, "actor_seed": seed, "init_state_id": init_id, "split": split, "eligible": False, "reason": "no_actor_generated_anchor", "best_goal_distance": best_distance}
    finally:
        env.close()
        nominal_env.close()


def replay_admission(event: dict[str, Any], bundle: ActorBundle) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts = []
    reconstructed = []
    for order_slot in range(3):
        try:
            env, _, audit = reconstruct_anchor(event, bundle, capture_trace=True)
            try:
                divergence = first_trace_divergence(event["pre_anchor_trace"], audit["trace"])
                row = {
                    "event_id": event["event_id"], "order_slot": order_slot, "error": None,
                    **audit["checks"], "branch_order_invariant": True,
                    "first_trace_divergence": divergence,
                }
                attempts.append(row)
                reconstructed.append(audit)
            finally:
                env.close()
        except Exception as exc:
            attempts.append({"event_id": event["event_id"], "order_slot": order_slot, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    if len(reconstructed) == 3:
        hashes = [
            (state_sha256(item["state"]), chunk_hash(item["chunk"]))
            for item in reconstructed
        ]
        invariant = len(set(hashes)) == 1
        for row in attempts:
            row["branch_order_invariant"] = invariant
    admission = admit_event_replay(attempts)
    return admission, attempts


def checkpoint_path(seed: int, task: str, split: str, init_id: int) -> Path:
    return ARTIFACT_ROOT / "actor_events/shards" / split / f"seed_{seed}" / f"{task}__init{init_id:02d}.json"


def run_shard(seed: int, task: str, split: str, device: str) -> None:
    bundle = ActorBundle.load(seed, device)
    for init_id in split_ids(split):
        path = checkpoint_path(seed, task, split, init_id)
        if path.is_file():
            print(f"EVENT_RESUME seed={seed} task={task} split={split} init={init_id}", flush=True)
            continue
        try:
            event, attempt = build_event(task, seed, init_id, split, bundle)
            admission, replay_attempts = replay_admission(event, bundle) if event is not None else ({"admitted": False, "failure_label": "NO_ACTOR_EVENT"}, [])
            if event is not None:
                event["replay_admission"] = admission
                event["replay_admission_attempts"] = replay_attempts
            payload = {"event": event if event is not None and admission["admitted"] else None, "candidate_event": event, "attempt": {**attempt, "admission": admission}, "complete": True}
        except Exception as exc:
            payload = {"event": None, "candidate_event": None, "attempt": {"task": task, "actor_seed": seed, "init_state_id": init_id, "split": split, "eligible": False, "reason": "exception", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}, "complete": True}
        atomic_write_json(path, payload)
        marker = ARTIFACT_ROOT / "first_completed_event.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker_payload = {
            "schema_version": 1,
            "status": "complete",
            "record_type": "completed_actor_event_attempt",
            "event_id": f"{task}__seed{seed:02d}__init{init_id:02d}",
            "task": task,
            "actor_seed": seed,
            "init_state_id": init_id,
            "split": split,
            "admitted": payload["event"] is not None,
            "registry_run_id": os.environ.get("PAI_CANARY_RUN_ID"),
            "uid": os.getuid(),
            "gid": os.getgid(),
        }
        try:
            descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "w") as handle:
                json.dump(marker_payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        print(f"EVENT_DONE seed={seed} task={task} split={split} init={init_id} admitted={int(payload['event'] is not None)}", flush=True)


def consolidate() -> None:
    output = ARTIFACT_ROOT / "actor_events"
    events, attempts, replay = [], [], []
    expected = 0
    for split in ("calibration", "evaluation"):
        for seed in ACTOR_SEEDS:
            for task in ALL_CANDIDATE_TASKS:
                for init_id in split_ids(split):
                    expected += 1
                    path = checkpoint_path(seed, task, split, init_id)
                    if not path.is_file():
                        continue
                    payload = json.loads(path.read_text())
                    attempts.append(payload["attempt"])
                    if payload["candidate_event"] is not None:
                        replay.extend(payload["candidate_event"].get("replay_admission_attempts", []))
                    if payload["event"] is not None:
                        events.append(payload["event"])
    if output.joinpath("events.jsonl").is_file():
        raise FileExistsError(output / "events.jsonl")
    write_once_jsonl(output / "events.jsonl", events)
    write_once_jsonl(output / "attempts.jsonl", attempts)
    write_once_jsonl(output / "replay_admission_attempts.jsonl", replay)
    counts = {}
    for split in ("calibration", "evaluation"):
        counts[split] = {
            task: {
                "admitted": sum(event["split"] == split and event["task"] == task for event in events),
                "by_seed": {str(seed): sum(event["split"] == split and event["task"] == task and int(event["actor_seed"]) == seed for event in events) for seed in ACTOR_SEEDS},
            }
            for task in ALL_CANDIDATE_TASKS
        }
    atomic_write_json(output / "summary.json", {"schema_version": 1, "expected_attempts": expected, "completed_attempts": len(attempts), "admitted_events": len(events), "unstable_event_exclusion_rate": 1.0 - len(events) / len(attempts) if attempts else 1.0, "counts": counts, "outcome_blind_admission": True})
    print(json.dumps({"expected": expected, "completed": len(attempts), "admitted": len(events)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=ACTOR_SEEDS)
    parser.add_argument("--task", choices=ALL_CANDIDATE_TASKS)
    parser.add_argument("--split", choices=("calibration", "evaluation"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--consolidate", action="store_true")
    args = parser.parse_args()
    if args.consolidate:
        consolidate()
        return
    if args.seed is None or args.task is None or args.split is None:
        parser.error("--seed, --task and --split are required unless --consolidate is used")
    run_shard(args.seed, args.task, args.split, args.device)


if __name__ == "__main__":
    main()
