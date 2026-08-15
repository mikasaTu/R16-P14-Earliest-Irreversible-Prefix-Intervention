from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from r16_p14_stage2a.envs import contact_pairs, current_observation, joint_qpos, restore_state, state_sha256
from r16_p14_stage2a.settings import TASK_SPECS
from r16_p14_stage2b.io_utils import atomic_write_json, load_jsonl, sha256_array, write_once_jsonl
from r16_p14_stage2b.runtime import ActorBundle, ActorHistory, normalized_disagreement

from .goal_geometry import target_distance
from .runtime import CauseTracker, apply_perturbation, hold_action, reconstruct_anchor, rollback_action
from .settings import (
    ACTOR_SEEDS,
    ARTIFACT_ROOT,
    DETECTION_PREFIX,
    EVALUATION_IDS,
    H_VALID,
    PREFIX_INDICES,
    RECOVERY_OPERATORS,
    TARGET_SHIFT_TASK,
)


def _balanced_take(events: list[dict[str, Any]], count: int = 24) -> list[dict[str, Any]]:
    by_seed = {
        seed: sorted([event for event in events if int(event["actor_seed"]) == seed], key=lambda item: int(item["init_state_id"]))
        for seed in ACTOR_SEEDS
    }
    selected = []
    while len(selected) < min(count, len(events)):
        changed = False
        for seed in ACTOR_SEEDS:
            if by_seed[seed] and len(selected) < count:
                selected.append(by_seed[seed].pop(0))
                changed = True
        if not changed:
            break
    return selected


def prepare_pool() -> None:
    qualification = json.loads((ARTIFACT_ROOT / "task_qualification/frozen_parameters.json").read_text())
    tasks = list(qualification["selected_tasks"])
    events = load_jsonl(ARTIFACT_ROOT / "actor_events/events.jsonl")
    pool = []
    summary = {"schema_version": 1, "tasks": tasks, "splits": {}}
    for split in ("calibration", "evaluation"):
        summary["splits"][split] = {}
        for task in tasks:
            candidates = [event for event in events if event["split"] == split and event["task"] == task]
            selected = _balanced_take(candidates, 24)
            parameters = [item["parameter"] for item in qualification["tasks"][task]["parameters"]]
            for index, event in enumerate(selected):
                parameter = parameters[index % len(parameters)]
                pool.append({
                    "event_instance_id": f"{event['event_id']}__{parameter['parameter_id']}",
                    "event_id": event["event_id"], "task": task, "split": split,
                    "actor_seed": event["actor_seed"], "init_state_id": event["init_state_id"],
                    "parameter": parameter, "parameter_id": parameter["parameter_id"],
                })
            seed_counts = Counter(int(item["actor_seed"]) for item in selected)
            severity_counts = Counter(item["parameter_id"] for item in pool if item["split"] == split and item["task"] == task)
            checks = {
                "events_ge_24": len(selected) >= 24,
                "each_generator_seed_ge_6": all(seed_counts[seed] >= 6 for seed in ACTOR_SEEDS),
                "each_severity_ge_8": len(severity_counts) >= 2 and all(value >= 8 for value in severity_counts.values()),
            }
            summary["splits"][split][task] = {
                "available_admitted_events": len(candidates), "selected_events": len(selected),
                "by_generator_seed": {str(seed): seed_counts[seed] for seed in ACTOR_SEEDS},
                "by_severity": dict(sorted(severity_counts.items())), "checks": checks,
                "passes": all(checks.values()),
            }
    summary["minimum_data_pass"] = all(item["passes"] for split in summary["splits"].values() for item in split.values())
    summary["continue_after_minimum_data_failure"] = True
    output = ARTIFACT_ROOT / "actor_events"
    write_once_jsonl(output / "formal_event_pool.jsonl", pool)
    atomic_write_json(output / "formal_event_pool_summary.json", summary)
    print(json.dumps({"events": len(pool), "minimum_data_pass": summary["minimum_data_pass"]}, sort_keys=True))


def _controller_state(env) -> list[tuple[Any, dict[str, Any]]]:
    result = []
    for robot in env.robots:
        controllers = robot.controller.values() if isinstance(robot.controller, dict) else [robot.controller]
        for controller in controllers:
            state = {}
            for key, value in vars(controller).items():
                if isinstance(value, np.ndarray):
                    state[key] = value.copy()
                elif isinstance(value, (np.generic, int, float, bool, str, type(None))):
                    state[key] = copy.deepcopy(value)
            result.append((controller, state))
    return result


def _restore_controller(snapshot: list[tuple[Any, dict[str, Any]]]) -> None:
    for controller, state in snapshot:
        for key, value in state.items():
            setattr(controller, key, value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value))


def _rng_state(env) -> list[tuple[Any, Any]]:
    result = []
    for owner in (env, env.env):
        for attr in ("np_random", "_np_random"):
            rng = getattr(owner, attr, None)
            if rng is not None and hasattr(rng, "get_state"):
                result.append((rng, copy.deepcopy(rng.get_state())))
    return result


def _restore_rng(snapshot: list[tuple[Any, Any]]) -> None:
    for rng, state in snapshot:
        rng.set_state(copy.deepcopy(state))


class EventRuntime:
    def __init__(self, event: dict[str, Any], generator: ActorBundle):
        env, history, audit = reconstruct_anchor(event, generator, capture_trace=False)
        if not audit["passed"]:
            env.close()
            raise ValueError(f"event replay rejected before matrix: {audit['checks']}")
        self.env = env
        self.event = event
        self.controller = _controller_state(env)
        self.rng = _rng_state(env)
        self.anchor_state = np.asarray(event["anchor_state"], dtype=np.float64)
        self.states = np.asarray(event["state_history"], dtype=np.float32)
        self.actions = np.asarray(event["action_history"], dtype=np.float32)
        self.anchor_timestep = int(event["anchor_global_step"])

    def reset(self) -> ActorHistory:
        restore_state(self.env, self.anchor_state)
        _restore_controller(self.controller)
        _restore_rng(self.rng)
        self.env.env.timestep = self.anchor_timestep
        if state_sha256(np.asarray(self.env.get_sim_state(), dtype=np.float64)) != self.event["anchor_state_hash"]:
            raise ValueError("anchor state restore is not byte exact")
        return ActorHistory(
            states=[row.copy() for row in self.states],
            actions=[row.copy() for row in self.actions],
        )

    def close(self) -> None:
        self.env.close()


def _step(env, history: ActorHistory, tracker: CauseTracker, action: np.ndarray, metrics: dict[str, Any]) -> None:
    observation, _, _, _ = env.step(np.asarray(action, dtype=np.float32))
    history.update(observation, action)
    tracker.observe_action(env, action)
    eef = np.asarray(observation.get("robot0_eef_pos", metrics["last_eef"]), dtype=np.float64)
    manipulated = joint_qpos(env, TASK_SPECS[tracker.task].manipulated_joint)
    metrics["eef_path_length"] += float(np.linalg.norm(eef - metrics["last_eef"]))
    metrics["object_path_length"] += float(np.linalg.norm(manipulated[:3] - metrics["last_object"][:3]))
    metrics["last_eef"] = eef
    metrics["last_object"] = manipulated
    metrics["contact_count"] += len(contact_pairs(env))
    metrics["actions_executed"] += 1


def _metrics(env, task: str) -> dict[str, Any]:
    observation = current_observation(env)
    spec = TASK_SPECS[task]
    assert spec.manipulated_joint
    return {
        "last_eef": np.asarray(observation.get("robot0_eef_pos", np.zeros(3)), dtype=np.float64),
        "last_object": joint_qpos(env, spec.manipulated_joint),
        "eef_path_length": 0.0, "object_path_length": 0.0,
        "contact_count": 0, "actions_executed": 0,
    }


def _tracker(event: dict[str, Any], history: ActorHistory) -> CauseTracker:
    initial = np.asarray(event["initial_manipulated_qpos"], dtype=np.float64)
    return CauseTracker(
        task=event["task"], initial_object_z=float(initial[2]),
        previous_gripper=float(history.actions[-1][-1]),
        ever_lifted=bool(event["phase"]["ever_lifted"]),
    )


def _detection(runtime: EventRuntime, history: ActorHistory, tracker: CauseTracker, parameter: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    for action in np.asarray(runtime.event["original_chunk"], dtype=np.float32)[:DETECTION_PREFIX]:
        _step(runtime.env, history, tracker, action, metrics)
    perturbation = apply_perturbation(runtime.env, runtime.event, parameter)
    tracker.observe_injection(runtime.env, tuple(tuple(item) for item in perturbation["before_contacts"]))
    history.refresh_latest_observation(current_observation(runtime.env))
    return {
        "perturbation": perturbation,
        "detection_goal_distance": target_distance(runtime.env, runtime.event["task"]),
    }


def _predict(bundle: ActorBundle, history: ActorHistory, task: str) -> np.ndarray:
    return np.asarray(bundle.predict(history.state_array(), history.action_array(), task), dtype=np.float32)


def _complete_tail(
    runtime: EventRuntime,
    history: ActorHistory,
    tracker: CauseTracker,
    bundle: ActorBundle,
    first_chunk: np.ndarray,
    *,
    execution_horizon: int,
    action_budget: int,
    policy_call_budget: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    task = runtime.event["task"]
    chunk = first_chunk
    calls = 1
    actions = 0
    success = bool(runtime.env.check_success())
    while actions < action_budget and not success:
        take = min(execution_horizon, action_budget - actions, len(chunk))
        for action in chunk[:take]:
            _step(runtime.env, history, tracker, action, metrics)
            actions += 1
            success = bool(runtime.env.check_success())
            if success or actions >= action_budget:
                break
        if actions < action_budget and not success:
            chunk = _predict(bundle, history, task)
            calls += 1
    if calls > policy_call_budget:
        raise RuntimeError(f"operator exceeded frozen call budget: {calls}>{policy_call_budget}")
    while calls < policy_call_budget:
        _predict(bundle, history, task)
        calls += 1
    return {"tail_actions": actions, "tail_actor_calls": calls, "task_success": bool(runtime.env.check_success())}


def _finish_row(
    runtime: EventRuntime,
    event_instance: dict[str, Any], tracker: CauseTracker,
    metrics: dict[str, Any], *, prefix_k: int, branch: str,
    actor_seed: int, prefix_actions: int, retained_cached: int,
    new_non_nominal: int, actor_calls: int, progress_at_k: float,
    disagreement: float | None, call_budget: int, action_budget: int,
    operator: str | None = None, prefix_cause_violation: bool = False,
) -> dict[str, Any]:
    success = bool(runtime.env.check_success())
    return {
        "event_instance_id": event_instance["event_instance_id"],
        "event_id": event_instance["event_id"], "task": event_instance["task"],
        "split": event_instance["split"], "init_state_id": event_instance["init_state_id"],
        "generator_actor_seed": event_instance["actor_seed"], "recovery_actor_seed": actor_seed,
        "parameter_id": event_instance["parameter_id"], "prefix_k": prefix_k,
        "branch": branch, "operator": operator,
        "task_success": success, "cause_violation": tracker.violation,
        "cause_violation_type": tracker.violation_type,
        "safe_success": bool(success and not tracker.violation),
        "S_obs_at_k": not prefix_cause_violation,
        "prefix_cause_violation": prefix_cause_violation,
        "retained_cached_actions": retained_cached,
        "new_non_nominal_actions": new_non_nominal,
        "prefix_actions": prefix_actions,
        "actor_calls": actor_calls, "allocated_actor_call_budget": call_budget,
        "total_post_detection_action_budget": action_budget,
        "total_actions_executed": metrics["actions_executed"],
        "completion_steps": metrics["actions_executed"],
        "eef_path_length": metrics["eef_path_length"],
        "object_path_length": metrics["object_path_length"],
        "contact_count": metrics["contact_count"],
        "task_progress_retained_at_k": progress_at_k,
        "cached_fresh_action_displacement": disagreement,
        "tail_execution_horizon": 4 if operator == "fresh_h4" else 16,
        "error": None,
    }


def matched_branch(runtime: EventRuntime, event_instance: dict[str, Any], bundle: ActorBundle, prefix_k: int, mode: str) -> dict[str, Any]:
    event = runtime.event
    history = runtime.reset()
    tracker = _tracker(event, history)
    metrics = _metrics(runtime.env, event["task"])
    detection = _detection(runtime, history, tracker, event_instance["parameter"], metrics)
    old = np.asarray(event["original_chunk"], dtype=np.float32)
    n = prefix_k - DETECTION_PREFIX
    fresh_d = None
    calls_before_k = 0
    if mode != "CACHED_NOQUERY":
        fresh_d = _predict(bundle, history, event["task"])
        calls_before_k = 1
    if mode in {"CACHED_MATCHED", "CACHED_NOQUERY"}:
        prefix = old[DETECTION_PREFIX:prefix_k]
        retained = n
        prefix_new = 0
    elif mode == "FRESH_MATCHED":
        assert fresh_d is not None
        prefix = fresh_d[:n]
        retained = 0
        prefix_new = n
    elif mode == "HOLD_PREFIX_MATCHED":
        prefix = np.asarray([hold_action(old[DETECTION_PREFIX - 1]) for _ in range(n)], dtype=np.float32).reshape(-1, 7)
        retained = 0
        prefix_new = n
    else:
        raise ValueError(mode)
    for action in prefix:
        _step(runtime.env, history, tracker, action, metrics)
    progress_at_k = detection["detection_goal_distance"] - target_distance(runtime.env, event["task"])
    prefix_cause_violation = bool(tracker.violation)
    disagreement = None
    if fresh_d is not None:
        disagreement = normalized_disagreement(old[min(prefix_k, H_VALID - 1)], fresh_d[min(n, H_VALID - 1)], bundle)
    first = _predict(bundle, history, event["task"])
    post_budget = max(0, TASK_SPECS[event["task"]].horizon - int(event["anchor_global_step"]) - DETECTION_PREFIX)
    tail_budget = max(0, post_budget - n)
    tail_call_budget = max(1, math.ceil(tail_budget / 16))
    tail = _complete_tail(runtime, history, tracker, bundle, first, execution_horizon=16, action_budget=tail_budget, policy_call_budget=tail_call_budget, metrics=metrics)
    total_calls = calls_before_k + tail["tail_actor_calls"]
    return _finish_row(
        runtime, event_instance, tracker, metrics, prefix_k=prefix_k, branch=mode,
        actor_seed=bundle.seed, prefix_actions=n, retained_cached=retained,
        new_non_nominal=prefix_new + tail["tail_actions"], actor_calls=total_calls,
        progress_at_k=progress_at_k, disagreement=disagreement,
        call_budget=calls_before_k + tail_call_budget, action_budget=post_budget,
        prefix_cause_violation=prefix_cause_violation,
    )


def recovery_branch(runtime: EventRuntime, event_instance: dict[str, Any], bundle: ActorBundle, prefix_k: int, operator: str) -> dict[str, Any]:
    event = runtime.event
    history = runtime.reset()
    tracker = _tracker(event, history)
    metrics = _metrics(runtime.env, event["task"])
    detection = _detection(runtime, history, tracker, event_instance["parameter"], metrics)
    old = np.asarray(event["original_chunk"], dtype=np.float32)
    n = prefix_k - DETECTION_PREFIX
    for action in old[DETECTION_PREFIX:prefix_k]:
        _step(runtime.env, history, tracker, action, metrics)
    progress_at_k = detection["detection_goal_distance"] - target_distance(runtime.env, event["task"])
    prefix_cause_violation = bool(tracker.violation)
    post_budget = max(0, TASK_SPECS[event["task"]].horizon - int(event["anchor_global_step"]) - DETECTION_PREFIX)
    tail_budget = max(0, post_budget - n)
    prelude = 0
    if operator == "hold_one_step_then_fresh_h16" and tail_budget > 0:
        _step(runtime.env, history, tracker, hold_action(old[max(0, prefix_k - 1)]), metrics)
        prelude = 1
    elif operator == "rollback_one_step_then_fresh_h16" and tail_budget > 0:
        _step(runtime.env, history, tracker, rollback_action(old[max(0, prefix_k - 1)]), metrics)
        prelude = 1
    horizon = 4 if operator == "fresh_h4" else 16
    first = _predict(bundle, history, event["task"])
    common_call_budget = max(1, math.ceil(tail_budget / 4))
    tail = _complete_tail(
        runtime, history, tracker, bundle, first,
        execution_horizon=horizon, action_budget=max(0, tail_budget - prelude),
        policy_call_budget=common_call_budget, metrics=metrics,
    )
    return _finish_row(
        runtime, event_instance, tracker, metrics, prefix_k=prefix_k,
        branch="RECOVERY_OPERATOR", operator=operator, actor_seed=bundle.seed,
        prefix_actions=n, retained_cached=n, new_non_nominal=prelude + tail["tail_actions"],
        actor_calls=tail["tail_actor_calls"], progress_at_k=progress_at_k,
        disagreement=None, call_budget=common_call_budget, action_budget=post_budget,
        prefix_cause_violation=prefix_cause_violation,
    )


def event_output_path(event_instance: dict[str, Any], smoke: bool) -> Path:
    scope = "integration_smoke" if smoke else "formal"
    return ARTIFACT_ROOT / "evaluation" / scope / event_instance["split"] / f"{event_instance['event_instance_id']}.json"


def run_event(event_instance: dict[str, Any], events: dict[str, dict[str, Any]], bundles: dict[int, ActorBundle], *, smoke: bool) -> None:
    output = event_output_path(event_instance, smoke)
    if output.is_file():
        print(f"EVAL_RESUME {event_instance['event_instance_id']}", flush=True)
        return
    event = events[event_instance["event_id"]]
    runtime = EventRuntime(event, bundles[int(event["actor_seed"])])
    matched, recovery, errors = [], [], []
    prefixes = (DETECTION_PREFIX, DETECTION_PREFIX + 2) if smoke else PREFIX_INDICES
    try:
        for prefix_k in prefixes:
            for mode in ("CACHED_MATCHED", "FRESH_MATCHED", "CACHED_NOQUERY", "HOLD_PREFIX_MATCHED"):
                try:
                    matched.append(matched_branch(runtime, event_instance, bundles[int(event["actor_seed"])], prefix_k, mode))
                except Exception as exc:
                    errors.append({"prefix_k": prefix_k, "branch": mode, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            for actor_seed in ACTOR_SEEDS:
                for operator in RECOVERY_OPERATORS:
                    try:
                        recovery.append(recovery_branch(runtime, event_instance, bundles[actor_seed], prefix_k, operator))
                    except Exception as exc:
                        errors.append({"prefix_k": prefix_k, "branch": "RECOVERY_OPERATOR", "operator": operator, "recovery_actor_seed": actor_seed, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    finally:
        runtime.close()
    expected_matched = len(prefixes) * 4
    expected_recovery = len(prefixes) * len(ACTOR_SEEDS) * len(RECOVERY_OPERATORS)
    payload = {
        "schema_version": 1, "complete": len(matched) == expected_matched and len(recovery) == expected_recovery and not errors,
        "event_instance": event_instance, "matched_rows": matched, "recovery_rows": recovery,
        "errors": errors, "expected_matched_rows": expected_matched, "expected_recovery_rows": expected_recovery,
    }
    atomic_write_json(output, payload)
    print(f"EVAL_DONE event={event_instance['event_instance_id']} matched={len(matched)}/{expected_matched} recovery={len(recovery)}/{expected_recovery} errors={len(errors)}", flush=True)


def run_worker(device: str, worker_index: int, worker_count: int, split: str | None, smoke: bool) -> None:
    pool = load_jsonl(ARTIFACT_ROOT / "actor_events/formal_event_pool.jsonl")
    if split is not None:
        pool = [item for item in pool if item["split"] == split]
    if smoke:
        picked = []
        for task in sorted({item["task"] for item in pool}):
            picked.append(next(item for item in pool if item["task"] == task and item["split"] == "evaluation"))
        pool = picked
    assigned = [item for item in pool if int(hashlib.sha256(item["event_instance_id"].encode()).hexdigest(), 16) % worker_count == worker_index]
    events = {event["event_id"]: event for event in load_jsonl(ARTIFACT_ROOT / "actor_events/events.jsonl")}
    bundles = {seed: ActorBundle.load(seed, device) for seed in ACTOR_SEEDS}
    for item in assigned:
        run_event(item, events, bundles, smoke=smoke)


def consolidate(smoke: bool = False) -> None:
    scope = "integration_smoke" if smoke else "formal"
    root = ARTIFACT_ROOT / "evaluation" / scope
    payloads = [json.loads(path.read_text()) for path in sorted(root.rglob("*.json"))]
    matched = [row for payload in payloads for row in payload.get("matched_rows", [])]
    recovery = [row for payload in payloads for row in payload.get("recovery_rows", [])]
    errors = [row for payload in payloads for row in payload.get("errors", [])]
    output = ARTIFACT_ROOT / ("integration_smoke" if smoke else "formal_matrix")
    write_once_jsonl(output / "matched_prefix_rows.jsonl", matched)
    write_once_jsonl(output / "recovery_operator_rows.jsonl", recovery)
    write_once_jsonl(output / "errors.jsonl", errors)
    summary = {
        "schema_version": 1, "scope": scope, "event_files": len(payloads),
        "complete_event_files": sum(bool(payload.get("complete")) for payload in payloads),
        "matched_rows": len(matched), "recovery_rows": len(recovery), "error_count": len(errors),
        "all_complete": bool(payloads) and all(payload.get("complete") for payload in payloads),
    }
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-pool", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--split", choices=("calibration", "evaluation"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--consolidate", action="store_true")
    args = parser.parse_args()
    if args.prepare_pool:
        prepare_pool()
    elif args.consolidate:
        consolidate(smoke=args.smoke)
    else:
        run_worker(args.device, args.worker_index, args.worker_count, args.split, args.smoke)


if __name__ == "__main__":
    main()
