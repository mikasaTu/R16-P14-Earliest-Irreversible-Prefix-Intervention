from __future__ import annotations

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from r16_p14_stage2a.envs import array_sha256, restore_state
from r16_p14_stage2a.settings import TASK_SPECS

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
    init_state_and_suite,
    normalized_disagreement,
    target_distance,
)
from .settings import (
    ACTOR_SEEDS,
    ARTIFACT_ROOT,
    CHUNK_LENGTH,
    EXECUTION_HORIZONS,
    EXPERIMENT_ROOT,
    PRIMARY_TASKS,
)


def rollout(
    env,
    *,
    task: str,
    init_id: int,
    init_state: np.ndarray,
    execution_horizon: int,
    bundle: ActorBundle,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = TASK_SPECS[task]
    env.reset()
    observation = restore_state(env, init_state)
    history = ActorHistory.initial(observation)
    monitor = TaskMonitor.create(env, task)
    start_distance = target_distance(env, task, monitor.goal)
    actions: list[np.ndarray] = []
    chunk_smoothness: list[float] = []
    chunk_hashes: list[str] = []
    prefix_records: list[dict[str, Any]] = []
    success = bool(env.check_success())
    execution_policy_calls = 0
    diagnostic_policy_calls = 0
    late_anchor_recorded = False
    while len(actions) < spec.horizon and not success:
        chunk = bundle.predict(history.state_array(), history.action_array(), task)
        execution_policy_calls += 1
        chunk_hashes.append(chunk_hash(chunk))
        chunk_smoothness.append(
            float(np.linalg.norm(np.diff(chunk[:, :6], axis=0), axis=1).mean())
        )
        preconditions = monitor.late_anchor_preconditions(env, chunk, CHUNK_LENGTH)
        is_first_late_anchor = not late_anchor_recorded and all(preconditions.values())
        if is_first_late_anchor:
            late_anchor_recorded = True
            anchor_distance = target_distance(env, task, monitor.goal)
            anchor_best_distance = monitor.best_distance
            anchor_chunk_hash = chunk_hashes[-1]
            anchor_step = len(actions)
            late_intermediate: list[dict[str, Any]] = []
        else:
            late_intermediate = []
        segment_limit = min(execution_horizon, spec.horizon - len(actions))
        for offset, action in enumerate(chunk[:segment_limit]):
            if success:
                break
            observation, _, _, _ = env.step(action)
            history.update(observation, action)
            actions.append(np.asarray(action, dtype=np.float32).copy())
            success = bool(env.check_success())
            monitor.observe(env, action, success)
            disagreement = None
            fresh_hash = None
            if offset + 1 < segment_limit and not success:
                fresh = bundle.predict(history.state_array(), history.action_array(), task)
                diagnostic_policy_calls += 1
                disagreement = normalized_disagreement(chunk[offset + 1], fresh[0], bundle)
                fresh_hash = chunk_hash(fresh)
                prefix_records.append(
                    {
                        "record_type": "intermediate_disagreement",
                        "task": task,
                        "actor_seed": bundle.seed,
                        "init_state_id": init_id,
                        "execution_horizon": execution_horizon,
                        "episode_step": len(actions),
                        "chunk_start_step": len(actions) - offset - 1,
                        "offset_after_chunk_start": offset + 1,
                        "old_chunk_hash": chunk_hashes[-1],
                        "old_chunk_next_action": np.asarray(chunk[offset + 1], dtype=np.float32).tolist(),
                        "fresh_actor_first_action": np.asarray(fresh[0], dtype=np.float32).tolist(),
                        "fresh_chunk_hash": fresh_hash,
                        "normalized_disagreement": disagreement,
                    }
                )
            if is_first_late_anchor:
                distance = target_distance(env, task, monitor.goal)
                progress_delta = float(anchor_distance - distance)
                late_intermediate.append(
                    {
                        "record_type": "late_anchor_prefix",
                        "task": task,
                        "actor_seed": bundle.seed,
                        "init_state_id": init_id,
                        "execution_horizon": execution_horizon,
                        "anchor_global_step": anchor_step,
                        "anchor_chunk_hash": anchor_chunk_hash,
                        "prefix_h": offset + 1,
                        "prefix_remains_phase_valid": bool(
                            not monitor.object_drop and not monitor.wrong_release and not monitor.phase_regression
                        ),
                        "grasp_remains_valid": bool(not monitor.object_drop and not monitor.wrong_release),
                        "no_task_specific_catastrophic_event": bool(
                            not monitor.object_drop and not monitor.wrong_release
                        ),
                        "task_progress_delta": progress_delta,
                        "normalized_disagreement_to_fresh": disagreement,
                        "fresh_chunk_hash": fresh_hash,
                        "task_success": success,
                    }
                )
        if is_first_late_anchor:
            prefix_records.extend(late_intermediate)
            executed = len(late_intermediate)
            distance = target_distance(env, task, monitor.goal)
            progress_delta = float(anchor_distance - distance)
            prefix_records.append(
                {
                    "record_type": "late_anchor_final",
                    "task": task,
                    "actor_seed": bundle.seed,
                    "init_state_id": init_id,
                    "execution_horizon": execution_horizon,
                    "requested_prefix_h": execution_horizon,
                    "executed_prefix_h": executed,
                    "anchor_global_step": anchor_step,
                    "anchor_chunk_hash": anchor_chunk_hash,
                    "prefix_remains_phase_valid": bool(
                        not monitor.object_drop and not monitor.wrong_release and not monitor.phase_regression
                    ),
                    "grasp_remains_valid": bool(not monitor.object_drop and not monitor.wrong_release),
                    "no_task_specific_catastrophic_event": bool(
                        not monitor.object_drop and not monitor.wrong_release
                    ),
                    "task_progress_delta": progress_delta,
                    "prefix_faithful": monitor.faithful(progress_delta, success),
                    "completed_task_inside_prefix": success,
                    "anchor_best_distance": float(anchor_best_distance),
                }
            )
    action_array = np.asarray(actions, dtype=np.float32)
    return {
        "record_type": "clean_chunk_execution_episode",
        "task": task,
        "actor_seed": bundle.seed,
        "checkpoint": str(bundle.checkpoint),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "init_state_id": init_id,
        "init_state_hash": sha256_array(init_state, np.float64),
        "execution_horizon": execution_horizon,
        "generated_chunk_length": CHUNK_LENGTH,
        "clean_success": bool(success),
        "safe_success": bool(success and not monitor.object_drop and not monitor.wrong_release),
        "grasp_lift_reached": bool(monitor.ever_stably_lifted),
        "late_phase_reached": bool(monitor.late_phase_reached),
        "object_drop": bool(monitor.object_drop),
        "wrong_release": bool(monitor.wrong_release),
        "phase_regression": bool(monitor.phase_regression),
        "policy_calls": execution_policy_calls,
        "diagnostic_policy_calls": diagnostic_policy_calls,
        "episode_length": len(actions),
        "action_space_path_length": (
            float(np.linalg.norm(action_array[:, :3], axis=1).sum()) if len(action_array) else 0.0
        ),
        "chunk_smoothness": float(np.mean(chunk_smoothness)) if chunk_smoothness else 0.0,
        "first_original_chunk_hash": chunk_hashes[0] if chunk_hashes else None,
        "last_original_chunk_hash": chunk_hashes[-1] if chunk_hashes else None,
        "chunk_trace_hash": sha256_array(np.asarray(chunk_hashes, dtype="S64")) if chunk_hashes else None,
        "late_anchor_found": late_anchor_recorded,
        "initial_target_distance": start_distance,
        "final_target_distance": target_distance(env, task, monitor.goal),
        "simulator_or_action_contract_error": False,
        "error": None,
    }, prefix_records


def episode_checkpoint_path(output: Path, seed: int, task: str, horizon: int, init_id: int) -> Path:
    return (
        output
        / "episode_checkpoints"
        / f"seed_{seed}"
        / f"{task}__h{horizon:02d}__init{init_id:02d}.json"
    )


def consolidate_seed_checkpoints(output: Path, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    episodes: list[dict[str, Any]] = []
    prefixes: list[dict[str, Any]] = []
    for task in PRIMARY_TASKS:
        for horizon in EXECUTION_HORIZONS:
            for init_id in range(20):
                checkpoint = episode_checkpoint_path(output, seed, task, horizon, init_id)
                if not checkpoint.is_file():
                    return None
                saved = json.loads(checkpoint.read_text())
                episodes.append(saved["episode"])
                prefixes.extend(saved["prefix_records"])
    return episodes, prefixes


def run_seed(seed: int, device: str, tasks: tuple[str, ...] = PRIMARY_TASKS) -> None:
    output = ARTIFACT_ROOT / "chunk_executability/shards"
    episode_path = output / f"seed_{seed}.episodes.jsonl"
    prefix_path = output / f"seed_{seed}.prefix_records.jsonl"
    if episode_path.is_file() and prefix_path.is_file():
        print(f"CHUNK_SEED_ALREADY_COMPLETE seed={seed}")
        return
    bundle = ActorBundle.load(seed, device)
    for task in tasks:
        env, _, init_states = init_state_and_suite(task)
        try:
            hashes = [sha256_array(init_states[index], np.float64) for index in range(20)]
            if len(set(hashes)) != 20:
                raise ValueError(f"duplicate init-state hashes for {task}")
            for horizon in EXECUTION_HORIZONS:
                for init_id in range(20):
                    checkpoint = episode_checkpoint_path(output, seed, task, horizon, init_id)
                    if checkpoint.is_file():
                        saved = json.loads(checkpoint.read_text())
                        print(
                            f"CHUNK_EPISODE_RESUME seed={seed} task={task} h={horizon} init={init_id}",
                            flush=True,
                        )
                        continue
                    try:
                        episode, records = rollout(
                            env,
                            task=task,
                            init_id=init_id,
                            init_state=init_states[init_id],
                            execution_horizon=horizon,
                            bundle=bundle,
                        )
                        write_once_json(
                            checkpoint,
                            {"episode": episode, "prefix_records": records, "complete": True},
                        )
                        print(
                            f"CHUNK_EPISODE seed={seed} task={task} h={horizon} init={init_id} "
                            f"success={int(episode['clean_success'])} late={int(episode['late_phase_reached'])} "
                            f"steps={episode['episode_length']}",
                            flush=True,
                        )
                    except Exception as exc:
                        episode = {
                                "record_type": "clean_chunk_execution_episode",
                                "task": task,
                                "actor_seed": seed,
                                "checkpoint": str(bundle.checkpoint),
                                "checkpoint_sha256": bundle.checkpoint_sha256,
                                "init_state_id": init_id,
                                "init_state_hash": hashes[init_id],
                                "execution_horizon": horizon,
                                "clean_success": False,
                                "late_phase_reached": False,
                                "simulator_or_action_contract_error": True,
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(),
                            }
                        write_once_json(
                            checkpoint,
                            {"episode": episode, "prefix_records": [], "complete": True},
                        )
        finally:
            env.close()
    consolidated = consolidate_seed_checkpoints(output, seed)
    if consolidated is None:
        print(f"CHUNK_SEED_PARTIAL seed={seed} tasks={','.join(tasks)}", flush=True)
        return
    episodes, prefixes = consolidated
    write_once_jsonl(episode_path, episodes)
    write_once_jsonl(prefix_path, prefixes)
    atomic_write_json(
        output / f"seed_{seed}.complete.json",
        {"seed": seed, "episode_count": len(episodes), "prefix_record_count": len(prefixes)},
    )


def mean(records: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(record.get(key, False)) for record in records])) if records else 0.0


def aggregate() -> dict[str, Any]:
    shard = ARTIFACT_ROOT / "chunk_executability/shards"
    episodes: list[dict[str, Any]] = []
    prefixes: list[dict[str, Any]] = []
    for seed in ACTOR_SEEDS:
        episodes.extend(load_jsonl(shard / f"seed_{seed}.episodes.jsonl"))
        prefixes.extend(load_jsonl(shard / f"seed_{seed}.prefix_records.jsonl"))
    expected = len(PRIMARY_TASKS) * len(ACTOR_SEEDS) * len(EXECUTION_HORIZONS) * 20
    if len(episodes) != expected:
        raise ValueError(f"expected {expected} episodes, found {len(episodes)}")
    finals = [record for record in prefixes if record["record_type"] == "late_anchor_final"]
    by_horizon: dict[str, Any] = {}
    baseline = [record for record in episodes if record["execution_horizon"] == 1]
    base_success = mean(baseline, "clean_success")
    base_late = mean(baseline, "late_phase_reached")
    passing: list[int] = []
    for horizon in EXECUTION_HORIZONS:
        subset = [record for record in episodes if record["execution_horizon"] == horizon]
        final_subset = [record for record in finals if record["execution_horizon"] == horizon]
        seed_drops = {}
        passing_seeds = 0
        for seed in ACTOR_SEEDS:
            seed_base = [record for record in baseline if record["actor_seed"] == seed]
            seed_h = [record for record in subset if record["actor_seed"] == seed]
            drop = mean(seed_base, "clean_success") - mean(seed_h, "clean_success")
            seed_drops[str(seed)] = drop
            passing_seeds += int(drop <= 0.20)
        success = mean(subset, "clean_success")
        late = mean(subset, "late_phase_reached")
        faithful = mean(final_subset, "prefix_faithful")
        errors = sum(bool(record.get("simulator_or_action_contract_error")) for record in subset)
        checks = {
            "success_within_h1_minus_0_15": success >= base_success - 0.15,
            "late_phase_within_h1_minus_0_15": late >= base_late - 0.15,
            "prefix_faithful_rate_ge_0_80": faithful >= 0.80,
            "at_least_two_seed_drops_le_0_20": passing_seeds >= 2,
            "no_simulator_or_action_contract_error": errors == 0,
        }
        if all(checks.values()):
            passing.append(horizon)
        task_metrics = {}
        for task in PRIMARY_TASKS:
            task_rows = [record for record in subset if record["task"] == task]
            task_finals = [record for record in final_subset if record["task"] == task]
            task_metrics[task] = {
                "episodes": len(task_rows),
                "clean_success_rate": mean(task_rows, "clean_success"),
                "late_phase_reach_rate": mean(task_rows, "late_phase_reached"),
                "prefix_faithful_rate": mean(task_finals, "prefix_faithful"),
                "late_anchor_count": len(task_finals),
                "object_drop_rate": mean(task_rows, "object_drop"),
                "wrong_release_rate": mean(task_rows, "wrong_release"),
                "phase_regression_rate": mean(task_rows, "phase_regression"),
                "mean_policy_calls": float(np.mean([row.get("policy_calls", 0) for row in task_rows])),
            }
        by_horizon[str(horizon)] = {
            "episodes": len(subset),
            "clean_success_rate": success,
            "late_phase_reach_rate": late,
            "prefix_faithful_rate": faithful,
            "late_anchor_count": len(final_subset),
            "seed_success_drops_from_h1": seed_drops,
            "passing_seed_count": passing_seeds,
            "simulator_or_action_contract_error_count": errors,
            "checks": checks,
            "passes_H_valid_contract": all(checks.values()),
            "tasks": task_metrics,
        }
    h_valid = max(passing) if passing else None
    status = "PASS" if h_valid is not None and h_valid >= 8 else "BLOCKED"
    summary = {
        "schema_version": 1,
        "status": status,
        "failure_label": None if status == "PASS" else "BLOCKED_BY_CHUNK_EXECUTABILITY",
        "H_valid": h_valid,
        "required_H_valid": 8,
        "episode_count": len(episodes),
        "prefix_record_count": len(prefixes),
        "h1_aggregate_success": base_success,
        "h1_aggregate_late_phase_reach": base_late,
        "horizons": by_horizon,
        "early_stop_applied": False,
        "downstream_continuation_required_by_user": True,
    }
    output = ARTIFACT_ROOT / "chunk_executability"
    atomic_write_json(output / "summary.json", summary)
    write_once_jsonl(output / "raw_episodes.jsonl", episodes)
    write_once_jsonl(output / "prefix_records.jsonl", prefixes)
    lines = [
        "# Phase A — action-chunk executability",
        "",
        f"Status: **{status}**; `H_valid={h_valid}` (required >=8).",
        "",
        "| horizon | success | late reach | prefix faithful | errors | passes |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for horizon in EXECUTION_HORIZONS:
        row = by_horizon[str(horizon)]
        lines.append(
            f"| {horizon} | {row['clean_success_rate']:.3f} | {row['late_phase_reach_rate']:.3f} | "
            f"{row['prefix_faithful_rate']:.3f} | {row['simulator_or_action_contract_error_count']} | "
            f"{'yes' if row['passes_H_valid_contract'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "A blocked result identifies an unsuitable frozen chunk substrate; it is not interpreted as an R16-P14 mechanism failure.",
            "The explicit user override requires later phases to run regardless of this gate.",
            "",
        ]
    )
    report = "\n".join(lines)
    atomic_write_text(output / "report.md", report)
    atomic_write_json(EXPERIMENT_ROOT / "chunk_executability/summary.json", summary)
    atomic_write_text(EXPERIMENT_ROOT / "chunk_executability/report.md", report)
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
