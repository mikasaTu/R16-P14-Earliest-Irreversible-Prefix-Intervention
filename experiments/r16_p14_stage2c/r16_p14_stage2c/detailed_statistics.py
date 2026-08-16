from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from r16_p14_stage2b.io_utils import atomic_write_json, load_jsonl

from .contracts import hierarchical_cluster_bootstrap
from .settings import ARTIFACT_ROOT, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED


def macro_task_bootstrap(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    task_clusters: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        task_clusters[str(row["task"])][int(row["init_state_id"])].append(row)
    if not task_clusters:
        raise ValueError("no task clusters")
    task_values = {
        task: np.asarray(
            [np.mean([value(row) for row in clusters[init_id]]) for init_id in sorted(clusters)],
            dtype=np.float64,
        )
        for task, clusters in task_clusters.items()
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        per_task = []
        for task in sorted(task_values):
            values = task_values[task]
            sampled = rng.integers(0, len(values), size=len(values))
            per_task.append(float(np.mean(values[sampled])))
        draws[index] = float(np.mean(per_task))
    point_by_task = {task: float(np.mean(values)) for task, values in sorted(task_values.items())}
    return {
        "unit": "macro_average_of_task_specific_init_state_cluster_means",
        "tasks": sorted(task_values),
        "cluster_count_by_task": {task: len(values) for task, values in sorted(task_values.items())},
        "row_count": len(rows),
        "estimate": float(np.mean(list(point_by_task.values()))),
        "estimate_by_task": point_by_task,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "replicates": replicates,
        "seed": seed,
    }


def _bootstrap(
    rows: list[dict[str, Any]],
    key: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any] | None:
    if not rows:
        return None
    return hierarchical_cluster_bootstrap(
        rows,
        lambda row: float(row[key]),
        replicates=replicates,
        seed=seed,
    )


def stratified_bootstraps(
    rows: list[dict[str, Any]],
    key: str,
    stratum: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if stratum == "task_parameter":
            label = f"{row['task']}/{row['parameter_id']}"
        else:
            label = str(row[stratum])
        grouped[label].append(row)
    return {
        label: _bootstrap(group, key, replicates=replicates, seed=seed + index)
        for index, (label, group) in enumerate(sorted(grouped.items()))
    }


def track_statistics(
    rows: list[dict[str, Any]],
    *,
    safe_key: str,
    action_key: str,
    actor_key: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    return {
        "row_count": len(evaluation),
        "safe_success": {
            "all_clusters": _bootstrap(evaluation, safe_key, replicates=replicates, seed=seed),
            "macro_task": macro_task_bootstrap(evaluation, lambda row: float(row[safe_key]), replicates=replicates, seed=seed + 1) if evaluation else None,
            "by_task": stratified_bootstraps(evaluation, safe_key, "task", replicates=replicates, seed=seed + 10),
            "by_task_severity": stratified_bootstraps(evaluation, safe_key, "task_parameter", replicates=replicates, seed=seed + 100),
            "by_actor_seed": stratified_bootstraps(evaluation, safe_key, actor_key, replicates=replicates, seed=seed + 1000),
        },
        "new_non_nominal_actions": {
            "all_clusters": _bootstrap(evaluation, action_key, replicates=replicates, seed=seed + 2),
            "macro_task": macro_task_bootstrap(evaluation, lambda row: float(row[action_key]), replicates=replicates, seed=seed + 3) if evaluation else None,
            "by_task": stratified_bootstraps(evaluation, action_key, "task", replicates=replicates, seed=seed + 20),
            "by_task_severity": stratified_bootstraps(evaluation, action_key, "task_parameter", replicates=replicates, seed=seed + 200),
            "by_actor_seed": stratified_bootstraps(evaluation, action_key, actor_key, replicates=replicates, seed=seed + 2000),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    track_a = load_jsonl(root / "matched_prefix/track_a_rows.jsonl")
    track_b = load_jsonl(root / "crossfit_replanability/crossfit_rows.jsonl")
    decision = json.loads((root / "decision.json").read_text())
    payload = {
        "schema_version": 1,
        "independent_cluster": ["task", "init_state_id"],
        "actor_seed_role": "repeated measurement within cluster",
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "validity": {
            "replay_contract": decision["replay_contract"],
            "second_failure_family": decision["second_failure_family"],
            "causal_or_deployable_inference_allowed": False,
            "interpretation": "All planned rows are reported, but these post-hoc stratified intervals are descriptive diagnostics because the replay and perturbation-family gates are blocked.",
        },
        "track_a": track_statistics(
            track_a,
            safe_key="safe_success_difference",
            action_key="new_action_difference",
            actor_key="generator_actor_seed",
            replicates=args.bootstrap_replicates,
            seed=BOOTSTRAP_SEED + 10_000,
        ),
        "track_b": track_statistics(
            track_b,
            safe_key="safe_success_difference",
            action_key="new_action_difference",
            actor_key="heldout_actor_seed",
            replicates=args.bootstrap_replicates,
            seed=BOOTSTRAP_SEED + 20_000,
        ),
        "evaluation_outcomes_used_for_selection": False,
        "note": "This file adds preregistered reporting strata only; it does not refit a baseline, task, severity, threshold, or decision.",
    }
    atomic_write_json(root / "statistics_detailed.json", payload)
    print(json.dumps({"status": "complete", "output": str(root / "statistics_detailed.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
