from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .envs import make_env
from .io_utils import atomic_write_json, atomic_write_jsonl, atomic_write_text
from .mechanics import build_candidate, reconstruct_prefix
from .settings import TASK_SPECS


ANCHORS = {
    "positive": "put_the_cream_cheese_in_the_bowl",
    "negative": "open_the_middle_drawer_of_the_cabinet",
}
BRANCH_POINTS = (2, 8, 16)
REPEATS = 3


def load_candidate_spec(
    task: str,
    frozen: dict[str, Any],
    screen: dict[str, Any],
) -> tuple[dict[str, float | int], int]:
    selected = frozen["tasks"][task]
    if selected["selected_parameters"] is not None:
        # A formally selected perturbation passed the phase contract on every
        # calibration demonstration, so demo 0 is a valid deterministic anchor.
        return selected["selected_parameters"], 0
    smoke = screen["structural_smoke_configs"][task]
    return smoke["parameters"], int(smoke["demo_id"])


def compare_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    reference = records[0]
    state_errors = [
        float(np.max(np.abs(np.asarray(record["state"]) - np.asarray(reference["state"]))))
        for record in records[1:]
    ]
    state_hash_match = all(record["state_hash"] == reference["state_hash"] for record in records[1:])
    contact_match = all(record["contacts_hash"] == reference["contacts_hash"] for record in records[1:])
    outcome_match = all(record["outcome"] == reference["outcome"] for record in records[1:])
    chunk_match = all(
        record["action_chunk_hash"] == reference["action_chunk_hash"]
        for record in records[1:]
    )
    maximum_error = max(state_errors, default=0.0)
    return {
        "state_hash_match": state_hash_match,
        "contact_match": contact_match,
        "outcome_match": outcome_match,
        "action_chunk_hash_exact": chunk_match,
        "max_state_error": maximum_error,
        "max_state_error_le_1e_9": maximum_error <= 1e-9,
        "passed": bool(
            state_hash_match
            and contact_match
            and outcome_match
            and chunk_match
            and maximum_error <= 1e-9
        ),
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Fresh-environment replay gate",
        "",
        "Every branch-point reconstruction uses an independent environment, restores the demonstration anchor, replays the exact nominal prefix, injects the same environment-only perturbation, and then reaches the branch point. Snapshot restore is not the primary initializer.",
        "",
        "| Anchor | Task | Branch points | Reconstruction passes | Max state error |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for label, item in summary["anchors"].items():
        lines.append(
            f"| {label} | {item['task']} | {item['branch_point_count']} | "
            f"{item['pass_count']}/{item['branch_point_count']} | {item['max_state_error']} |"
        )
    lines.extend(["", f"Overall status: **{summary['status']}**.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--frozen-perturbations", type=Path, required=True)
    parser.add_argument("--task-screen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.frozen_perturbations.read_text())
    screen = json.loads(args.task_screen.read_text())
    raw_records: list[dict[str, Any]] = []
    anchor_summaries: dict[str, Any] = {}
    for label, task in ANCHORS.items():
        parameters, demo_id = load_candidate_spec(task, frozen, screen)
        base_env, _ = make_env(task, config_dir=args.config_dir, seed=0)
        try:
            candidate = build_candidate(base_env, task, demo_id, parameters)
        finally:
            base_env.close()
        groups = []
        for branch_k in BRANCH_POINTS:
            group_records = []
            for repeat in range(REPEATS):
                env, _ = make_env(task, config_dir=args.config_dir, seed=0)
                try:
                    _, info = reconstruct_prefix(env, candidate, prefix_k=branch_k)
                finally:
                    env.close()
                serializable = {
                    "anchor_label": label,
                    "task": task,
                    "candidate_id": candidate.candidate_id,
                    "config_id": candidate.config_id,
                    "branch_k": branch_k,
                    "repeat": repeat,
                    "initializer": "fresh_independent_env_prefix_reconstruction",
                    "state": info["state"].tolist(),
                    "state_hash": info["state_hash"],
                    "contacts_hash": info["contacts_hash"],
                    "outcome": info["outcome"],
                    "action_chunk_hash": info["action_chunk_hash"],
                    "shared_mutable_env": False,
                }
                group_records.append(serializable)
                raw_records.append(serializable)
            comparison = compare_group(group_records)
            groups.append({"branch_k": branch_k, "repeats": REPEATS, **comparison})
        anchor_summaries[label] = {
            "task": task,
            "demo_id": candidate.demo_id,
            "config_id": candidate.config_id,
            "branch_point_count": len(groups),
            "reconstructions_per_branch_point": REPEATS,
            "pass_count": sum(group["passed"] for group in groups),
            "max_state_error": max(group["max_state_error"] for group in groups),
            "contact_outcome_agreement": all(
                group["contact_match"] and group["outcome_match"] for group in groups
            ),
            "action_chunk_hash_exact": all(group["action_chunk_hash_exact"] for group in groups),
            "groups": groups,
        }
    total = sum(item["branch_point_count"] for item in anchor_summaries.values())
    passes = sum(item["pass_count"] for item in anchor_summaries.values())
    maximum_error = max(item["max_state_error"] for item in anchor_summaries.values())
    passed = bool(
        passes / total >= 0.99
        and all(item["contact_outcome_agreement"] for item in anchor_summaries.values())
        and all(item["action_chunk_hash_exact"] for item in anchor_summaries.values())
        and maximum_error <= 1e-9
    )
    summary = {
        "schema_version": 1,
        "status": "PASS" if passed else "BLOCKED_BY_REPLAY",
        "initializer": "fresh_independent_env_prefix_reconstruction",
        "snapshot_restore_primary": False,
        "anchors": anchor_summaries,
        "branch_point_count": total,
        "branch_point_pass_count": passes,
        "branch_point_state_match_rate": passes / total,
        "max_state_error": maximum_error,
        "contact_and_outcome_agreement": all(
            item["contact_outcome_agreement"] for item in anchor_summaries.values()
        ),
        "action_chunk_hash_exact": all(
            item["action_chunk_hash_exact"] for item in anchor_summaries.values()
        ),
    }
    atomic_write_jsonl(args.output_dir / "branch_reconstructions.jsonl", raw_records)
    atomic_write_json(args.output_dir / "summary.json", summary)
    atomic_write_text(args.output_dir / "report.md", render(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("BLOCKED_BY_REPLAY")


if __name__ == "__main__":
    main()
