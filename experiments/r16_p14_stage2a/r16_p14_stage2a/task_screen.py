from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .envs import feature_from_obs, make_env
from .io_utils import atomic_write_json, atomic_write_jsonl, atomic_write_text
from .mechanics import build_candidate, reconstruct_prefix, replay_equal
from .settings import CALIBRATION_DEMOS, TASK_NAMES, TASK_SPECS, perturbation_grid


def screen_task(task_name: str, config_dir: Path) -> dict[str, Any]:
    spec = TASK_SPECS[task_name]
    dataset_exists = spec.hdf5_path.is_file()
    demo_count = 0
    if dataset_exists:
        with h5py.File(spec.hdf5_path, "r") as dataset:
            demo_count = len(dataset["data"])
    env_a, suite = make_env(task_name, config_dir=config_dir, seed=0)
    env_b, _ = make_env(task_name, config_dir=config_dir, seed=0)
    selected = None
    errors: list[str] = []
    replay = None
    phase_contract = None
    try:
        init_states = np.asarray(suite.get_task_init_states(spec.task_id))
        observation = env_a.regenerate_obs_from_state(init_states[0])
        feature_shape_valid = feature_from_obs(observation).shape == (95,)
        action_shape_valid = tuple(env_a.env.action_spec[0].shape) == (7,)
        for parameters in perturbation_grid(task_name):
            for demo_id in CALIBRATION_DEMOS:
                try:
                    candidate = build_candidate(env_a, task_name, demo_id, parameters)
                    _, left = reconstruct_prefix(env_a, candidate, prefix_k=2)
                    _, right = reconstruct_prefix(env_b, candidate, prefix_k=2)
                    if left["phase_contract_valid"]:
                        selected = candidate
                        phase_contract = left["phase_contract"]
                        replay = replay_equal(left, right)
                        break
                except Exception as error:  # retain every structural failure
                    errors.append(
                        f"demo={demo_id} {type(error).__name__}: {error}"
                    )
            if selected is not None:
                break
        criteria = {
            "dataset_exists": dataset_exists,
            "at_least_50_demonstrations": demo_count >= 50,
            "stable_phase_anchor_defined": bool(
                phase_contract and phase_contract["stable_phase_anchor"]
            ),
            "environment_only_perturbation_implementable": bool(
                phase_contract and phase_contract["environment_only"]
            ),
            "injection_instant_has_no_direct_violation": bool(
                phase_contract and phase_contract["no_injection_cause_violation"]
            ),
            "at_least_8_actions_remain_in_chunk": bool(
                phase_contract and phase_contract["at_least_8_chunk_actions_remaining"]
            ),
            "fresh_replanning_theoretically_available": bool(
                feature_shape_valid and action_shape_valid and not env_a.check_success()
            ),
            "fresh_env_prefix_reconstruction_available": bool(
                replay and replay["passed"]
            ),
        }
        status = "eligible" if all(criteria.values()) else "structurally_ineligible"
        return {
            "task": task_name,
            "task_id": spec.task_id,
            "family": spec.family,
            "status": status,
            "criteria": criteria,
            "structural_smoke_config_id": selected.config_id if selected else None,
            "structural_smoke_parameters": selected.parameters if selected else None,
            "structural_smoke_candidate_id": selected.candidate_id if selected else None,
            "structural_smoke_demo_id": selected.demo_id if selected else None,
            "fresh_replay": replay,
            "dataset_demo_count": demo_count,
            "errors_before_first_structurally_valid_config": errors,
            "intervention_outcome_inspected": False,
        }
    finally:
        env_a.close()
        env_b.close()


def render(records: list[dict[str, Any]]) -> str:
    lines = [
        "# Structural task eligibility",
        "",
        "Screening used only source geometry, demonstrations, injection-instant checks, and fresh-environment prefix reconstruction. No recovery or proposed-method outcome was inspected.",
        "",
        "| Task | Family | Status | Structural smoke configuration | Calibration demo |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for record in records:
        lines.append(
            f"| {record['task']} | {record['family']} | {record['status']} | "
            f"{record['structural_smoke_config_id'] or 'none'} | "
            f"{record['structural_smoke_demo_id'] if record['structural_smoke_demo_id'] is not None else 'none'} |"
        )
    lines.extend(
        [
            "",
            "The initial six-task roster is retained. No replacement was used, and no task was removed based on Stage-2A gain.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = [screen_task(task, args.config_dir) for task in TASK_NAMES]
    roster = {
        "schema_version": 1,
        "initial_roster": list(TASK_NAMES),
        "frozen_roster": list(TASK_NAMES),
        "replacement_used": False,
        "replacement_count": 0,
        "replacement_due_to_gain": False,
        "eligible_task_count": sum(record["status"] == "eligible" for record in records),
        "all_tasks_eligible": all(record["status"] == "eligible" for record in records),
        "structural_smoke_configs": {
            record["task"]: {
                "config_id": record["structural_smoke_config_id"],
                "parameters": record["structural_smoke_parameters"],
                "demo_id": record["structural_smoke_demo_id"],
            }
            for record in records
        },
    }
    atomic_write_jsonl(args.output_dir / "eligibility.jsonl", records)
    atomic_write_json(args.output_dir / "task_roster_frozen.json", roster)
    atomic_write_text(args.output_dir / "report.md", render(records))
    print(json.dumps(roster, indent=2, sort_keys=True))
    if not roster["all_tasks_eligible"]:
        raise SystemExit("STRUCTURAL_TASK_SCREEN_INCOMPLETE")


if __name__ == "__main__":
    main()
