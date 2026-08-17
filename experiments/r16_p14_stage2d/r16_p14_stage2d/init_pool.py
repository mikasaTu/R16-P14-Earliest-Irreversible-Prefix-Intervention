from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from r16_p14_stage2a.envs import (
    contact_pairs,
    current_observation,
    feature_from_obs,
    joint_qpos,
    make_env,
    state_sha256,
)
from r16_p14_stage2a.settings import TASK_SPECS
from r16_p14_stage2a.mechanics import cause_contact

from .io_utils import atomic_write_json, sha256_array, sha256_file
from .settings import (
    ALL_INIT_IDS,
    ARTIFACT_ROOT,
    CALIBRATION_IDS,
    EVALUATION_IDS,
    EXPERIMENT_ROOT,
    INFRASTRUCTURE_IDS,
    MIRROR_EXPERIMENT_OUTPUTS,
    RESET_SEED_BASE,
    RESERVE_IDS,
    TASKS,
)


def reset_seed(task: str, init_id: int) -> int:
    return RESET_SEED_BASE + TASKS.index(task) * 10_000 + int(init_id)


def initial_collision_valid(env, task: str) -> tuple[bool, list[tuple[str, str]]]:
    pairs = list(contact_pairs(env))
    spec = TASK_SPECS[task]
    if task == "put_the_bowl_on_the_stove" and cause_contact(env, spec):
        return False, pairs
    manipulated = spec.manipulated_joint.removesuffix("_joint0")
    target = spec.target_joint.removesuffix("_joint0") if spec.target_joint else None
    if target is not None:
        for left, right in pairs:
            if manipulated in left + right and target in left + right:
                return False, pairs
    return True, pairs


def object_workspace_valid(env) -> tuple[bool, dict[str, list[float]]]:
    positions: dict[str, list[float]] = {}
    valid = True
    for name, model in sorted(env.env.objects_dict.items()):
        joints = list(model.joints)
        if not joints:
            continue
        qpos = joint_qpos(env, joints[0])
        positions[name] = qpos.tolist()
        valid &= bool(
            np.isfinite(qpos).all()
            and -0.60 <= qpos[0] <= 0.60
            and -0.60 <= qpos[1] <= 0.60
            and 0.50 <= qpos[2] <= 1.50
        )
    return bool(valid), positions


def generate_task(
    task: str,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    env, _ = make_env(task, seed=reset_seed(task, 0))
    states: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    try:
        candidate_offset = 0
        while len(states) < len(ALL_INIT_IDS):
            init_id = len(states)
            seed = reset_seed(task, candidate_offset)
            candidate_offset += 1
            env.seed(seed)
            env.reset()
            state = np.ascontiguousarray(env.get_sim_state(), dtype=np.float64)
            feature = feature_from_obs(current_observation(env))
            first_hash = state_sha256(state)
            collision_valid, contacts = initial_collision_valid(env, task)
            workspace_valid, positions = object_workspace_valid(env)
            initial_predicates = list(env.env.parsed_problem.get("initial_state", []))
            not_success = not bool(env.check_success())
            finite = bool(np.isfinite(state).all() and np.isfinite(feature).all())

            env.seed(seed)
            env.reset()
            repeated = np.ascontiguousarray(env.get_sim_state(), dtype=np.float64)
            deterministic = state_sha256(repeated) == first_hash
            valid = bool(
                finite and collision_valid and workspace_valid and not_success and deterministic
            )
            if not valid:
                rejections.append(
                    {
                        "task": task,
                        "candidate_reset_seed": seed,
                        "finite": finite,
                        "collision_valid": collision_valid,
                        "workspace_valid": workspace_valid,
                        "task_not_successful": not_success,
                        "deterministic_repeat": deterministic,
                        "contacts": contacts,
                        "reason": "outcome_blind_initial_validity_gate",
                    }
                )
                print(
                    f"INIT_REJECT task={task} seed={seed} collision={int(collision_valid)} "
                    f"workspace={int(workspace_valid)} finite={int(finite)}",
                    flush=True,
                )
                if candidate_offset > 10_000:
                    raise RuntimeError(f"could not obtain 100 valid reset states for {task}")
                continue
            states.append(state)
            records.append(
                {
                    "task": task,
                    "init_state_id": init_id,
                    "reset_seed": seed,
                    "state_hash": first_hash,
                    "state_shape": list(state.shape),
                    "feature_hash": sha256_array(feature, np.float32),
                    "finite": finite,
                    "initial_bddl_predicates": initial_predicates,
                    "task_not_successful": not_success,
                    "collision_valid": collision_valid,
                    "contacts": contacts,
                    "workspace_valid": workspace_valid,
                    "object_qpos": positions,
                    "deterministic_repeat": deterministic,
                    "valid": valid,
                    "outcomes_inspected": False,
                }
            )
            print(f"INIT_POOL task={task} id={init_id:02d} valid=1", flush=True)
    finally:
        env.close()
    return np.stack(states), records, rejections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = ARTIFACT_ROOT / "init_pool"
    npz_path = output / "init_states.npz"
    if npz_path.exists() and not args.force:
        print(json.dumps({"status": "RESUME", "path": str(npz_path)}, sort_keys=True))
        return
    output.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for task in TASKS:
        arrays[task], task_records, task_rejections = generate_task(task)
        records.extend(task_records)
        rejections.extend(task_rejections)
    temporary = output / ".init_states.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(npz_path)
    splits = {
        "schema_version": 1,
        "infrastructure": list(INFRASTRUCTURE_IDS),
        "calibration": list(CALIBRATION_IDS),
        "evaluation": list(EVALUATION_IDS),
        "reserve_untouched_after_freeze": list(RESERVE_IDS),
        "pairwise_disjoint": True,
        "split_before_outcomes": True,
    }
    pool_digest = hashlib.sha256()
    for task in TASKS:
        pool_digest.update(task.encode() + b"\0")
        pool_digest.update(np.ascontiguousarray(arrays[task], dtype=np.float64).tobytes())
    manifest = {
        "schema_version": 1,
        "tasks": {
            task: {
                "count": int(len(arrays[task])),
                "shape": list(arrays[task].shape),
                "array_sha256": sha256_array(arrays[task], np.float64),
                "unique_state_hashes": len(
                    {row["state_hash"] for row in records if row["task"] == task}
                ),
                "all_valid": all(row["valid"] for row in records if row["task"] == task),
            }
            for task in TASKS
        },
        "pool_ordered_bytes_sha256": pool_digest.hexdigest(),
        "npz_sha256": sha256_file(npz_path),
        "states": records,
        "rejected_reset_candidates": rejections,
        "reserve_policy": "generated_and_hashed_only; downstream Stage-2D loaders reject ids 80-99",
        "outcomes_inspected": False,
    }
    atomic_write_json(output / "manifest.json", manifest)
    atomic_write_json(output / "splits.json", splits)
    report = (
        "# Fresh init-state pool\n\n"
        "Generated 100 deterministic reset-randomized, outcome-blind initial states per task. "
        "Every state passed finite-state, initial non-success, collision, workspace, and exact "
        "same-seed reset checks. IDs 80–99 are frozen reserve and are rejected by downstream loaders.\n\n"
        f"Pool hash: `{manifest['pool_ordered_bytes_sha256']}`.\n"
    )
    (output / "report.md").write_text(report)
    if MIRROR_EXPERIMENT_OUTPUTS:
        mirror = EXPERIMENT_ROOT / "init_pool"
        mirror.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "splits.json", "report.md"):
            (mirror / name).write_bytes((output / name).read_bytes())
    print(
        json.dumps(
            {
                "status": "PASS",
                "states_per_task": 100,
                "pool_sha256": manifest["pool_ordered_bytes_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
