from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch

from r16_p14_stage1.data import Normalization
from r16_p14_stage1.envs import (
    action_chunk_sha256,
    contact_pairs,
    feature_from_obs,
    make_env,
    restore_state,
    state_sha256,
)
from r16_p14_stage1.model import ChunkedBCMLP, predict_chunk
from r16_p14_stage1.perturbations import safety_violation
from r16_p14_stage1.settings import FEATURE_KEYS, TASK_SPECS


TASK = "put_the_bowl_on_the_plate"
RUN_ID = "r16p14-libero-stage1-pilot-v1"
OBJECT_JOINTS = ("akita_black_bowl_1_joint0", "plate_1_joint0")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _hash_arrays(items: Iterable[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(items, key=lambda item: item[0]):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _contact_hash(sequence: list[tuple[tuple[str, str], ...]]) -> str:
    return hashlib.sha256(repr(sequence).encode("utf-8")).hexdigest()


def _json_safe_observations(obs: dict[str, Any]) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    for key in FEATURE_KEYS:
        output[key] = np.asarray(obs[key], dtype=np.float64).reshape(-1).tolist()
    return output


def _current_observations(env) -> dict[str, Any]:
    env._post_process()
    env._update_observables(force=True)
    return env.env._get_observations()


def _robot_state(env) -> dict[str, Any]:
    robot = env.robots[0]
    joint_names = list(robot.robot_joints)
    joint_positions = [
        float(np.asarray(env.sim.data.get_joint_qpos(name)).reshape(-1)[0])
        for name in joint_names
    ]
    joint_velocities = [
        float(np.asarray(env.sim.data.get_joint_qvel(name)).reshape(-1)[0])
        for name in joint_names
    ]
    return {
        "joint_names": joint_names,
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
    }


def _branch_signature(env, perturbation_joint: str) -> dict[str, Any]:
    simulator_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
    obs = _current_observations(env)
    feature = feature_from_obs(obs)
    object_poses = {
        name: np.asarray(env.sim.data.get_joint_qpos(name), dtype=np.float64)
        .reshape(-1)
        .tolist()
        for name in OBJECT_JOINTS
    }
    robot = _robot_state(env)
    contacts = contact_pairs(env)
    controller_obs = _json_safe_observations(obs)
    controller_arrays = [
        (key, np.asarray(value, dtype=np.float64))
        for key, value in controller_obs.items()
    ]
    bowl_z = object_poses["akita_black_bowl_1_joint0"][2]
    perturbation_state = np.asarray(
        env.sim.data.get_joint_qpos(perturbation_joint), dtype=np.float64
    ).reshape(-1)
    return {
        "full_simulator_state": simulator_state.tolist(),
        "full_simulator_state_hash": state_sha256(simulator_state),
        "policy_observation_feature": np.asarray(feature, dtype=np.float64).tolist(),
        "policy_observation_feature_hash": _hash_arrays(
            [("feature", np.asarray(feature, dtype=np.float32))]
        ),
        "object_joint_poses": object_poses,
        "object_joint_poses_hash": _hash_arrays(
            [(name, np.asarray(value, dtype=np.float64)) for name, value in object_poses.items()]
        ),
        "robot_state": robot,
        "robot_state_hash": _hash_arrays(
            [
                ("qpos", np.asarray(robot["joint_positions"], dtype=np.float64)),
                ("qvel", np.asarray(robot["joint_velocities"], dtype=np.float64)),
            ]
        ),
        "contact_pairs": [list(pair) for pair in contacts],
        "contact_pairs_hash": hashlib.sha256(repr(contacts).encode("utf-8")).hexdigest(),
        "task_phase": {"object_lifted": bool(bowl_z > 0.94)},
        "perturbation_state": perturbation_state.tolist(),
        "perturbation_state_hash": _hash_arrays(
            [(perturbation_joint, perturbation_state)]
        ),
        "success": bool(env.check_success()),
        "controller_visible_observation": controller_obs,
        "controller_visible_observation_hash": _hash_arrays(controller_arrays),
    }


def _apply_recorded_perturbation(
    env, perturbation: dict[str, Any]
) -> dict[str, Any]:
    joint = str(perturbation["joint"])
    actual_before = np.asarray(
        env.sim.data.get_joint_qpos(joint), dtype=np.float64
    ).reshape(-1)
    recorded_before = np.asarray(perturbation["before_qpos"], dtype=np.float64)
    recorded_after = np.asarray(perturbation["after_qpos"], dtype=np.float64)
    if actual_before.shape != recorded_after.shape:
        raise ValueError(
            f"perturbation joint shape mismatch: {actual_before.shape} != {recorded_after.shape}"
        )
    env.sim.data.set_joint_qpos(joint, recorded_after.copy())
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    applied_after = np.asarray(
        env.sim.data.get_joint_qpos(joint), dtype=np.float64
    ).reshape(-1)
    return {
        "joint": joint,
        "actual_before": actual_before.tolist(),
        "recorded_before": recorded_before.tolist(),
        "before_max_abs_error": float(np.max(np.abs(actual_before - recorded_before))),
        "applied_after": applied_after.tolist(),
        "recorded_after": recorded_after.tolist(),
        "after_max_abs_error": float(np.max(np.abs(applied_after - recorded_after))),
        "immediate_contact_pairs": [list(pair) for pair in contact_pairs(env)],
    }


def _load_demo(task: str, demo_id: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(TASK_SPECS[task].hdf5_path, "r") as dataset:
        demo = dataset[f"data/demo_{demo_id}"]
        return (
            np.asarray(demo["states"], dtype=np.float64),
            np.asarray(demo["actions"], dtype=np.float32),
        )


def _load_policy(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ChunkedBCMLP(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    normalization = Normalization.from_dict(payload["normalization"])
    return model, normalization


def _predict_exact_chunk(
    env,
    *,
    anchor_state: np.ndarray,
    model,
    normalization,
    device: torch.device,
    expected_hash: str,
) -> np.ndarray:
    obs = restore_state(env, anchor_state)
    chunk = predict_chunk(model, normalization, feature_from_obs(obs), device)
    actual_hash = action_chunk_sha256(chunk)
    if actual_hash != expected_hash:
        raise RuntimeError(
            "frozen action chunk could not be reconstructed exactly; "
            f"expected={expected_hash} actual={actual_hash} device={device}"
        )
    return chunk


def _run_suffix(env, task: str, suffix: np.ndarray) -> dict[str, Any]:
    contacts: list[tuple[tuple[str, str], ...]] = []
    phase = {
        "object_lifted": bool(
            np.asarray(
                env.sim.data.get_joint_qpos("akita_black_bowl_1_joint0"),
                dtype=np.float64,
            )[2]
            > 0.94
        )
    }
    cause_violation = False
    violation_type = None
    for action in suffix:
        env.step(np.asarray(action, dtype=np.float32))
        contacts.append(contact_pairs(env))
        if not cause_violation:
            cause_violation, violation_type = safety_violation(
                env, task, action=np.asarray(action, dtype=np.float32), phase=phase
            )
    final_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
    return {
        "contact_sequence": [
            [list(pair) for pair in step_contacts] for step_contacts in contacts
        ],
        "contact_sequence_hash": _contact_hash(contacts),
        "task_success": bool(env.check_success()),
        "cause_violation": bool(cause_violation),
        "violation_type": violation_type,
        "final_simulator_state": final_state.tolist(),
        "final_simulator_state_hash": state_sha256(final_state),
    }


def reconstruct_run(
    env,
    *,
    record: dict[str, Any],
    anchor_state: np.ndarray,
    chunk: np.ndarray,
    branch_k: int,
) -> dict[str, Any]:
    restore_state(env, anchor_state)
    prefix_contacts: list[tuple[tuple[str, str], ...]] = []
    perturbation_application = None
    insertion = int(record["insertion_prefix"])
    for index, action in enumerate(chunk[:branch_k]):
        env.step(np.asarray(action, dtype=np.float32))
        prefix_contacts.append(contact_pairs(env))
        if index + 1 == insertion:
            perturbation_application = _apply_recorded_perturbation(
                env, record["perturbation"]
            )
    if perturbation_application is None:
        raise AssertionError("branch point precedes perturbation insertion")
    signature = _branch_signature(env, record["perturbation"]["joint"])
    suffix = _run_suffix(env, record["task"], chunk[branch_k:])
    return {
        "initializer": "fresh_independent_env_prefix_reconstruction",
        "prefix_contact_sequence": [
            [list(pair) for pair in step_contacts]
            for step_contacts in prefix_contacts
        ],
        "prefix_contact_sequence_hash": _contact_hash(prefix_contacts),
        "perturbation_application": perturbation_application,
        "branch_point": signature,
        "suffix": suffix,
    }


def snapshot_restore_runs(
    env,
    *,
    record: dict[str, Any],
    anchor_state: np.ndarray,
    chunk: np.ndarray,
    branch_k: int,
    repeats: int,
) -> list[dict[str, Any]]:
    restore_state(env, anchor_state)
    insertion = int(record["insertion_prefix"])
    snapshots = [np.asarray(env.get_sim_state(), dtype=np.float64).copy()]
    for index, action in enumerate(chunk):
        env.step(np.asarray(action, dtype=np.float32))
        if index + 1 == insertion:
            _apply_recorded_perturbation(env, record["perturbation"])
        snapshots.append(np.asarray(env.get_sim_state(), dtype=np.float64).copy())
    snapshot = snapshots[branch_k]
    runs = []
    for _ in range(repeats):
        restore_state(env, snapshot)
        runs.append(
            {
                "initializer": "snapshot_restore_control",
                "branch_point": _branch_signature(
                    env, record["perturbation"]["joint"]
                ),
                "suffix": _run_suffix(env, record["task"], chunk[branch_k:]),
            }
        )
    return runs


def compare_runs(runs: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("comparison requires at least two runs")
    reference = runs[0]
    branch_hash_fields = (
        "full_simulator_state_hash",
        "policy_observation_feature_hash",
        "object_joint_poses_hash",
        "robot_state_hash",
        "contact_pairs_hash",
        "perturbation_state_hash",
        "controller_visible_observation_hash",
    )
    branch_exact_by_field = {
        field: all(
            run["branch_point"][field] == reference["branch_point"][field]
            for run in runs[1:]
        )
        for field in branch_hash_fields
    }
    prefix_contact_match = all(
        run.get("prefix_contact_sequence_hash")
        == reference.get("prefix_contact_sequence_hash")
        for run in runs[1:]
    )
    suffix_contact_match = all(
        run["suffix"]["contact_sequence_hash"]
        == reference["suffix"]["contact_sequence_hash"]
        for run in runs[1:]
    )
    outcome_match = all(
        (
            run["suffix"]["task_success"],
            run["suffix"]["cause_violation"],
            run["suffix"]["violation_type"],
        )
        == (
            reference["suffix"]["task_success"],
            reference["suffix"]["cause_violation"],
            reference["suffix"]["violation_type"],
        )
        for run in runs[1:]
    )
    reference_final = np.asarray(
        reference["suffix"]["final_simulator_state"], dtype=np.float64
    )
    final_errors = [
        float(
            np.max(
                np.abs(
                    reference_final
                    - np.asarray(
                        run["suffix"]["final_simulator_state"], dtype=np.float64
                    )
                )
            )
        )
        for run in runs[1:]
    ]
    max_final_error = max(final_errors, default=0.0)
    state_exact = all(branch_exact_by_field.values())
    passed = bool(
        state_exact
        and prefix_contact_match
        and suffix_contact_match
        and outcome_match
        and max_final_error <= tolerance
    )
    return {
        "repeat_count": len(runs),
        "branch_point_exact_by_field": branch_exact_by_field,
        "branch_point_exact": state_exact,
        "prefix_contact_sequence_match": prefix_contact_match,
        "suffix_contact_sequence_match": suffix_contact_match,
        "outcome_match": outcome_match,
        "max_final_state_abs_error": max_final_error,
        "tolerance": tolerance,
        "passed": passed,
    }


def _render_report(summary: dict[str, Any]) -> str:
    old = summary["snapshot_restore_control"]
    new = summary["prefix_reconstruction"]
    return "\n".join(
        [
            "# Phase B — Bowl replay reconstruction repair",
            "",
            "## Result",
            "",
            f"Replay gate: **{'PASS' if summary['gate_passed'] else 'FAIL'}**",
            "",
            "| Initializer | Candidate-level insertion pass | All branch-point pass | Contact/outcome match | Maximum final-state error |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| Stage-1 snapshot restore (published) | {100 * summary['published_stage1_insertion_pass_rate']:.1f}% | n/a | n/a | n/a |",
            f"| Snapshot restore control, 5 repeats | {100 * old['insertion_branch_point_pass_rate']:.1f}% | {100 * old['branch_point_pass_rate']:.1f}% | {100 * old['contact_and_outcome_match_rate']:.1f}% | {old['maximum_final_state_abs_error']:.3g} |",
            f"| Fresh-env prefix reconstruction, 5 repeats | {100 * new['insertion_branch_point_pass_rate']:.1f}% | {100 * new['branch_point_pass_rate']:.1f}% | {100 * new['contact_and_outcome_match_rate']:.1f}% | {new['maximum_final_state_abs_error']:.3g} |",
            "",
            "## Contract",
            "",
            "Each reconstruction starts from one of five fully independent environment instances, restores the demonstration phase anchor, replays the exact frozen CUDA-reconstructed action chunk, applies the recorded environment perturbation at `d`, and continues to the requested branch prefix. No mutated branch state is shared between repetitions.",
            "",
            f"The gate requires at least 99% branch-point pass rate, 100% contact/outcome agreement, and final-state max absolute error no greater than {summary['tolerance']:.1e}.",
            "",
            "## Runtime note",
            "",
            "Simulation and reconstruction are CPU-side. A single local A800 is used only for the frozen MLP forward pass because the Stage-1 action hashes were generated with PyTorch 2.5.1+cu124 CUDA; CPU inference produces different floating-point bytes and is rejected by the action-hash contract.",
            "",
            "This phase tests branch-state correctness only. It does not establish expert mechanism feasibility or policy performance.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    raw_path = (
        repo_root
        / "artifacts/formal_pilot/shards"
        / TASK
        / RUN_ID
        / "oracle_audit"
        / TASK
        / "oracle_prefix_labels.jsonl"
    )
    records = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.max_candidates]
    if len(records) != args.max_candidates:
        raise AssertionError(
            f"requested {args.max_candidates} candidates, found {len(records)}"
        )
    checkpoint = repo_root / "artifacts/checkpoints" / TASK / "seed_7/checkpoint.pt"
    device = torch.device(args.device)
    model, normalization = _load_policy(checkpoint, device)
    config = repo_root / "experiments/r16_p14_libero_stage1/libero_config"

    reconstruction_envs = [
        make_env(TASK, config_dir=config, seed=0)[0] for _ in range(args.repeats)
    ]
    snapshot_env = make_env(TASK, config_dir=config, seed=0)[0]
    output_rows: list[dict[str, Any]] = []
    action_hash_matches = 0
    started = time.time()
    try:
        for candidate_index, record in enumerate(records):
            states, _ = _load_demo(TASK, int(record["demo_id"]))
            anchor_state = states[int(record["anchor_step"])]
            chunk = _predict_exact_chunk(
                reconstruction_envs[0],
                anchor_state=anchor_state,
                model=model,
                normalization=normalization,
                device=device,
                expected_hash=record["action_chunk_hash"],
            )
            action_hash_matches += 1
            stage1_by_k = {
                int(item["prefix_k"]): item for item in record["prefixes"]
            }
            for branch_k in sorted(stage1_by_k):
                reconstruction_runs = [
                    reconstruct_run(
                        env,
                        record=record,
                        anchor_state=anchor_state,
                        chunk=chunk,
                        branch_k=branch_k,
                    )
                    for env in reconstruction_envs
                ]
                snapshot_runs = snapshot_restore_runs(
                    snapshot_env,
                    record=record,
                    anchor_state=anchor_state,
                    chunk=chunk,
                    branch_k=branch_k,
                    repeats=args.repeats,
                )
                reconstruction_comparison = compare_runs(
                    reconstruction_runs, args.tolerance
                )
                snapshot_comparison = compare_runs(snapshot_runs, args.tolerance)
                stage1_snapshot_hash = stage1_by_k[branch_k]["snapshot_hash"]
                output_rows.append(
                    {
                        "schema_version": 1,
                        "candidate_index": candidate_index,
                        "candidate_id": record["candidate_id"],
                        "demo_id": record["demo_id"],
                        "anchor_step": record["anchor_step"],
                        "insertion_prefix": record["insertion_prefix"],
                        "branch_prefix_k": branch_k,
                        "severity": record["severity"],
                        "action_chunk_hash": record["action_chunk_hash"],
                        "stage1_snapshot_hash": stage1_snapshot_hash,
                        "reconstruction_matches_stage1_snapshot_hash": (
                            reconstruction_runs[0]["branch_point"][
                                "full_simulator_state_hash"
                            ]
                            == stage1_snapshot_hash
                        ),
                        "reconstruction": {
                            "comparison": reconstruction_comparison,
                            "runs": reconstruction_runs,
                        },
                        "snapshot_restore_control": {
                            "comparison": snapshot_comparison,
                            "runs": snapshot_runs,
                        },
                    }
                )
            print(
                f"REPLAY_CANDIDATE_COMPLETE {candidate_index + 1}/{len(records)} "
                f"candidate={record['candidate_id']} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    finally:
        for env in reconstruction_envs:
            env.close()
        snapshot_env.close()

    def method_summary(key: str) -> dict[str, Any]:
        comparisons = [row[key]["comparison"] for row in output_rows]
        insertion = [
            comparison
            for row, comparison in zip(output_rows, comparisons, strict=True)
            if row["branch_prefix_k"] == row["insertion_prefix"]
        ]
        return {
            "branch_point_count": len(comparisons),
            "branch_point_pass_count": sum(int(item["passed"]) for item in comparisons),
            "branch_point_pass_rate": sum(int(item["passed"]) for item in comparisons)
            / len(comparisons),
            "insertion_branch_point_count": len(insertion),
            "insertion_branch_point_pass_count": sum(
                int(item["passed"]) for item in insertion
            ),
            "insertion_branch_point_pass_rate": sum(
                int(item["passed"]) for item in insertion
            )
            / len(insertion),
            "contact_and_outcome_match_count": sum(
                int(
                    item["prefix_contact_sequence_match"]
                    and item["suffix_contact_sequence_match"]
                    and item["outcome_match"]
                )
                for item in comparisons
            ),
            "contact_and_outcome_match_rate": sum(
                int(
                    item["prefix_contact_sequence_match"]
                    and item["suffix_contact_sequence_match"]
                    and item["outcome_match"]
                )
                for item in comparisons
            )
            / len(comparisons),
            "maximum_final_state_abs_error": max(
                item["max_final_state_abs_error"] for item in comparisons
            ),
        }

    reconstruction_summary = method_summary("reconstruction")
    snapshot_summary = method_summary("snapshot_restore_control")
    published_passes = sum(
        int(record["replay_determinism"]["passed"]) for record in records
    )
    raw_match_count = sum(
        int(row["reconstruction_matches_stage1_snapshot_hash"])
        for row in output_rows
    )
    gate_passed = bool(
        args.max_candidates == 30
        and args.repeats == 5
        and reconstruction_summary["branch_point_pass_rate"] >= 0.99
        and reconstruction_summary["contact_and_outcome_match_rate"] == 1.0
        and reconstruction_summary["maximum_final_state_abs_error"]
        <= args.tolerance
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "phase": "B_replay_reconstruction_repair",
        "task": TASK,
        "candidate_count": len(records),
        "repeats_per_branch_point": args.repeats,
        "action_chunk_hash_match_count": action_hash_matches,
        "branch_point_count": len(output_rows),
        "published_stage1_insertion_pass_count": published_passes,
        "published_stage1_insertion_pass_rate": published_passes / len(records),
        "reconstruction_matches_stage1_snapshot_hash_count": raw_match_count,
        "reconstruction_matches_stage1_snapshot_hash_rate": raw_match_count
        / len(output_rows),
        "snapshot_restore_control": snapshot_summary,
        "prefix_reconstruction": reconstruction_summary,
        "tolerance": args.tolerance,
        "gate_passed": gate_passed,
        "required_decision_on_failure": "BLOCKED_BY_REPLAY",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "elapsed_seconds": time.time() - started,
        },
    }
    output_dir = args.output_dir.resolve()
    _atomic_text(
        output_dir / "branch_reconstructions.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
    )
    _atomic_text(
        output_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(output_dir / "report.md", _render_report(summary))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if not 1 <= args.max_candidates <= 30:
        parser.error("--max-candidates must be in [1, 30]")
    return args


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
