#!/usr/bin/env python3
"""Verify the published R16-P14 formal-pilot evidence without LIBERO."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


TASKS = (
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack",
)
SEEDS = (7, 17, 29)
HORIZONS = (1, 4, 8, 16)
RUN_ID = "r16p14-libero-stage1-pilot-v1"
DECISION = "REVISE_TASKS_OR_PERTURBATIONS"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expect_sha(path: Path, expected: str) -> None:
    actual = sha256(path)
    assert actual == expected, f"SHA-256 mismatch for {path}: {actual} != {expected}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    experiment = root / "experiments/r16_p14_libero_stage1"
    formal = root / "artifacts/formal_pilot"
    checkpoints = root / "artifacts/checkpoints"
    manifest = load_json(experiment / "source_manifest.json")
    summary = load_json(formal / "report/experiment_summary.json")

    assert summary["status"] == "complete"
    assert summary["run_id"] == RUN_ID
    assert summary["decision"] == DECISION
    assert len(summary["training"]) == 9
    assert {(row["task"], row["seed"], row["step"]) for row in summary["training"]} == {
        (task, seed, 8000) for task in TASKS for seed in SEEDS
    }
    assert sum(bool(value) for value in summary["gates"].values()) == 1
    assert summary["gates"]["two_distinct_cause_operator_winners"] is True

    clean_rows = 0
    oracle_rows = 0
    parity_checks = 0
    for task in TASKS:
        shard = formal / "shards" / task / RUN_ID
        episodes = load_jsonl(shard / "baseline_eval/episodes.jsonl")
        assert len(episodes) == 120, f"unexpected clean count for {task}"
        coverage = Counter(
            (row["policy_seed"], row["execution_horizon"]) for row in episodes
        )
        assert coverage == Counter(
            {(seed, horizon): 10 for seed in SEEDS for horizon in HORIZONS}
        )
        assert all(row["task"] == task and row["run_id"] == RUN_ID for row in episodes)
        clean_rows += len(episodes)

        oracle_path = (
            shard / "oracle_audit" / task / "oracle_prefix_labels.jsonl"
        )
        oracle = load_jsonl(oracle_path)
        assert len(oracle) == 30, f"unexpected oracle count for {task}"
        assert all(row["task"] == task and row["run_id"] == RUN_ID for row in oracle)
        assert all(row["model_seed"] == 7 and row["prefix_stride"] == 2 for row in oracle)
        oracle_rows += len(oracle)

        parity = load_json(shard / "baseline_eval/instrumentation_parity.json")
        assert parity["passed"] is True
        assert parity["enabled"]["action_sha256"] == parity["disabled"]["action_sha256"]
        assert parity["enabled"]["chunk_hashes"] == parity["disabled"]["chunk_hashes"]
        assert parity["enabled"]["state_hashes"] == parity["disabled"]["state_hashes"]
        assert parity["enabled"]["contact_hashes"] == parity["disabled"]["contact_hashes"]
        parity_checks += 1

        assert summary["oracle"][task]["candidate_count"] == 30

    checkpoint_count = 0
    for task in TASKS:
        for seed in SEEDS:
            directory = checkpoints / task / f"seed_{seed}"
            complete = load_json(directory / ".complete.json")
            training = load_json(directory / ".training_complete.json")
            checkpoint = directory / complete["checkpoint"]
            assert complete["step"] == 8000
            expect_sha(checkpoint, complete["checkpoint_sha256"])
            assert training["task"] == task
            assert training["seed"] == seed
            assert training["step"] == 8000
            assert training["run_id"] == RUN_ID
            checkpoint_count += 1

    artifact_hashes = manifest["formal_artifacts"]
    hashed_artifacts = {
        "training_complete_sha256": formal / "TRAINING_COMPLETE.json",
        "evaluation_complete_sha256": formal / "EVALUATION_COMPLETE.json",
        "experiment_summary_sha256": formal / "report/experiment_summary.json",
        "decision_sha256": formal / "report/decision.json",
        "report_sha256": formal / "report/report.md",
    }
    for key, path in hashed_artifacts.items():
        expect_sha(path, artifact_hashes[key])

    launcher = root / "infra/pai/launchers/r16p14_libero_stage1.sh"
    expect_sha(launcher, manifest["formal_run"]["payload_sha256"])
    expect_sha(
        root / "artifacts/SHA256SUMS",
        manifest["release_artifacts_manifest_sha256"],
    )
    expect_sha(
        experiment / "libero_config/config.yaml",
        manifest["libero_config_sha256"],
    )

    training_marker = load_json(formal / "TRAINING_COMPLETE.json")
    evaluation_marker = load_json(formal / "EVALUATION_COMPLETE.json")
    assert training_marker["status"] == "complete" and len(training_marker["results"]) == 9
    assert evaluation_marker["status"] == "complete"
    assert evaluation_marker["decision"] == DECISION

    job = load_json(root / "infra/pai/successful_job_summary.json")
    assert job["job_id"] == manifest["formal_run"]["pai_job_id"]
    assert job["status"] == "Succeeded"
    assert job["resource_contract"]["gpu_count"] == 2
    assert job["resource_contract"]["observed_platform_restarts"] == 0

    result = {
        "status": "passed",
        "decision": DECISION,
        "formal_run": manifest["formal_run"]["registry_run_id"],
        "pai_job_id": job["job_id"],
        "clean_rollouts": clean_rows,
        "oracle_candidates": oracle_rows,
        "complete_checkpoints": checkpoint_count,
        "instrumentation_parity_checks": parity_checks,
        "verified_formal_artifact_hashes": len(hashed_artifacts) + 3,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
