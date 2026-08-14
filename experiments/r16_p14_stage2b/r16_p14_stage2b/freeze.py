from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r16_p14_stage2a.envs import configure_libero, get_suite
from r16_p14_stage2a.settings import TASK_SPECS

from .io_utils import atomic_write_json, atomic_write_text, canonical_json, sha256_array, sha256_file, sha256_json
from .settings import (
    ACTOR_SEEDS,
    ALL_INIT_IDS,
    CHECKPOINTS,
    LIBERO_CONFIG,
    PRIMARY_TASKS,
    PROJECT_ROOT,
    STAGE1B_COMMIT,
    STAGE1B_TREE,
    STAGE1_COMMIT,
    STAGE1_TREE,
    STAGE2A_ARTIFACT_ROOT,
    STAGE2A_EVIDENCE_COMMIT,
    STAGE2A_EVIDENCE_TREE,
    STAGE2A_EXPERIMENT_ROOT,
    STAGE2A_REPORT_COMMIT,
    STAGE2A_REPORT_TREE,
)


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def git_object(commit: str) -> dict[str, Any]:
    try:
        resolved = git("rev-parse", commit)
        tree = git("rev-parse", f"{commit}^{{tree}}")
        return {"commit": resolved, "tree": tree, "resolvable": True}
    except subprocess.CalledProcessError:
        return {"commit": commit, "tree": None, "resolvable": False}


def payload_contract(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    normalization = {
        key: np.asarray(value, dtype=np.float32).tolist()
        for key, value in payload["normalization"].items()
    }
    model_hashes = {
        key: sha256_array(value.detach().cpu().numpy())
        for key, value in sorted(payload["model"].items())
    }
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "seed": int(payload["seed"]),
        "run_id": payload["run_id"],
        "step": int(payload["step"]),
        "model_config": payload["model_config"],
        "model_config_sha256": sha256_json(payload["model_config"]),
        "normalization_sha256": sha256_json(normalization),
        "model_state_combined_sha256": sha256_json(model_hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    configure_libero(LIBERO_CONFIG)
    suite = get_suite(LIBERO_CONFIG)
    init_contract: dict[str, Any] = {}
    for task in PRIMARY_TASKS:
        spec = TASK_SPECS[task]
        states = np.asarray(suite.get_task_init_states(spec.task_id), dtype=np.float64)
        hashes = [sha256_array(state, np.float64) for state in states[:50]]
        init_contract[task] = {
            "available_count": int(len(states)),
            "first_50_distinct": len(set(hashes)) == 50,
            "combined_first_50_sha256": sha256_array(states[:50], np.float64),
            "per_init_sha256": {str(index): hashes[index] for index in ALL_INIT_IDS},
        }
    immutable_files = {
        "stage1_decision": PROJECT_ROOT / "experiments/r16_p14_libero_stage1/decision.json",
        "stage1b_decision": PROJECT_ROOT / "experiments/r16_p14_libero_stage1b/decision.json",
        "stage1b_report": PROJECT_ROOT / "experiments/r16_p14_libero_stage1b/reports/REPORT.md",
        "stage2a_decision": STAGE2A_EXPERIMENT_ROOT / "decision.json",
        "stage2a_report": STAGE2A_EXPERIMENT_ROOT / "reports/REPORT.md",
        "stage2a_summary": STAGE2A_EXPERIMENT_ROOT / "reports/first_checkpoint_summary.json",
        "stage2a_frozen_perturbations": STAGE2A_ARTIFACT_ROOT / "perturbations/frozen_parameters.json",
        "stage2a_actor_model_source": STAGE2A_EXPERIMENT_ROOT / "r16_p14_stage2a/actor_model.py",
        "stage2a_actor_settings_source": STAGE2A_EXPERIMENT_ROOT / "r16_p14_stage2a/settings.py",
        "libero_config": LIBERO_CONFIG / "config.yaml",
        "stage2a_libero_source_manifest": STAGE2A_ARTIFACT_ROOT / "source_freeze/source_manifest.json",
        "stage2a_libero_source_hashes": STAGE2A_ARTIFACT_ROOT / "source_freeze/libero_source_files.sha256",
    }
    line_records = []
    file_contract = {}
    for name, path in immutable_files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        file_contract[name] = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest}
        line_records.append(f"{digest}  {path.relative_to(PROJECT_ROOT)}\n")
    historical_unresolvable = git_object("a1b61194a8382f5b1a247b9cd9b140645ff2aeb8")
    result = {
        "schema_version": 1,
        "repository": {
            "head": git("rev-parse", "HEAD"),
            "tree": git("rev-parse", "HEAD^{tree}"),
            "branch": git("branch", "--show-current"),
            "remote": git("remote", "get-url", "origin"),
            "status": git("status", "--short"),
        },
        "stage1": {**git_object(STAGE1_COMMIT), "expected_tree": STAGE1_TREE},
        "stage1b": {
            **git_object(STAGE1B_COMMIT),
            "expected_tree": STAGE1B_TREE,
            "universal_hypothesis": "KILLED_IMMUTABLE",
        },
        "stage2a_evidence": {**git_object(STAGE2A_EVIDENCE_COMMIT), "expected_tree": STAGE2A_EVIDENCE_TREE},
        "stage2a_report": {**git_object(STAGE2A_REPORT_COMMIT), "expected_tree": STAGE2A_REPORT_TREE},
        "historical_stage2a_freeze_reference": {
            **historical_unresolvable,
            "status": "recorded_but_not_resolvable_in_current_repository",
            "authoritative_replacement": STAGE1_COMMIT,
        },
        "immutable_files": file_contract,
        "actor_checkpoints": {str(seed): payload_contract(CHECKPOINTS[seed]) for seed in ACTOR_SEEDS},
        "task_init_states": init_contract,
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "numpy": np.__version__,
        },
    }
    atomic_write_text(output / "immutable_files.sha256", "".join(line_records))
    atomic_write_json(output / "source_manifest.json", result)
    atomic_write_json(output / "task_init_states.json", init_contract)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
