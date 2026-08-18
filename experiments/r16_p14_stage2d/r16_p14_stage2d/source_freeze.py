from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import robosuite
import torch

from .settings import (
    ACTOR_SEEDS,
    ARTIFACT_ROOT,
    EXPERIMENT_ROOT,
    IMMUTABLE_PARENT,
    IMMUTABLE_TREE,
    PLAN_SHA256,
    PROJECT_ROOT,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def git_text(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def command_text(*args: str) -> str:
    try:
        return subprocess.check_output(list(args), text=True, stderr=subprocess.STDOUT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return f"UNAVAILABLE: {exc}"


def json_digest(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def tensor_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256_bytes(header + b"\0" + array.tobytes())


def checkpoint_record(seed: int) -> dict[str, Any]:
    path = PROJECT_ROOT / f"artifacts/stage2a/actor/checkpoints/seed_{seed}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    normalization = {
        key: tensor_digest(value) for key, value in sorted(payload["normalization"].items())
    }
    return {
        **file_record(path),
        "seed": seed,
        "step": int(payload["step"]),
        "parameter_count": int(payload["parameter_count"]),
        "model_config": payload["model_config"],
        "model_config_sha256": json_digest(payload["model_config"]),
        "normalization_component_sha256": normalization,
        "normalization_sha256": json_digest(normalization),
        "training_contract_sha256": json_digest(payload["training_contract"]),
        "task_names": list(payload["task_names"]),
    }


def immutable_test_suite() -> dict[str, Any]:
    names = [
        line
        for line in git_text("ls-tree", "-r", "--name-only", IMMUTABLE_PARENT).splitlines()
        if ("/tests/" in line or Path(line).name.startswith("test_"))
        and line.endswith(".py")
    ]
    records = []
    combined = hashlib.sha256()
    for name in sorted(names):
        data = subprocess.check_output(
            ["git", "show", f"{IMMUTABLE_PARENT}:{name}"], cwd=PROJECT_ROOT
        )
        digest = sha256_bytes(data)
        records.append({"path": name, "size": len(data), "sha256": digest})
        combined.update(name.encode() + b"\0" + digest.encode() + b"\n")
    return {"count": len(records), "combined_sha256": combined.hexdigest(), "files": records}


def optional_records(paths: list[Path]) -> list[dict[str, Any]]:
    return [file_record(path) for path in paths if path.is_file()]


def manifest() -> dict[str, Any]:
    head = git_text("rev-parse", "HEAD")
    tree = git_text("rev-parse", "HEAD^{tree}")
    if head != IMMUTABLE_PARENT or tree != IMMUTABLE_TREE:
        raise RuntimeError(
            f"source freeze must run on immutable parent; got head={head}, tree={tree}"
        )
    stage1b_decision = PROJECT_ROOT / "experiments/r16_p14_libero_stage1b/decision.json"
    stage1b_report = PROJECT_ROOT / "experiments/r16_p14_libero_stage1b/reports/REPORT.md"
    stage2c_decision = PROJECT_ROOT / "artifacts/stage2c/decision.json"
    stage2c_report = PROJECT_ROOT / "artifacts/stage2c/REPORT.md"
    stage2c_replay_json = PROJECT_ROOT / "artifacts/stage2c/replay_contract_diagnostic.json"
    stage2c_replay_md = PROJECT_ROOT / "artifacts/stage2c/replay_contract_diagnostic.md"
    config = PROJECT_ROOT / "experiments/r16_p14_libero_stage1/libero_config/config.yaml"
    source_plan = Path(
        "/workspace/leon/.codex/attachments/dd3a4b22-d4a7-4749-9ae1-242124fdaf35/pasted-text.txt"
    )
    if not source_plan.is_file() or sha256_file(source_plan) != PLAN_SHA256:
        raise RuntimeError("attached Step6 plan is missing or changed")
    artifact_manifests = optional_records(
        [
            PROJECT_ROOT / "artifacts/stage1b/SHA256SUMS",
            PROJECT_ROOT / "artifacts/stage2a/SHA256SUMS",
            PROJECT_ROOT / "artifacts/stage2b/SHA256SUMS",
            PROJECT_ROOT / "artifacts/stage2c/SHA256SUMS",
            PROJECT_ROOT / "artifacts/stage2c/source_freeze/manifest.json",
        ]
    )
    return {
        "schema_version": 1,
        "captured_utc": "2026-08-18",
        "immutable_parent": {
            "head": head,
            "tree": tree,
            # The original parent-HEAD capture predates the status snapshot;
            # never fabricate a clean status after the fact.  Preserve the
            # missingness explicitly in every regenerated manifest.
            "start_status_short": "UNAVAILABLE_AT_IMMUTABLE_PARENT_CAPTURE",
            "start_status_capture": "not_recorded_at_initial_freeze; historical evidence is missing",
            "subject": git_text("show", "-s", "--format=%s", IMMUTABLE_PARENT),
            "commit_object_sha256": sha256_bytes(
                subprocess.check_output(["git", "cat-file", "commit", IMMUTABLE_PARENT], cwd=PROJECT_ROOT)
            ),
        },
        "source_plan": {
            "attachment_path": str(source_plan),
            "size": int(source_plan.stat().st_size),
            "sha256": PLAN_SHA256,
        },
        "immutable_evidence": {
            "stage1b_decision": file_record(stage1b_decision),
            "stage1b_report": file_record(stage1b_report),
            "stage2c_decision": file_record(stage2c_decision),
            "stage2c_report": file_record(stage2c_report),
            "stage2c_replay_diagnostic_json": file_record(stage2c_replay_json),
            "stage2c_replay_diagnostic_report": file_record(stage2c_replay_md),
        },
        "act_checkpoints": [checkpoint_record(seed) for seed in ACTOR_SEEDS],
        "actor_source": {
            "stage2a_git_tree": git_text("rev-parse", f"{IMMUTABLE_PARENT}:experiments/r16_p14_stage2a/r16_p14_stage2a"),
            "stage2b_git_tree": git_text("rev-parse", f"{IMMUTABLE_PARENT}:experiments/r16_p14_stage2b/r16_p14_stage2b"),
            "stage2c_git_tree": git_text("rev-parse", f"{IMMUTABLE_PARENT}:experiments/r16_p14_stage2c/r16_p14_stage2c"),
        },
        "libero": {
            "source_path": str((PROJECT_ROOT / "libero").resolve()),
            "source_git_tree": git_text("rev-parse", f"{IMMUTABLE_PARENT}:libero"),
            "config": file_record(config),
        },
        "test_suite_at_parent": immutable_test_suite(),
        "artifact_manifests": artifact_manifests,
        "environment": {
            "python_executable": sys.executable,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "mujoco": mujoco.__version__,
            "robosuite": robosuite.__version__,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
            "cpu": command_text("lscpu"),
            "gpus": command_text(
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ),
        },
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "artifact_root": str(ARTIFACT_ROOT),
            "libero_dataset": "/mnt/cpfs/zbl-cpfs-new/dataset/leon/libero/libero_goal",
            "python_environment": str(Path(sys.executable).resolve().parents[1]),
        },
        "feishu": json.loads((EXPERIMENT_ROOT / "feishu_documents.json").read_text()),
    }


def write_manifest(payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    digest_line = f"{sha256_bytes(encoded.encode())}  manifest.json\n"
    for root in (ARTIFACT_ROOT / "source_freeze", EXPERIMENT_ROOT / "source_freeze"):
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text(encoded)
        (root / "manifest.sha256").write_text(digest_line)


def main() -> None:
    payload = manifest()
    write_manifest(payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "head": payload["immutable_parent"]["head"],
                "tree": payload["immutable_parent"]["tree"],
                "checkpoint_count": len(payload["act_checkpoints"]),
                "test_count": payload["test_suite_at_parent"]["count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
