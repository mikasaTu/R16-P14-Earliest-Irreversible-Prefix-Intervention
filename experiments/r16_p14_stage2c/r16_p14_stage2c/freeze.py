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

import numpy as np
import torch

from r16_p14_stage2a.envs import make_env
from r16_p14_stage2a.settings import TASK_SPECS

from .settings import ALL_CANDIDATE_TASKS, ARTIFACT_ROOT, EXPERIMENT_ROOT, PROJECT_ROOT


IMMUTABLE_PARENT = "6eae66d23313cc97231249bfa1c40dc1767ea727"
IMMUTABLE_TREE = "ddcedd60f4f4e2878f8a4400d65e9e888f00cdd1"
START_TRACKED_TREE_LIST_SHA256 = "3efc93a46203ffa87a2168301e4cc8cb96de4d6fb2633688272b59dd9b529028"
START_COMMIT_PAYLOAD_SHA256 = "a5e620f51379fd3d349a2641853d21a64fe3c51dc02bdf5163be70c391ae3707"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def hash_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def init_state_manifest() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in ALL_CANDIDATE_TASKS:
        env, suite = make_env(task)
        try:
            states = np.asarray(
                suite.get_task_init_states(TASK_SPECS[task].task_id), dtype=np.float64
            )
            result[task] = {
                "count": int(len(states)),
                "shape": list(states.shape),
                "sha256": hashlib.sha256(np.ascontiguousarray(states).tobytes()).hexdigest(),
            }
        finally:
            env.close()
    return result


def main() -> None:
    output = EXPERIMENT_ROOT / "source_freeze"
    output.mkdir(parents=True, exist_ok=True)
    stage2b_decision = PROJECT_ROOT / "experiments/r16_p14_stage2b/decision.json"
    stage2b_report = PROJECT_ROOT / "experiments/r16_p14_stage2b/reports/REPORT.md"
    shutil.copyfile(stage2b_decision, output / "stage2b_decision.json")
    shutil.copyfile(stage2b_report, output / "stage2b_REPORT.md")
    checkpoints = [
        PROJECT_ROOT / f"artifacts/stage2a/actor/checkpoints/seed_{seed}.pt"
        for seed in (7, 17, 29)
    ]
    payload = {
        "schema_version": 1,
        "captured_utc": "2026-08-15",
        "immutable_parent": {
            "head": IMMUTABLE_PARENT,
            "tree": IMMUTABLE_TREE,
            "start_status_short": "",
            "tracked_tree_listing_sha256": START_TRACKED_TREE_LIST_SHA256,
            "commit_object_payload_sha256": START_COMMIT_PAYLOAD_SHA256,
            "subject": "Publish Stage2B Step4 final report",
        },
        "stage2b_final_evidence": {
            "commit": IMMUTABLE_PARENT,
            "tree": IMMUTABLE_TREE,
            "decision": {"path": str(stage2b_decision.relative_to(PROJECT_ROOT)), "sha256": sha256_file(stage2b_decision)},
            "report": {"path": str(stage2b_report.relative_to(PROJECT_ROOT)), "sha256": sha256_file(stage2b_report)},
        },
        "act_checkpoints": [
            {"seed": int(path.stem.split("_")[-1]), "path": str(path.relative_to(PROJECT_ROOT)), "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in checkpoints
        ],
        "actor_source": hash_inventory(PROJECT_ROOT / "experiments/r16_p14_stage2a/r16_p14_stage2a"),
        "libero_source_tree": {
            "git_tree": run_git("rev-parse", f"{IMMUTABLE_PARENT}:libero"),
            "config_sha256": sha256_file(PROJECT_ROOT / "experiments/r16_p14_libero_stage1/libero_config/config.yaml"),
        },
        "task_init_states": init_state_manifest(),
        "stage2b_raw_artifacts": hash_inventory(PROJECT_ROOT / "artifacts/stage2b"),
        "test_suite": hash_inventory(PROJECT_ROOT / "experiments/r16_p14_stage2b/tests"),
        "environment": {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
        },
        "feishu": {
            "step5_wiki_node": "SmWbwfM2Yik5kmkVlpKcdvuynmX",
            "step5_doc": "Zz6odH0AYolaLBxzwMmc9Latnqe",
            "report_wiki_node": "BnfvwJCoYi2EOokK072cdmqqneg",
            "report_doc": "OzWNdE2SkoDbL5xq79mc2l5Rnfh",
        },
        "artifact_root": str(ARTIFACT_ROOT),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (output / "manifest.json").write_text(encoded)
    (output / "manifest.sha256").write_text(
        hashlib.sha256(encoded.encode()).hexdigest() + "  manifest.json\n"
    )
    print(json.dumps({"status": "PASS", "manifest": str(output / "manifest.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
