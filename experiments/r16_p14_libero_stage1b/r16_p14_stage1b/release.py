from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from r16_p14_stage1.io_utils import atomic_write_json


EXPERIMENT_PATH = Path("experiments/r16_p14_libero_stage1b")
ARTIFACT_PATH = Path("artifacts/stage1b")


def count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(repo_root: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    decision = load(repo_root / EXPERIMENT_PATH / "decision.json")
    phase_a = load(repo_root / ARTIFACT_PATH / "offline_reanalysis/summary.json")
    phase_b = load(repo_root / ARTIFACT_PATH / "replay_gate/summary.json")
    phase_c = load(
        repo_root / ARTIFACT_PATH / "expert_chunk_calibration/expert_gate_summary.json"
    )
    checks["final_decision"] = (
        decision["status"] == "complete"
        and decision["decision"] == "KILL_CORE_HYPOTHESIS"
    )
    checks["phase_a_90_records"] = (
        count_jsonl(
            repo_root / ARTIFACT_PATH / "offline_reanalysis/reanalysis.jsonl"
        )
        == 90
    )
    checks["phase_a_bowl_signal_does_not_survive"] = not bool(
        phase_a["tasks"]["put_the_bowl_on_the_plate"]["corrected"][
            "stage1_positive_signal_survives"
        ]
    )
    checks["phase_b_gate_pass"] = bool(
        phase_b["gate_passed"]
        and phase_b["candidate_count"] == 30
        and phase_b["prefix_reconstruction"]["branch_point_pass_count"] == 180
        and phase_b["prefix_reconstruction"]["maximum_final_state_abs_error"] == 0
    )
    calibration_path = (
        repo_root
        / ARTIFACT_PATH
        / "expert_chunk_calibration/calibration_records.jsonl"
    )
    calibration_records = [
        json.loads(line)
        for line in calibration_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record_types: dict[str, int] = {}
    for record in calibration_records:
        record_type = str(record["record_type"])
        record_types[record_type] = record_types.get(record_type, 0) + 1
    checks["phase_c_record_counts"] = (
        len(calibration_records) == 499
        and record_types.get("calibration_nominal") == 440
        and record_types.get("calibration_prefix_audit") == 59
    )
    checks["phase_c_replay_exact"] = (
        phase_c["calibration_prefix_replay"]["pass_count"] == 885
        and phase_c["calibration_prefix_replay"]["branch_point_count"] == 885
    )
    checks["phase_c_gate_fail_one_task"] = (
        not phase_c["gate_passed"]
        and phase_c["qualified_task_count"] == 1
        and phase_c["decision"] == "KILL_CORE_HYPOTHESIS"
    )
    checks["evaluation_and_heldout_uninspected"] = (
        phase_c["evaluation_demo_ids_inspected"] == []
        and phase_c["heldout_demo_ids_inspected"] == []
    )
    checks["phase_d_e_not_executed"] = (
        not phase_c["phase_d_executed"]
        and not phase_c["phase_e_executed"]
        and not decision["training_executed"]
        and not decision["learned_risk_executed"]
    )
    required = [
        EXPERIMENT_PATH / "preregistration.yaml",
        EXPERIMENT_PATH / "metric_contract.md",
        EXPERIMENT_PATH / "source_manifest.json",
        EXPERIMENT_PATH / "commands.sh",
        EXPERIMENT_PATH / "reports/FIRST_CHECKPOINT.md",
        EXPERIMENT_PATH / "reports/REPORT.md",
        ARTIFACT_PATH / "offline_reanalysis/old_to_new_metrics.csv",
        ARTIFACT_PATH / "replay_gate/branch_reconstructions.jsonl",
        ARTIFACT_PATH / "expert_chunk_calibration/grid_summary.csv",
        ARTIFACT_PATH / "expert_chunk_calibration/selected_config_paired_metrics.csv",
        ARTIFACT_PATH / "expert_chunk_calibration/negative_results.md",
        ARTIFACT_PATH / "policy_baseline/NOT_EXECUTED.md",
        ARTIFACT_PATH / "revised_oracle/NOT_EXECUTED.md",
    ]
    checks["required_files_present"] = all(
        (repo_root / relative).is_file() for relative in required
    )
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "status": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "counts": {
            "phase_a_records": 90,
            "phase_b_candidates": phase_b["candidate_count"],
            "phase_b_branch_points": phase_b["branch_point_count"],
            "phase_c_records": len(calibration_records),
            "phase_c_record_types": record_types,
            "phase_c_replay_branch_points": phase_c["calibration_prefix_replay"][
                "branch_point_count"
            ],
        },
    }


def write_test_summary(repo_root: Path) -> dict[str, Any]:
    junit_path = repo_root / ARTIFACT_PATH / "test_results/pytest.xml"
    root = ET.parse(junit_path).getroot()
    suites = list(root.findall("testsuite")) if root.tag == "testsuites" else [root]
    summary = {
        "schema_version": 1,
        "command": (
            "/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python "
            "-m pytest experiments/r16_p14_libero_stage1b/tests -q "
            "--junitxml=artifacts/stage1b/test_results/pytest.xml"
        ),
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "elapsed_seconds": sum(float(suite.attrib.get("time", 0)) for suite in suites),
        "status": "pass",
        "junit_xml": "artifacts/stage1b/test_results/pytest.xml",
    }
    if summary["failures"] or summary["errors"]:
        summary["status"] = "fail"
    atomic_write_json(
        repo_root / ARTIFACT_PATH / "test_results/summary.json", summary
    )
    return summary


def manifest_files(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for root in (repo_root / EXPERIMENT_PATH, repo_root / ARTIFACT_PATH):
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(repo_root)
            if path.name == "SHA256SUMS":
                continue
            if "__pycache__" in relative.parts or path.suffix == ".pyc":
                continue
            if ".tmp" in path.name:
                continue
            paths.append(path)
    return sorted(paths, key=lambda item: str(item.relative_to(repo_root)))


def write_manifest(repo_root: Path) -> dict[str, Any]:
    lines: list[str] = []
    total_bytes = 0
    for path in manifest_files(repo_root):
        content = path.read_bytes()
        total_bytes += len(content)
        digest = hashlib.sha256(content).hexdigest()
        lines.append(f"{digest}  {path.relative_to(repo_root)}")
    target = repo_root / ARTIFACT_PATH / "SHA256SUMS"
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return {"file_count": len(lines), "total_bytes": total_bytes, "path": str(target)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    result = validate(repo_root)
    if args.write_report:
        result["tests"] = write_test_summary(repo_root)
        atomic_write_json(
            repo_root / ARTIFACT_PATH / "release_validation.json", result
        )
    if args.write_manifest:
        result["manifest"] = write_manifest(repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
