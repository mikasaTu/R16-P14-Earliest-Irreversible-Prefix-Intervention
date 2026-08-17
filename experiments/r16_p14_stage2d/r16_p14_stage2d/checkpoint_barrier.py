from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, load_jsonl
from .settings import (
    ARTIFACT_ROOT,
    CALIBRATION_ARMS,
    EXPERIMENT_ROOT,
    IMMUTABLE_PARENT,
    MIRROR_EXPERIMENT_OUTPUTS,
    PREFIX_INDICES,
    PROJECT_ROOT,
    TASKS,
)


def read_json(relative: str) -> dict[str, Any]:
    return json.loads((ARTIFACT_ROOT / relative).read_text())


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-commit",
        help="Committed HEAD containing the imported Phase-1 evidence; defaults to HEAD.",
    )
    args = parser.parse_args()
    evidence_commit = args.evidence_commit or git("rev-parse", "HEAD")
    if git("cat-file", "-t", evidence_commit) != "commit":
        raise RuntimeError("evidence commit is not a Git commit")

    isolation = read_json("branch_isolation/summary.json")
    init_manifest = read_json("init_pool/manifest.json")
    splits = read_json("init_pool/splits.json")
    events = read_json("actor_events/summary.json")
    qualification = read_json("perturbation_qualification/summary.json")
    frozen = read_json("perturbation_qualification/frozen_parameters.json")
    pool_summary = read_json("actor_events/formal_event_pool_summary.json")
    calibration_pool = [
        row
        for row in load_jsonl(ARTIFACT_ROOT / "actor_events/formal_event_pool.jsonl")
        if row["split"] == "calibration"
    ]
    event_gate = {
        task: bool(events.get("availability_gate", {}).get("calibration", {}).get(task, False))
        for task in TASKS
    }
    family_gate = {
        task: frozen["tasks"][task]["status"] == "PASS" for task in TASKS
    }
    blockers = []
    if isolation["status"] != "PASS":
        blockers.append("BLOCKED_BY_BRANCH_ISOLATION")
    if not all(event_gate.values()):
        blockers.append("BLOCKED_BY_EVENT_CONSTRUCTION")
    if not all(family_gate.values()):
        blockers.append("BLOCKED_BY_PERTURBATION_QUALIFICATION")

    atlas_rows = len(calibration_pool) * len(PREFIX_INDICES) * len(CALIBRATION_ARMS)
    payload = {
        "schema_version": 1,
        "barrier": "before_calibration_atlas",
        "evidence_commit": evidence_commit,
        "evidence_tree": git("rev-parse", f"{evidence_commit}^{{tree}}"),
        "immutable_parent": IMMUTABLE_PARENT,
        "files_changed_since_parent": git(
            "diff", "--name-only", f"{IMMUTABLE_PARENT}..{evidence_commit}"
        ).splitlines(),
        "branch_isolation": isolation,
        "init_pool": {
            "pool_ordered_bytes_sha256": init_manifest["pool_ordered_bytes_sha256"],
            "npz_sha256": init_manifest["npz_sha256"],
            "splits": splits,
        },
        "actor_events": {
            "counts": events["counts"],
            "availability_gate": event_gate,
            "error_count": events["error_count"],
            "missing_shards": events["missing_shards"],
        },
        "perturbation_qualification": {
            "status": qualification["status"],
            "family_gate": family_gate,
            "grid": qualification["grid"],
            "frozen_parameters": frozen["tasks"],
        },
        "formal_pool": pool_summary,
        "planned_calibration_atlas_rows": atlas_rows,
        "planned_calibration_dimensions": {
            "event_instances": len(calibration_pool),
            "prefixes": len(PREFIX_INDICES),
            "arms": len(CALIBRATION_ARMS),
        },
        "blockers": blockers,
        "scientific_auto_continue_allowed": not blockers,
        "user_override_diagnostic_continuation": bool(blockers),
        "next_command": (
            "submit Stage2D atlas PAI phase from this committed barrier; preserve "
            "diagnostic_only=true if blockers is nonempty"
        ),
    }
    atomic_write_json(ARTIFACT_ROOT / "checkpoint_barrier.json", payload)
    report = [
        "# Mandatory pre-atlas checkpoint barrier",
        "",
        f"Evidence commit: `{payload['evidence_commit']}`",
        f"Evidence tree: `{payload['evidence_tree']}`",
        f"Branch isolation: **{isolation['status']}**",
        f"Init-pool hash: `{payload['init_pool']['pool_ordered_bytes_sha256']}`",
        f"Event availability: `{event_gate}`",
        f"Perturbation families: `{family_gate}`",
        f"Frozen parameters: `{json.dumps(frozen['tasks'], sort_keys=True)}`",
        f"Planned atlas volume: {atlas_rows} fresh-process branches",
        f"Blockers: `{blockers}`",
        "",
        "If any blocker is present, the explicit user override permits the remaining matrix to run only as diagnostic evidence. It cannot change the failed gate or produce a positive/accepted label.",
        "",
        f"Next command: {payload['next_command']}",
        "",
    ]
    (ARTIFACT_ROOT / "checkpoint_barrier.md").write_text("\n".join(report))
    if MIRROR_EXPERIMENT_OUTPUTS:
        for name in ("checkpoint_barrier.json", "checkpoint_barrier.md"):
            (EXPERIMENT_ROOT / name).write_bytes((ARTIFACT_ROOT / name).read_bytes())
    print(
        json.dumps(
            {
                "status": "PASS" if not blockers else "BLOCKED_DIAGNOSTIC_CONTINUATION",
                "blockers": blockers,
                "planned_atlas_rows": atlas_rows,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
