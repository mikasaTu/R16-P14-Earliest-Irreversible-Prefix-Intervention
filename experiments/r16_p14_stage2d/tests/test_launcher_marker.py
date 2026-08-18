from __future__ import annotations

import json
import hashlib
import os
import subprocess
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def test_launcher_write_marker_creates_atomic_uid_owned_marker(tmp_path: Path) -> None:
    """Execute the launcher’s actual write_marker function, not a duplicate."""
    launcher = (EXPERIMENT_ROOT / "pai/launcher.sh").read_text()
    start = launcher.index("write_marker() {")
    end = launcher.index("\nwrite_terminal_receipt() {", start)
    marker_function = launcher[start:end]
    artifact_root = tmp_path / "stage2d_artifacts"
    script = f"""set -Eeuo pipefail
{marker_function}
write_marker FIRST_REAL_WORK.json
"""
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": os.environ.get("PYTHON", os.sys.executable),
            "R16_P14_STAGE2D_ARTIFACT_ROOT": str(artifact_root),
            "PAI_CANARY_RUN_ID": "marker-test-run",
            "PHASE": "phase2",
        }
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    marker = artifact_root.parent / "pai_state" / "FIRST_REAL_WORK.json"
    assert marker.is_file()
    payload = json.loads(marker.read_text())
    assert payload["status"] == "complete"
    assert payload["phase"] == "phase2"
    assert payload["registry_run_id"] == "marker-test-run"
    assert payload["uid"] == os.getuid()
    assert payload["gid"] == os.getgid()
    assert marker.stat().st_mode & 0o777 == 0o600


def _import_function() -> str:
    launcher = (EXPERIMENT_ROOT / "pai/launcher.sh").read_text()
    start = launcher.index("import_resume_shards() {")
    end = launcher.index("\nrun_event_worker() {", start)
    return launcher[start:end]


def _write_resume_fixture(root: Path) -> dict:
    run_root = root / "stage2d-resume-fixture-v1"
    source = run_root / "stage2d_artifacts/confirmatory_evaluation/replay_shards"
    source.mkdir(parents=True)
    items = []
    for index in range(2):
        relative = Path("stage2d_artifacts/confirmatory_evaluation/replay_shards") / f"fixture-{index}.json"
        payload = json.dumps({"fixture": index, "value": "immutable"}, sort_keys=True) + "\n"
        path = run_root / relative
        path.write_text(payload)
        items.append(
            {
                "relative_path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "run_id": "stage2d-resume-fixture-v1",
        "job_id": "fixture-job",
        "status": "STOPPED_BY_USER_FOR_MARKER_REPAIR",
        "source_commit": "c6ddbdac9044466ace601ef354443177d5168456",
        "source_tree": "fa501143513de15cbec1459baa413b385a943192",
        "immutable_completed_evidence": True,
        "expected_replay_shard_count": 2,
        "items": items,
    }
    manifest_path = run_root / "pai_state/RECOVERY_MANIFEST.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def _run_import(tmp_path: Path, resume_root: Path, artifact_root: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    current_commit = subprocess.check_output(
        ["git", "-C", str(EXPERIMENT_ROOT.parent.parent), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    current_tree = subprocess.check_output(
        ["git", "-C", str(EXPERIMENT_ROOT.parent.parent), "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()
    base_env = os.environ.copy()
    base_env.update(
        {
            "PYTHON": os.environ.get("PYTHON", os.sys.executable),
            "ARTIFACT_ROOT": str(artifact_root),
            "EXPECTED_SOURCE_COMMIT": current_commit,
            "EXPECTED_SOURCE_TREE": current_tree,
            "RESUME_SOURCE_COMMIT": "c6ddbdac9044466ace601ef354443177d5168456",
            "RESUME_SOURCE_TREE": "fa501143513de15cbec1459baa413b385a943192",
            "RESUME_ROOT": str(resume_root),
            "RESUME_FROM_RUN": "stage2d-resume-fixture-v1",
            "STAGE2D_RESUME_TEST_MODE": "1",
            "PHASE": "phase2",
        }
    )
    base_env.update(overrides)
    script = f"""set -Eeuo pipefail
{_import_function()}
import_resume_shards
"""
    return subprocess.run(
        ["bash", "-c", script],
        env=base_env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_launcher_resume_import_fixture_is_atomic_and_fail_closed(tmp_path: Path) -> None:
    """Portable fixture covers hash checks, no-overwrite, receipt, and mismatch rejection."""
    fixture_root = tmp_path / "resume"
    manifest = _write_resume_fixture(fixture_root)
    artifact_root = tmp_path / "output/stage2d_artifacts"
    completed = _run_import(tmp_path, fixture_root, artifact_root)
    assert completed.returncode == 0, completed.stderr
    destination = artifact_root / "confirmatory_evaluation/replay_shards"
    assert len(list(destination.glob("*.json"))) == len(manifest["items"])
    receipt = json.loads((artifact_root.parent / "pai_state/RESUME_IMPORT.json").read_text())
    assert receipt["status"] == "IMPORTED_IMMUTABLE_COMPLETED_SHARDS"
    assert receipt["source_replay_shard_count"] == len(manifest["items"])
    assert receipt["completed_shards_not_overwritten"] is True
    assert receipt["uid"] == os.getuid()
    assert receipt["gid"] == os.getgid()

    divergent = destination / "fixture-0.json"
    divergent.write_text("tampered\n")
    rejected_divergent = _run_import(tmp_path, fixture_root, artifact_root)
    assert rejected_divergent.returncode != 0
    assert divergent.read_text() == "tampered\n"

    rejected = _run_import(
        tmp_path,
        fixture_root,
        tmp_path / "mismatch_artifacts/stage2d_artifacts",
        RESUME_SOURCE_COMMIT="0" * 40,
    )
    assert rejected.returncode != 0
    assert not (tmp_path / "mismatch_artifacts/stage2d_artifacts/confirmatory_evaluation").exists()

    rejected_phase = _run_import(
        tmp_path,
        fixture_root,
        tmp_path / "wrong-phase/stage2d_artifacts",
        PHASE="phase3",
    )
    assert rejected_phase.returncode != 0
    assert not (tmp_path / "wrong-phase/stage2_artifacts/confirmatory_evaluation").exists()


def test_launcher_resume_import_real_v4_evidence(tmp_path: Path) -> None:
    """Separate real-CPFS integration evidence for the stopped v4 run."""
    real_root = Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2d/pai")
    source_run = real_root / "r16p14-stage2d-phase2-20260819-v4"
    manifest = json.loads((source_run / "pai_state/RECOVERY_MANIFEST.json").read_text())
    assert manifest["status"] == "STOPPED_BY_USER_FOR_MARKER_REPAIR"
    result = _run_import(
        tmp_path,
        real_root,
        tmp_path / "real-output/stage2d_artifacts",
        RESUME_FROM_RUN="r16p14-stage2d-phase2-20260819-v4",
        STAGE2D_RESUME_TEST_MODE="0",
    )
    assert result.returncode == 0, result.stderr
    destination = tmp_path / "real-output/stage2d_artifacts/confirmatory_evaluation/replay_shards"
    assert len(list(destination.glob("*.json"))) == manifest["replay_shard_count"]
