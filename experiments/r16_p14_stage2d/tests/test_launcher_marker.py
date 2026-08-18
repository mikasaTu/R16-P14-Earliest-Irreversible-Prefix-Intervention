from __future__ import annotations

import json
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


def test_launcher_resume_import_is_hash_checked_and_fail_closed(tmp_path: Path) -> None:
    """Exercise the real shell import against the stopped v4 evidence root."""
    launcher = (EXPERIMENT_ROOT / "pai/launcher.sh").read_text()
    start = launcher.index("import_resume_shards() {")
    end = launcher.index("\nrun_event_worker() {", start)
    import_function = launcher[start:end]
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
            "ARTIFACT_ROOT": str(tmp_path / "stage2d_artifacts"),
            "EXPECTED_SOURCE_COMMIT": current_commit,
            "EXPECTED_SOURCE_TREE": current_tree,
            "RESUME_SOURCE_COMMIT": "c6ddbdac9044466ace601ef354443177d5168456",
            "RESUME_SOURCE_TREE": "fa501143513de15cbec1459baa413b385a943192",
            "RESUME_ROOT": "/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2d/pai",
            "RESUME_FROM_RUN": "r16p14-stage2d-phase2-20260819-v4",
        }
    )
    script = f"""set -Eeuo pipefail
{import_function}
import_resume_shards
import_resume_shards
"""
    completed = subprocess.run(
        ["bash", "-c", script],
        env=base_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    destination = tmp_path / "stage2d_artifacts/confirmatory_evaluation/replay_shards"
    assert len(list(destination.glob("*.json"))) == 188
    receipt = json.loads((tmp_path / "pai_state/RESUME_IMPORT.json").read_text())
    assert receipt["status"] == "IMPORTED_IMMUTABLE_COMPLETED_SHARDS"
    assert receipt["source_replay_shard_count"] == 188
    assert receipt["completed_shards_not_overwritten"] is True
    assert receipt["uid"] == os.getuid()
    assert receipt["gid"] == os.getgid()

    mismatch_env = dict(base_env)
    mismatch_env["ARTIFACT_ROOT"] = str(tmp_path / "mismatch_artifacts")
    mismatch_env["RESUME_SOURCE_COMMIT"] = "0" * 40
    rejected = subprocess.run(
        ["bash", "-c", script],
        env=mismatch_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert not (tmp_path / "mismatch_artifacts/confirmatory_evaluation").exists()
