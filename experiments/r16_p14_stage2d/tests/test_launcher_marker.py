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
