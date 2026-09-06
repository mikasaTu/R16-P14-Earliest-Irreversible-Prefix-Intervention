from __future__ import annotations

import errno
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from experiments.r16_p14_stage2f.pai import control


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_read_json_retry_reopens_after_estale_enoent_and_json(tmp_path):
    target = tmp_path / "jobs.json"
    with patch.object(
        Path,
        "read_text",
        side_effect=[OSError(errno.ESTALE, "stale"), OSError(errno.ENOENT, "gone"), "{\"jobs\": []}"],
    ) as read:
        assert control.read_json_retry(target, attempts=3, delay=0) == {"jobs": []}
    assert read.call_count == 3


def test_read_json_retry_persistent_estale_is_not_success(tmp_path):
    target = tmp_path / "jobs.json"
    with patch.object(Path, "read_text", side_effect=OSError(errno.ESTALE, "stale")) as read:
        with pytest.raises(OSError):
            control.read_json_retry(target, attempts=3, delay=0)
    assert read.call_count == 3


def test_read_json_retry_permission_error_is_not_retried(tmp_path):
    target = tmp_path / "jobs.json"
    with patch.object(Path, "read_text", side_effect=PermissionError(errno.EACCES, "denied")) as read:
        with pytest.raises(PermissionError):
            control.read_json_retry(target, delay=0)
    assert read.call_count == 1


def test_run_keeps_loop_and_fail_closed_on_persistent_manifest_io(tmp_path, monkeypatch):
    manifest = tmp_path / "control" / "jobs.json"
    registry = tmp_path / "registry"
    registry.mkdir()
    known = {
        "jobs": [{
            "job_id": "dlcknow12345678",
            "status": "Running",
            "gpus": 2,
            "created_at_utc": "2026-09-07T00:00:00+00:00",
        }],
        "dev14_gpu_hours_upper_bound": 0.5,
    }
    control.REGISTRY = registry
    calls = []

    class StopLoop(Exception):
        pass

    with patch.object(control, "_cycle", side_effect=[known, OSError(errno.ESTALE, "stale")]), \
         patch.object(control, "cli", side_effect=lambda args: calls.append(args) or ""), \
         patch.object(control.time, "sleep", side_effect=[None, StopLoop]):
        with pytest.raises(StopLoop):
            control.run(manifest, interval=0, once=False)

    heartbeat = json.loads((manifest.parent / "heartbeat.json").read_text())
    assert heartbeat["resume_allowed"] is False
    assert heartbeat["stale"] is True
    assert heartbeat["controller_io_error"] == "OSError"
    assert heartbeat["stop_job_ids"] == ["dlcknow12345678"]
    assert (manifest.parent / "STOP").read_text().strip() in {"CONTROL_IO_FAILURE", "GPU_BUDGET"}
    assert calls == [["stop", "job", "dlcknow12345678", "--force", "--quiet"]]


def test_fail_closed_retries_a_stop_that_failed(tmp_path, monkeypatch):
    base = tmp_path / "control"
    registry = tmp_path / "registry"
    registry.mkdir()
    control.REGISTRY = registry
    known = {"jobs": [{"job_id": "dlcretry12345678", "status": "Running", "gpus": 2,
                        "created_at_utc": "2026-09-07T00:00:00+00:00"}]}
    calls = []

    def stop(args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("temporary stop failure")
        return ""

    monkeypatch.setattr(control, "cli", stop)
    attempted = control._fail_closed(base, known, OSError(errno.ESTALE, "stale"), set())
    assert attempted == set()
    attempted = control._fail_closed(base, known, OSError(errno.ESTALE, "stale"), attempted)
    assert attempted == {"dlcretry12345678"}
    assert len(calls) == 2


def test_active_rejected_still_blocks_resume_and_stops_job(tmp_path, monkeypatch):
    manifest = tmp_path / "control" / "jobs.json"
    registry = tmp_path / "registry"
    registry.mkdir()
    _write(manifest, {
        "jobs": [{
            "job_id": "dlcrejected12345678",
            "status": "Running",
            "gpus": 2,
            "created_at_utc": "2026-09-07T02:00:00+00:00",
            "placement_rejected": True,
        }],
    })
    control.REGISTRY = registry
    calls = []
    monkeypatch.setattr(control, "get_job", lambda jid: {"Status": "Running", "UseOversoldResource": True})
    monkeypatch.setattr(control, "cli", lambda args: calls.append(args) or "")
    control._cycle(manifest, manifest.parent)
    heartbeat = json.loads((manifest.parent / "heartbeat.json").read_text())
    assert heartbeat["resume_allowed"] is False
    assert calls == [["stop", "job", "dlcrejected12345678", "--force", "--quiet"]]
