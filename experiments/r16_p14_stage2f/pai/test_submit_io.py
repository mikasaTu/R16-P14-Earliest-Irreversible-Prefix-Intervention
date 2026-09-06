from __future__ import annotations

import errno
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.r16_p14_stage2f.pai import submit


def test_submit_read_retry_reopens_estale_and_json(tmp_path):
    target = tmp_path / "result.json"
    with patch.object(Path, "read_text", side_effect=[OSError(errno.ESTALE, "stale"), "{", '{"job_id": "dlc12345678"}']) as read:
        assert submit.read_json_retry(target, attempts=3, delay=0) == {"job_id": "dlc12345678"}
    assert read.call_count == 3


def test_submit_read_retry_persistent_enoent_fails_without_submit_retry(tmp_path):
    target = tmp_path / "result.json"
    with patch.object(Path, "read_text", side_effect=OSError(errno.ENOENT, "missing")) as read:
        with pytest.raises(OSError):
            submit.read_json_retry(target, attempts=2, delay=0)
    assert read.call_count == 2
