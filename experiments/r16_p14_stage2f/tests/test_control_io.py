"""A transient CPFS inode replacement must retry, not mask persistent errors."""
import errno,json,sys
from pathlib import Path
from unittest.mock import patch
import pytest
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.r16_p14_stage2f.s1.common import read_json_retry
def test_stale_inode_reopens():
    with patch.object(Path,"read_text",side_effect=[OSError(errno.ESTALE,"stale"),'{"time":"ok"}']) as read:
        assert read_json_retry("/unused",delay=0)=={"time":"ok"}
        assert read.call_count==2
def test_persistent_stale_is_not_success():
    with patch.object(Path,"read_text",side_effect=OSError(errno.ESTALE,"stale")) as read:
        with pytest.raises(OSError):read_json_retry("/unused",attempts=3,delay=0)
        assert read.call_count==3
def test_permission_error_not_retried():
    with patch.object(Path,"read_text",side_effect=PermissionError(errno.EACCES,"denied")) as read:
        with pytest.raises(PermissionError):read_json_retry("/unused",delay=0)
        assert read.call_count==1
