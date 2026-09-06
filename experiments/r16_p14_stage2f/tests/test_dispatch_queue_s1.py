from pathlib import Path
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
EXPERIMENTS=Path(__file__).resolve().parents[2]
for suffix in ("", "r16_p14_stage2a", "r16_p14_stage2b", "r16_p14_stage2c", "r16_p14_stage2d"):
    sys.path.insert(0,str(EXPERIMENTS/suffix))
from r16_p14_stage2f.s1 import dispatch

def large_result_child(request, result_queue):
    result_queue.put({"status":"OK","pid":os.getpid(),"payload":"x"*(2*1024*1024)})

def test_large_spawn_result_is_drained_before_join(monkeypatch):
    monkeypatch.setattr(dispatch,"_child_entry",large_result_child)
    row=dispatch.run_spawned_branch(
        {"event_id":"queue-test","event_instance_id":"queue-test","original_chunk_hash":"chunk"},
        7,"fresh_h4",2,4,8,timeout_s=5.0)
    assert row["status"]=="OK", row.get("error")
    assert len(row["payload"])==2*1024*1024
    assert row["pid"]!=os.getpid()
    assert row["dispatcher_child_exitcode"]==0

def test_resume_compatibility_is_bound_to_exact_dispatch_repair():
    import json
    from r16_p14_stage2f.s1.matrix import compatible_runtime_row
    receipt=json.loads((EXPERIMENTS/"r16_p14_stage2f/s1/dispatch_compatibility.json").read_text())
    current={**receipt["other_module_sha256"],"dispatch.py":receipt["new_dispatch_sha256"]}
    previous={**receipt["other_module_sha256"],"dispatch.py":receipt["old_dispatch_sha256"]}
    row={"source_commit":receipt["allowed_existing_source_commits"][0],"runtime_module_hashes":previous}
    assert compatible_runtime_row(row,current)
    assert not compatible_runtime_row({**row,"source_commit":"0"*40},current)
    altered={**previous,"runtime.py":"0"*64}
    assert not compatible_runtime_row({**row,"runtime_module_hashes":altered},current)
    assert not compatible_runtime_row(row,{**current,"dispatch.py":"0"*64})
