"""Submit one prepared S1 template through the canonical controller and register it."""
from __future__ import annotations
import argparse,datetime as dt,fcntl,json,os,re,subprocess
from pathlib import Path
REG=Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/pai-job-registry")
BASE=Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2f/s1-20260907/control")
def write(path,value):
    tmp=path.with_name(path.name+".submit.tmp");tmp.write_text(json.dumps(value,indent=2)+"\n");os.replace(tmp,path)
def submit(template,run_id,phase,task):
    os.chdir(REG)
    if not re.fullmatch("[A-Za-z0-9][A-Za-z0-9._-]{2,63}",run_id):raise ValueError("run id")
    target=REG/"runs"/run_id
    if (target/"result.json").exists():raise RuntimeError("run already sealed; reconciliation required")
    # All CLI output stays private; publish only allowlisted run and resource data.
    completed=subprocess.run([str(REG/"bin/pai-job"),"submit",str(Path(template).resolve()),"--run-id",run_id,
        "--config","/workspace/leon/.dlc/config"],capture_output=True,text=True,timeout=1200)
    rp=target/"result.json"
    if not rp.is_file():
        raise RuntimeError(f"no CreateJob receipt; canonical sealed run needs inspection, rc={completed.returncode}")
    result=json.loads(rp.read_text());jid=result.get("job_id")
    if not re.fullmatch("dlc[a-z0-9]{8,}",jid or ""):
        raise RuntimeError("uncertain CreateJob receipt; claim preserved and no automatic retry")
    resolved=json.loads((target/"resolved.json").read_text());e=resolved["evidence"]
    entry=dict(run_id=run_id,root_run_id=run_id,job_id=jid,phase=phase,task=task,gpus=2,
        source_commit=e["source_commit"],source_tree=e["source_tree"],status="Created",
        created_at_utc=result["finished_at_utc"],artifact_dir=resolved["artifact_dir"],
        canonical_submission_state=result["submission_state"],canonical_exit_code=completed.returncode,
        persisted_completion_verified=False)
    manifest=BASE/"jobs.json"
    with manifest.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);state=json.loads(manifest.read_text())
        if any(j["job_id"]==jid or j["run_id"]==run_id for j in state["jobs"]):
            raise RuntimeError("duplicate registration requires reconciliation")
        claim=state["submission_claims"][run_id]
        entry["created_at_utc"]=claim["claimed_at_utc"]
        claim.update(job_id=jid,state="registered")
        state["jobs"].append(entry);write(manifest,state)
    print(json.dumps(entry,indent=2))
    if completed.returncode!=0:raise RuntimeError("JobId registered for external control; canonical readback needs reconciliation")
    return entry
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--template",required=True);p.add_argument("--run-id",required=True)
    p.add_argument("--phase",choices=("collect","grid","atlas"),required=True);p.add_argument("--task",required=True)
    submit(**vars(p.parse_args()))
