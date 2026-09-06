"""Submit one prepared S1 template through the canonical controller and register it."""
from __future__ import annotations
import argparse,datetime as dt,fcntl,json,os,re,subprocess
import errno,time
from pathlib import Path
REG=Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/pai-job-registry")
BASE=Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2f/s1-20260907/control")
TRANSIENT_READ_ERRNOS={errno.ENOENT,errno.ESTALE}
READ_ATTEMPTS=8
READ_DELAY=0.25
def read_json_retry(path,attempts=READ_ATTEMPTS,delay=READ_DELAY):
    """Read sealed submit artifacts without retrying CreateJob itself."""
    path=Path(path)
    if attempts<1:raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:return json.loads(path.read_text())
        except OSError as exc:
            if exc.errno not in TRANSIENT_READ_ERRNOS or attempt==attempts-1:raise
        except json.JSONDecodeError:
            if attempt==attempts-1:raise
        time.sleep(delay)
    raise RuntimeError("unreachable JSON read retry")
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
    if target.is_dir():
        for label,value in (("wrapper.stdout.txt",completed.stdout),("wrapper.stderr.txt",completed.stderr)):
            log=target/label;log.write_text(value);log.chmod(0o600)
    rp=target/"result.json"
    if not rp.is_file():
        raise RuntimeError(f"no CreateJob receipt; canonical sealed run needs inspection, rc={completed.returncode}")
    result=read_json_retry(rp);jid=result.get("job_id")
    if not re.fullmatch("dlc[a-z0-9]{8,}",jid or ""):
        raise RuntimeError("uncertain CreateJob receipt; claim preserved and no automatic retry")
    resolved=read_json_retry(target/"resolved.json");e=resolved["evidence"]
    entry=dict(run_id=run_id,root_run_id=run_id,job_id=jid,phase=phase,task=task,gpus=2,
        source_commit=e["source_commit"],source_tree=e["source_tree"],status="Created",
        created_at_utc=result["finished_at_utc"],artifact_dir=resolved["artifact_dir"],
        canonical_submission_state=result["submission_state"],canonical_exit_code=completed.returncode,
        persisted_completion_verified=False)
    manifest=BASE/"jobs.json"
    with manifest.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);state=read_json_retry(manifest)
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
