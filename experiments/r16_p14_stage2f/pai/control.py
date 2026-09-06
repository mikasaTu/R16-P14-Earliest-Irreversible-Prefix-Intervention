"""External controller policy; state-changing CLI runs only in canonical registry."""
from __future__ import annotations
import runpy
import argparse,datetime as dt,errno,hashlib,json,os,re,subprocess,time
from pathlib import Path
from zoneinfo import ZoneInfo
REGISTRY=Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/pai-job-registry")
ACTIVE={"Created","Preparing","Queuing","Running","Restarting","Pending","Starting","Stopping"}
TERMINAL={"Succeeded","Failed","Stopped","Deleted"}
TRANSIENT_READ_ERRNOS={errno.ENOENT,errno.ESTALE}
READ_ATTEMPTS=8
READ_DELAY=0.25
def now_utc():return dt.datetime.now(dt.timezone.utc)
def blackout(t=None):
    t=(t or now_utc()).astimezone(ZoneInfo("Asia/Shanghai"));m=t.hour*60+t.minute
    return 565<=m<580 or 1165<=m<1180
def conservative_hours(jobs,t=None):
    t=t or now_utc();total=0.
    for j in jobs:
        start=dt.datetime.fromisoformat(j["created_at_utc"].replace("Z","+00:00"))
        end=dt.datetime.fromisoformat(j["terminal_at_utc"].replace("Z","+00:00")) if j.get("terminal_at_utc") else t
        total+=max(0,(end-start).total_seconds())*int(j["gpus"])/3600
    return total
def decision(jobs,t=None,extra_gpu_hours=0.5):
    active=[j for j in jobs if j.get("status") not in TERMINAL]
    total_upper=conservative_hours(jobs,t)+extra_gpu_hours
    reason="BLACKOUT" if blackout(t) else "GPU_BUDGET" if total_upper>=19.8 else None
    if len(active)>2 or any(j["gpus"]>2 for j in active):reason="RESOURCE_CAP"
    return {"stop_job_ids":[j["job_id"] for j in active] if reason else [],"reason":reason,
            "resume_allowed":not blackout(t) and total_upper<19.8,
            "conservative_gpu_hours":total_upper,"dev14_gpu_hours_reserved_upper_bound":extra_gpu_hours}
def atomic(path,obj):
    path=Path(path);p=path.with_name(path.name+".tmp");p.write_text(json.dumps(obj,indent=2)+"\n");os.replace(p,path)
def read_json_retry(path,attempts=READ_ATTEMPTS,delay=READ_DELAY):
    """Reopen CPFS control JSON after bounded inode replacement failures."""
    path=Path(path)
    if attempts<1:raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:return json.loads(path.read_text())
        except OSError as exc:
            if exc.errno not in TRANSIENT_READ_ERRNOS or attempt==attempts-1:raise
        except json.JSONDecodeError:
            if attempt==attempts-1:raise
        time.sleep(delay)
    raise RuntimeError("unreachable read retry")
def cli(args):
    if Path.cwd().resolve()!=REGISTRY:raise RuntimeError("canonical registry cwd required")
    conf=json.loads((REGISTRY/"config/toolchain.json").read_text());pin=conf["dlc"]
    binary=REGISTRY/pin["file"]
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=pin["sha256"]:raise RuntimeError("DLC pin drift")
    command=[str(binary),*args,"--config","/workspace/leon/.dlc/config"]
    return subprocess.run(command,capture_output=True,text=True,timeout=45,check=True).stdout
def get_job(jid):
    if not re.fullmatch(r"dlc[a-z0-9]+",jid):raise ValueError(jid)
    raw=cli(["get","job",jid,"--show_detail"]);data=json.loads(raw[raw.index("{"):])
    # All environment values and payloads remain in memory; persist only allowlisted readback.
    keys=("JobId","Status","ResourceId","OversoldType","UseOversoldResource",
          "GmtCreateTime","GmtRunningTime","GmtFinishTime","Duration")
    result={k:data.get(k) for k in keys}
    conf=json.loads((REGISTRY/"config/toolchain.json").read_text());pin=conf["openapi_idle_source"]
    helper=REGISTRY/pin["file"]
    if hashlib.sha256(helper.read_bytes()).hexdigest()!=pin["sha256"]:raise RuntimeError("OpenAPI helper pin drift")
    api=runpy.run_path(str(helper),run_name="s1_readonly_placement")
    credentials=api["checked_config"](Path("/workspace/leon/.dlc/config"),conf)
    query=api["canonical_query"]({"JobIds":jid,"PageNumber":"1","PageSize":"10","WorkspaceId":str(conf["workspace_id"]),"ResourceId":"quota1ssrabud0bh"})
    try:
        response=api["fetch_list_jobs"](endpoint=conf["endpoint"],query_string=query,access_id=credentials["access_id"],access_key=credentials["access_key"],security_token=credentials["security_token"])
    finally:credentials.clear()
    jobs=response.get("Jobs",[])
    if len(jobs)!=1 or jobs[0].get("JobId")!=jid:raise RuntimeError("exact ListJobs binding failed")
    observed=jobs[0]
    result["UseOversoldResource"]=observed.get("UseOversoldResource")
    result["OversoldType"]=observed.get("OversoldType") or observed.get("Settings",{}).get("OversoldType")
    result["placement_source"]="exact_openapi_ListJobs"
    return result
def _append_error(base,record):
    try:
        with (Path(base)/"control_errors.jsonl").open("a") as handle:
            handle.write(json.dumps(record,sort_keys=True)+"\n")
    except OSError:
        # Error reporting cannot turn a fail-closed controller into an
        # unhandled exception while the same CPFS mount is unhealthy.
        return False
    return True

def _fail_closed(base,known_state,exc,stop_attempted):
    """Publish a deny heartbeat and stop last-known active jobs on I/O loss."""
    base=Path(base)
    try:base.mkdir(parents=True,exist_ok=True)
    except OSError:
        # Continue the fail-closed stop path even if this same mount cannot
        # create its log directory; CLI StopJob is the safer last-known-job
        # action and heartbeat writes below will be attempted independently.
        pass
    raw_jobs=list((known_state or {}).get("jobs",[])) if isinstance(known_state,dict) else []
    jobs=[job for job in raw_jobs if isinstance(job,dict)]
    active=[job for job in jobs if job.get("status") not in TERMINAL and job.get("job_id")]
    stop_ids=[str(job["job_id"]) for job in active]
    attempted=set(stop_attempted or ())
    checked=now_utc()
    reason="CONTROL_IO_FAILURE"
    try:
        d=decision(jobs,checked,extra_gpu_hours=float((known_state or {}).get("dev14_gpu_hours_upper_bound",0.5)))
        reason=d.get("reason") or reason
    except Exception:
        d={"stop_job_ids":stop_ids,"reason":reason}
    try:(base/"STOP").write_text(reason+"\n")
    except Exception as marker_exc:
        _append_error(base,{"time":checked.isoformat(),"error":"STOP_WRITE_FAILED","error_type":type(marker_exc).__name__})
    for jid in stop_ids:
        if jid in attempted:continue
        try:
            cli(["stop","job",jid,"--force","--quiet"])
            with (base/"control_actions.jsonl").open("a") as handle:
                handle.write(json.dumps({"time":checked.isoformat(),"action":"StopJob","job_id":jid,"reason":reason},sort_keys=True)+"\n")
            attempted.add(jid)
        except Exception as stop_exc:
            _append_error(base,{"time":checked.isoformat(),"job_id":jid,"error":type(stop_exc).__name__,"reason":"FAIL_CLOSED_STOP_FAILED"})
    payload={
        "time":checked.isoformat(),"pid":os.getpid(),"resume_allowed":False,
        "reason":reason,"controller_io_error":type(exc).__name__,
        "controller_io_message":str(exc)[:300],"stale":True,
        "stop_job_ids":stop_ids,
        "jobs":[{"job_id":job.get("job_id"),"status":job.get("status")} for job in jobs if job.get("job_id")],
    }
    try:atomic(base/"heartbeat.json",payload)
    except Exception as heartbeat_exc:
        _append_error(base,{"time":checked.isoformat(),"error":"HEARTBEAT_WRITE_FAILED","error_type":type(heartbeat_exc).__name__})
    return attempted

def _cycle(manifest,base):
    """Run one normal cycle; all control JSON reads are bounded."""
    state=read_json_retry(manifest)
    if not isinstance(state,dict) or not isinstance(state.get("jobs"),list):
        raise ValueError("control manifest must contain a jobs list")
    jobs=state["jobs"]
    for job in jobs:
        if job.get("status") in TERMINAL:continue
        try:readback=get_job(job["job_id"])
        except Exception as exc:
            job["control_readback_error"]=type(exc).__name__
            job["readback_error_streak"]=job.get("readback_error_streak",0)+1
            if job["readback_error_streak"]>=3:job["placement_rejected"]=True
            continue
        job["status"]=readback["Status"];job["readback"]=readback;job["readback_error_streak"]=0
        if job["status"] in TERMINAL:job["terminal_at_utc"]=readback.get("GmtFinishTime") or now_utc().isoformat()
        fatal=Path(job.get("artifact_dir","/nonexistent"))/"pai_state/FATAL_ERROR.json"
        if fatal.is_file() and job["status"] not in TERMINAL:
            job["placement_rejected"]=True;job["application_fatal_error"]=True
        if job["status"]=="Running" and readback.get("UseOversoldResource") is not True:
            job["placement_rejected"]=True
    d=decision(jobs,extra_gpu_hours=float(state.get("dev14_gpu_hours_upper_bound",0.5)))
    active_rejected=[j for j in jobs if j.get("placement_rejected") and j.get("status") not in TERMINAL]
    d["stop_job_ids"]+= [j["job_id"] for j in active_rejected]
    if active_rejected:d["resume_allowed"]=False
    if d["stop_job_ids"]:
        (base/"STOP").write_text(d["reason"] or "PLACEMENT_REJECTED")
        for jid in set(d["stop_job_ids"]):
            for job in jobs:
                if job["job_id"]==jid:
                    job["stop_reason"]=d["reason"] or ("APPLICATION_ERROR" if job.get("application_fatal_error") else "PLACEMENT_REJECTED")
                    job["stop_requested_utc"]=now_utc().isoformat()
            try:cli(["stop","job",jid,"--force","--quiet"])
            except Exception as exc:
                _append_error(base,{"time":now_utc().isoformat(),"job_id":jid,"error":type(exc).__name__})
                continue
            with (base/"control_actions.jsonl").open("a") as handle:
                handle.write(json.dumps({"time":now_utc().isoformat(),"action":"StopJob","job_id":jid,"reason":d["reason"] or "PLACEMENT_REJECTED"},sort_keys=True)+"\n")
    elif d["resume_allowed"] and not active_rejected:
        (base/"STOP").unlink(missing_ok=True)
    import fcntl
    with manifest.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        latest=read_json_retry(manifest)
        if not isinstance(latest,dict) or not isinstance(latest.get("jobs"),list):raise ValueError("control manifest must contain a jobs list")
        updates={j["job_id"]:j for j in jobs}
        for j in latest["jobs"]:
            if j["job_id"] in updates:
                update=updates[j["job_id"]]
                for key in ("status","readback","terminal_at_utc","placement_rejected","application_fatal_error","stop_reason","stop_requested_utc","control_readback_error","readback_error_streak"):
                    if key in update:j[key]=update[key]
        state=latest;atomic(manifest,state)
    atomic(base/"heartbeat.json",{"time":now_utc().isoformat(),"pid":os.getpid(),**d,"jobs":[{"job_id":j["job_id"],"status":j.get("status")} for j in jobs]})
    return state

def run(manifest,interval=15,once=False):
    manifest=Path(manifest).resolve();base=manifest.parent;os.chdir(REGISTRY)
    known_state=None;stop_attempted=set()
    while True:
        try:
            known_state=_cycle(manifest,base)
            stop_attempted.clear()
        except Exception as exc:
            stop_attempted=_fail_closed(base,known_state,exc,stop_attempted)
        # New submissions are performed by the parent only after checking this receipt.
        if once:return
        time.sleep(interval)
def self_test():
    def at(s):return dt.datetime.fromisoformat(s).replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    cases=[("09:24",False),("09:25",True),("09:30",True),("09:39",True),("09:40",False),("19:25",True),("19:40",False)]
    for hhmm,v in cases:assert blackout(at("2026-09-07T"+hhmm)) is v
    j={"job_id":"dlctest1","gpus":2,"status":"Running","created_at_utc":"2026-09-06T00:00:00+00:00"}
    assert decision([j],dt.datetime(2026,9,6,10,tzinfo=dt.timezone.utc))["reason"]=="GPU_BUDGET"
    j["terminal_at_utc"]="2026-09-06T01:00:00+00:00";j["status"]="Stopped"
    assert conservative_hours([j])==2.
    return {"passed":True,"blackout_boundaries":len(cases),"budget_accounting":"creation_to_terminal_upper_bound","scheduling":"stop admission five minutes early"}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--manifest");p.add_argument("--self-test",action="store_true");p.add_argument("--once",action="store_true")
    a=p.parse_args()
    if a.self_test:print(json.dumps(self_test()))
    else:run(a.manifest,once=a.once)
