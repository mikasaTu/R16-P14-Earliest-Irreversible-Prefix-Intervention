"""External controller policy; state-changing CLI runs only in canonical registry."""
from __future__ import annotations
import runpy
import argparse,datetime as dt,hashlib,json,os,re,subprocess,time
from pathlib import Path
from zoneinfo import ZoneInfo
REGISTRY=Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/pai-job-registry")
ACTIVE={"Created","Preparing","Queuing","Running","Restarting","Pending","Starting","Stopping"}
TERMINAL={"Succeeded","Failed","Stopped","Deleted"}
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
def decision(jobs,t=None):
    active=[j for j in jobs if j.get("status") not in TERMINAL]
    reason="BLACKOUT" if blackout(t) else "GPU_BUDGET" if conservative_hours(jobs,t)>=19.8 else None
    if len(active)>2 or any(j["gpus"]>2 for j in active):reason="RESOURCE_CAP"
    return {"stop_job_ids":[j["job_id"] for j in active] if reason else [],"reason":reason,
            "resume_allowed":not blackout(t) and conservative_hours(jobs,t)<19.8,
            "conservative_gpu_hours":conservative_hours(jobs,t)}
def atomic(path,obj):
    path=Path(path);p=path.with_name(path.name+".tmp");p.write_text(json.dumps(obj,indent=2)+"\n");os.replace(p,path)
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
def run(manifest,interval=15,once=False):
    manifest=Path(manifest);base=manifest.parent;os.chdir(REGISTRY)
    while True:
        state=json.loads(manifest.read_text());jobs=state["jobs"]
        for job in jobs:
            if job.get("status") in TERMINAL:continue
            try:readback=get_job(job["job_id"])
            except Exception as exc:
                job["control_readback_error"]=type(exc).__name__;job["placement_rejected"]=True
                continue
            job["status"]=readback["Status"];job["readback"]=readback
            if job["status"] in TERMINAL:job["terminal_at_utc"]=readback.get("GmtFinishTime") or now_utc().isoformat()
            if job["status"]=="Running" and readback.get("UseOversoldResource") is not True:
                job["placement_rejected"]=True
        d=decision(jobs)
        d["stop_job_ids"]+= [j["job_id"] for j in jobs if j.get("placement_rejected") and j.get("status") not in TERMINAL]
        if d["stop_job_ids"]:
            (base/"STOP").write_text(d["reason"] or "PLACEMENT_REJECTED")
            for jid in set(d["stop_job_ids"]):
                try:cli(["stop","job",jid,"--force","--quiet"])
                except Exception as exc:
                    with (base/"control_errors.jsonl").open("a") as f:f.write(json.dumps({"time":now_utc().isoformat(),"job_id":jid,"error":type(exc).__name__})+"\n")
                    continue
                with (base/"control_actions.jsonl").open("a") as f:
                    f.write(json.dumps({"time":now_utc().isoformat(),"action":"StopJob","job_id":jid,"reason":d["reason"] or "PLACEMENT_REJECTED"})+"\n")
        elif d["resume_allowed"] and not any(j.get("placement_rejected") for j in jobs):
            (base/"STOP").unlink(missing_ok=True)
        atomic(manifest,state)
        atomic(base/"heartbeat.json",{"time":now_utc().isoformat(),"pid":os.getpid(),**d,"jobs":[{"job_id":j["job_id"],"status":j.get("status")} for j in jobs]})
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
