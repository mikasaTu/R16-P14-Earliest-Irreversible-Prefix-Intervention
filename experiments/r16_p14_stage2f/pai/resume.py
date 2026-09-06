"""Resume only this lineage's blackout-stopped jobs after the guarded window."""
from __future__ import annotations
import argparse,datetime as dt,fcntl,json,os,re,subprocess,time
from pathlib import Path
REG=Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/pai-job-registry")
TERMINAL={"Succeeded","Failed","Stopped","Deleted"}
def write(path,value):
    tmp=path.with_name(path.name+".resume.tmp");tmp.write_text(json.dumps(value,indent=2)+"\n");os.replace(tmp,path)
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def eligible(state,heartbeat):
    age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(heartbeat["time"])).total_seconds()
    if age>60 or not heartbeat.get("resume_allowed"):return []
    if sum(j.get("status") not in TERMINAL for j in state["jobs"])>=2:return []
    return [j for j in state["jobs"] if j.get("status")=="Stopped" and j.get("stop_reason")=="BLACKOUT" and not j.get("resume_claimed") and not j.get("resumed_by")]
def run(manifest):
    os.chdir(REG);manifest=Path(manifest);base=manifest.parent
    while True:
        try:
            if not (base/"heartbeat.json").exists():time.sleep(5);continue
            with manifest.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);state=json.loads(manifest.read_text());h=json.loads((base/"heartbeat.json").read_text())
                candidates=eligible(state,h)
                if not candidates:time.sleep(1);continue
                source=candidates[0];source["resume_claimed"]=now()
                source["resume_sequence"]=int(source.get("resume_sequence",0))+1
                root_id=source.get("root_run_id",source["run_id"])
                new_id=f"{root_id}-b{source['resume_sequence']}"
                if not re.fullmatch("[A-Za-z0-9][A-Za-z0-9._-]{2,63}",new_id):raise RuntimeError("invalid generated resume id")
                source["resume_target_run_id"]=new_id;write(manifest,state)
            cli=REG/"bin/pai-job";target=REG/"runs"/new_id
            if target.exists():raise RuntimeError("resume target already exists; reconcile before any retry")
            subprocess.run([str(cli),"clone","--from-run",source["run_id"],"--run-id",new_id],capture_output=True,text=True,check=True,timeout=300)
            status=subprocess.run([str(cli),"submit-resolved","--run-id",new_id,"--config","/workspace/leon/.dlc/config"],capture_output=True,text=True,timeout=1200)
            result_path=target/"result.json"
            if not result_path.exists():raise RuntimeError(f"submission requires reconciliation rc={status.returncode}")
            result=json.loads(result_path.read_text());jid=result.get("job_id")
            if not re.fullmatch("dlc[a-z0-9]{8,}",jid or ""):raise RuntimeError("invalid resumed JobId receipt")
            resolved=json.loads((target/"resolved.json").read_text())
            entry={key:source[key] for key in ("phase","task","gpus","source_commit","source_tree") if key in source}
            entry.update(run_id=new_id,root_run_id=root_id,resume_sequence=source["resume_sequence"],job_id=jid,status="Created",
              created_at_utc=now(),artifact_dir=str(Path(source["artifact_dir"]).parent/new_id),
              resumed_from_job_id=source["job_id"],resumed_from_run_id=source["run_id"],resume_reason="BLACKOUT",
              persisted_completion_verified=False)
            with manifest.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);state=json.loads(manifest.read_text())
                for item in state["jobs"]:
                    if item["job_id"]==source["job_id"]:item["resumed_by"]=jid
                if any(j["job_id"]==jid for j in state["jobs"]):raise RuntimeError("duplicate resumed job receipt")
                state["jobs"].append(entry);write(manifest,state)
            with (base/"resume_actions.jsonl").open("a") as f:f.write(json.dumps({"time":now(),"job_id":jid,"run_id":new_id,"from_job_id":source["job_id"]})+"\n")
        except BaseException as exc:
            with (base/"resume_errors.jsonl").open("a") as f:f.write(json.dumps({"time":now(),"error_type":type(exc).__name__,"message":str(exc)[:500]})+"\n")
        time.sleep(15)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);run(p.parse_args().manifest)
