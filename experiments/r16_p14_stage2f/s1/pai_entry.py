"""PAI entrypoint for committed immutable Stage-2F collection/evaluation shards."""
from __future__ import annotations
import argparse,json,os,time,subprocess
from pathlib import Path
from .common import ROOT,TASKS,atomic_json,file_sha,guard_execution
def wait_controller(control_root):
    p=Path(control_root);run_id=os.environ["PAI_CANARY_RUN_ID"]
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        guard_execution()
        hp=p/"heartbeat.json";mp=p/"jobs.json"
        if hp.exists() and mp.exists():
            h=json.loads(hp.read_text());m=json.loads(mp.read_text())
            from datetime import datetime,timezone
            age=(datetime.now(timezone.utc)-datetime.fromisoformat(h["time"])).total_seconds()
            for j in m["jobs"]:
                if j["run_id"]==run_id and age<90 and j.get("readback",{}).get("UseOversoldResource") is True:
                    if j.get("status")!="Running":continue
                    if j.get("readback",{}).get("ResourceId")!="quota1ssrabud0bh":raise RuntimeError("wrong robot quota")
                    return j
        time.sleep(5)
    raise RuntimeError("no fresh external controller/actual idle receipt")
def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=("collect","grid","atlas"),required=True)
    p.add_argument("--task",choices=TASKS,required=True);p.add_argument("--output-root",required=True)
    p.add_argument("--control-root",required=True);p.add_argument("--source-commit",required=True);p.add_argument("--source-manifest-sha256",required=True)
    p.add_argument("--workers",type=int,default=12);args=p.parse_args()
    if (os.getuid(),os.getgid())!=(2254,2254):raise RuntimeError("requires2254:2254")
    os.environ["S1_STOP_FILE"]=str(Path(args.control_root)/"STOP")
    os.environ["S1_CONTROL_HEARTBEAT"]=str(Path(args.control_root)/"heartbeat.json")
    manifest_path=ROOT/"SOURCE_MANIFEST.json"
    if file_sha(manifest_path)!=args.source_manifest_sha256:raise RuntimeError("payload manifest binding drift")
    for rel,expected in json.loads(manifest_path.read_text()).items():
        if file_sha(ROOT/rel)!=expected:raise RuntimeError(f"payload code/asset drift: {rel}")
    job=wait_controller(args.control_root)
    os.environ["S1_SOURCE_COMMIT"]=args.source_commit;os.environ["S1_JOB_ID"]=job["job_id"]
    from .assets import configure_assets
    assets=configure_assets()
    import torch,wandb
    if torch.cuda.device_count()!=2:raise RuntimeError("exactly two visible A800 GPUs required")
    if not all("A800" in torch.cuda.get_device_name(i) for i in range(2)):raise RuntimeError("A800 required")
    task_index=TASKS.index(args.task)
    w=wandb.init(entity="chen_jian-cj-workspace",project="r16-p14-stage2f-s1",
       id=os.environ["PAI_CANARY_RUN_ID"],resume="allow",
       config=dict(source_commit=args.source_commit,phase=args.phase,task=args.task,
                   actor_seeds=[7,17,29],zero_injection=True,training=False,policy_call_cap=8,
                   job_id=job["job_id"],gpu_count=2))
    try:
        state=Path(os.environ["PAI_CANARY_RUN_DIR"])/"pai_state";state.mkdir(exist_ok=True)
        atomic_json(state/"RUNTIME.json",dict(uid=os.getuid(),gid=os.getgid(),pid=os.getpid(),
            source_commit=args.source_commit,job_id=job["job_id"],phase=args.phase,assets=assets,
            python=os.sys.executable,cuda=torch.version.cuda,torch=torch.__version__,
            devices=[torch.cuda.get_device_name(i) for i in range(2)]),immutable=False)
        if args.phase=="collect":
            from .collector import run_task
            result=run_task(args.task,Path(args.output_root)/"phase0b",device="cuda",workers=args.workers)
        else:
            from .matrix import run_task
            result=run_task(args.phase,args.task,args.output_root,workers=args.workers,device="cuda")
        atomic_json(state/"COMPLETED.json",dict(uid=os.getuid(),gid=os.getgid(),source_commit=args.source_commit,
                    phase=args.phase,task=args.task,result=result),immutable=False)
        w.log({"completed":1});w.finish()
    except BaseException:
        w.finish(exit_code=1);raise
if __name__=="__main__":
    try:main()
    except BaseException as exc:
        import traceback
        state_dir=os.environ.get("PAI_CANARY_RUN_DIR")
        if state_dir:
            paused="BLACKOUT" in str(exc) or "GPU_BUDGET" in str(exc)
            atomic_json(Path(state_dir)/"pai_state"/("PAUSED.json" if paused else "FATAL_ERROR.json"),
                {"error_type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc(),"uid":os.getuid()},immutable=False)
        raise
