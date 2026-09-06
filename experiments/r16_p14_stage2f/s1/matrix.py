"""S1 exact configured grid and atlas with sealed evaluation admission."""
from __future__ import annotations
import itertools,json,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from .common import SEEDS,OPERATORS,TASKS,atomic_json,digest,file_sha,guard_execution
BUDGETS=tuple(itertools.product((4,8,16),(8,16,32)))
GRID_K=(2,4,8,12,16)
ATLAS_K=(2,4,6,8,10,12,14,16)
REFERENCES=(("immediate_fresh",2),("fixed_delay_2",4),("fixed_delay_4",6),("fixed_delay_8",10))
def load_events(root,task,split):
    if split=="evaluation":selected_budget(root)
    folder="sealed_evaluation" if split=="evaluation" else "episodes"
    events=[]
    for p in sorted((Path(root)/"phase0b"/folder/task).glob("*.json")):
        row=json.loads(p.read_text())
        if row.get("task")!=task or row.get("status")!="COMPLETE":raise RuntimeError("invalid event shard binding")
        if row["split"]==split and row["qualified_natural_failure"]:
            if row.get("event") is None:raise RuntimeError("qualified row lacks event")
            event=row["event"]
            if event.get("task")!=task or event.get("split")!=split or event.get("init_state_id")!=row["init_state_id"]:raise RuntimeError("event/shard identity mismatch")
            event["episode_shard_sha256"]=file_sha(p);events.append(event)
    return sorted(events,key=lambda e:(int(e["init_state_id"]),int(e["actor_seed"]),e["event_instance_id"]))
def selected_budget(root):
    root=Path(root);receipt=root/"phase1/selection_receipt.json"
    authorization=root/"phase1/selection_authorization.json"
    if not receipt.is_file() or not authorization.is_file():raise RuntimeError("evaluation OPEN_DENY: selection not committed")
    auth=json.loads(authorization.read_text())
    from .selection import verify_commit_proof
    verify_commit_proof(receipt,auth)
    data=json.loads(receipt.read_text())
    if data.get("status")!="SELECTED" or data.get("selection_source")!="calibration_only":raise RuntimeError("no valid calibration selection")
    if data.get("planned_sample_complete") is False or data.get("sample_complete") is False:
        raise RuntimeError("evaluation OPEN_DENY: planned calibration sample incomplete")
    budget=data.get("selected_budget") or {}
    if budget.get("tail_horizon") not in (4,8,16) or budget.get("action_budget") not in (8,16,32) or budget.get("policy_call_cap")!=8:raise RuntimeError("budget outside preregistered grid")
    candidates=data.get("ranked_candidates",[])
    if not candidates or candidates[0].get("budget")!=budget or candidates[0].get("qualifies") is not True:raise RuntimeError("selection rank binding invalid")
    for field,check in (("oracle_by_task",lambda x:.25<=x<=.85),("gap_by_task",lambda x:x>=.15)):
        values=candidates[0].get(field,{})
        if set(values)!=set(TASKS) or not all(check(float(x)) for x in values.values()):raise RuntimeError("K2 two-task qualification missing")
    return data["selected_budget"]
def compatible_runtime_row(row,module_hashes):
    previous=row.get("runtime_module_hashes")
    if previous==module_hashes:
        return True
    receipt=json.loads((Path(__file__).parent/"dispatch_compatibility.json").read_text())
    if row.get("source_commit") not in receipt["allowed_existing_source_commits"]:
        return False
    if module_hashes.get("dispatch.py")!=receipt["new_dispatch_sha256"]:
        return False
    if not isinstance(previous,dict) or set(previous)!=set(module_hashes):
        return False
    for name,expected in receipt["other_module_sha256"].items():
        if previous.get(name)!=expected or module_hashes.get(name)!=expected:
            return False
    return previous.get("dispatch.py")==receipt["old_dispatch_sha256"]

def record_first_work(row_path):
    state_dir=os.environ.get("PAI_CANARY_RUN_DIR")
    if not state_dir:return
    import fcntl
    first=Path(state_dir)/"pai_state/FIRST_REAL_WORK.json";first.parent.mkdir(parents=True,exist_ok=True)
    with first.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not first.exists():atomic_json(first,{"uid":os.getuid(),"gid":os.getgid(),"shard":str(row_path),"sha256":file_sha(row_path)})
def run_task(phase,task,output_root,workers=12,device="cuda"):
    from .dispatch import run_spawned_branch
    root=Path(output_root);phase_dir=root/("phase1" if phase=="grid" else "phase2")
    if phase=="grid":
        events=load_events(root,task,"calibration")[:20];budgets=BUDGETS;prefixes=GRID_K
        # Execute every available independent request without fabricating missing events.
        # Incomplete planned sampling can never authorize evaluation selection.
    elif phase=="atlas":
        b=selected_budget(root)
        budgets=((int(b["tail_horizon"]),int(b["action_budget"])),);prefixes=ATLAS_K
        events=load_events(root,task,"calibration")+load_events(root,task,"evaluation")
    else:raise ValueError(phase)
    module_hashes={name:file_sha(Path(__file__).parent/name) for name in ("runtime.py","measurement.py","dispatch.py","common.py","assets.py","runtime_identity.py")}
    requests=[]
    for event,(tail,budget),seed in itertools.product(events,budgets,SEEDS):
        for op,k in itertools.product(OPERATORS,prefixes):
            requests.append((event,dict(recovery_actor_seed=seed,operator=op,prefix_k=k,
                     tail_horizon=tail,action_budget=budget,policy_call_cap=8),False))
        for op,k in REFERENCES:
            requests.append((event,dict(recovery_actor_seed=seed,operator=op,prefix_k=k,
                     tail_horizon=tail,action_budget=budget,policy_call_cap=8),True))
    import threading
    cancelled=threading.Event()
    def run(item):
        if cancelled.is_set():return
        index,(event,req,reference)=item;guard_execution()
        key=digest({"event":event["event_instance_id"],"event_hash":digest(event),**req,"reference":reference})
        path=phase_dir/"shards"/task/f"{key}.json"
        if path.exists():
            row=json.loads(path.read_text())
            if not compatible_runtime_row(row,module_hashes) or row.get("runtime_receipt_sha256")!=os.environ.get("S1_RUNTIME_RECEIPT_SHA256"):
                raise RuntimeError(f"existing shard runtime binding changed: {path}")
            if row.get("status")!="COMPLETE" and row.get("error_type")!="PrefixOutsideTaskHorizon":
                raise RuntimeError(f"immutable failed shard {path}")
            return
        # Phase1 calibration branches are identical interventions at shared k/budget.
        prior=root/"phase1/shards"/task/f"{key}.json"
        if phase=="atlas" and event["split"]=="calibration" and prior.exists():
            row=json.loads(prior.read_text())
            if row.get("status")!="COMPLETE" and row.get("error_type")!="PrefixOutsideTaskHorizon":
                raise RuntimeError("cannot reuse failed calibration")
            if not compatible_runtime_row(row,module_hashes) or row.get("runtime_receipt_sha256")!=os.environ.get("S1_RUNTIME_RECEIPT_SHA256"):raise RuntimeError("calibration runtime changed before atlas")
            row={**row,"reused_calibration_shard":str(prior),"reused_calibration_sha256":file_sha(prior)}
        else:
            trace=phase_dir/"contact_topology"/task/f"{key}.jsonl.gz"
            row=run_spawned_branch(event=event,**req,device=(f"cuda:{index%2}" if device=="cuda" else device),trace_path=str(trace),cancel_event=cancelled)
            row["trace_path"]=str(trace)
            row["runtime_status"]=row.get("status")
            if row.get("status")=="OK" and not row.get("blocked") and not row.get("error"):
                row["status"]="COMPLETE"
            if row.get("prefix_k")!=req["prefix_k"] or row.get("operator")!=req["operator"]:
                row["status"]="BLOCKED";row["request_mismatch"]=True
            row["returned_configuration"]={key:row.get(key) for key in req}
            row.update(**req)
            row.update(source_commit=os.environ.get("S1_SOURCE_COMMIT"),runtime_module_hashes=module_hashes,runtime_receipt_sha256=os.environ.get("S1_RUNTIME_RECEIPT_SHA256"),
                 pai_run_id=os.environ.get("PAI_CANARY_RUN_ID"),job_id=os.environ.get("S1_JOB_ID"),task=task,event_instance_id=event["event_instance_id"],init_state_id=event["init_state_id"],
                 generator_actor_seed=event["actor_seed"],split=event["split"],is_reference=reference,
                 configured_budget={"tail_horizon":req["tail_horizon"],"action_budget":req["action_budget"],"policy_call_cap":8})
        if any(not row.get(k) for k in ("pid","env_hash","chunk_hash")):
            row["status"]="BLOCKED";row["provenance_missing"]=True
        atomic_json(path,row)
        if row.get("error_type")=="PrefixOutsideTaskHorizon":
            print(f"preserved infeasible prefix; continuing independent requests: {path}",flush=True)
            return
        if row.get("status")!="COMPLETE":raise RuntimeError(f"branch blocked: {path}")
        record_first_work(path)
        print(f"persisted {phase} task={task} index={index+1}/{len(requests)}",flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run,item) for item in enumerate(requests)]
        try:
            for f in as_completed(futures):f.result()
        except BaseException:
            cancelled.set()
            for f in futures:f.cancel()
            marker=os.environ.get("S1_STOP_FILE")
            if marker:Path(marker).write_text("S1_MATRIX_FAILURE")
            raise
    paths=sorted((phase_dir/"shards"/task).glob("*.json"))
    blocked=sum(json.loads(p.read_text()).get("status")!="COMPLETE" for p in paths)
    result={"task":task,"phase":phase,"events":len(events),"requested_rows":len(requests),"persisted_rows":len(paths),
            "core_rows":sum(not json.loads(p.read_text()).get("is_reference",False) for p in paths),"blocked_contract_rows":blocked,
            "status":"COMPLETE_WITH_BLOCKED_CONTRACT_ROWS" if blocked else "COMPLETE"}
    result.update(planned_events=20 if phase=="grid" else len(events),
                  planned_sample_complete=phase!="grid" or len(events)==20,
                  missing_planned_events=max(0,20-len(events)) if phase=="grid" else 0)
    if not result["planned_sample_complete"]:
        result["status"]="COMPLETE_AVAILABLE_REQUESTS_SAMPLE_SHORTFALL"
    if len(paths)!=len(requests):raise RuntimeError("shard count mismatch")
    atomic_json(phase_dir/f"completion_{task}.json",result)
    return result
