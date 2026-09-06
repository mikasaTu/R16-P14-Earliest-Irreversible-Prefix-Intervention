"""Outcome-blind reset pool and full clean episodes; no recovery outcomes."""
from __future__ import annotations
from .common import ROOT,TASKS,SEEDS,atomic_json,digest,file_sha,split_for,spawn_call,guard_execution
import argparse, json, os, time
from pathlib import Path

def make_init(task,init_state_id):
    import numpy as np
    from r16_p14_stage2a.envs import make_env,state_sha256
    seed=2016214+10000*TASKS.index(task)+int(init_state_id)
    env,_=make_env(task,seed=seed)
    try:
        state=np.asarray(env.get_sim_state(),dtype=np.float64)
        if not np.isfinite(state).all() or env.check_success():
            raise RuntimeError("invalid preregistered reset; resampling forbidden")
        return dict(task=task,init_state_id=int(init_state_id),split=split_for(init_state_id),
                    reset_seed=seed,state=state.tolist(),state_hash=state_sha256(state))
    finally:env.close()

def collect_episode(task,init_state_id,actor_seed,pool_path,output_root,device="cpu"):
    import numpy as np
    import torch
    from r16_p14_stage2a.envs import make_env,restore_state,joint_qpos,state_sha256,contact_pairs
    from r16_p14_stage2a.settings import TASK_SPECS
    from r16_p14_stage2b.runtime import ActorBundle,ActorHistory,chunk_hash
    from r16_p14_stage2d.io_utils import sha256_array
    from .measurement import SimulationStepRecorder,label_trace
    started=time.monotonic()
    torch.set_num_threads(1);np.random.seed(0);torch.manual_seed(0)
    pool_path=Path(pool_path);pool=json.loads(pool_path.read_text())
    if pool["task"]!=task:raise RuntimeError("pool task mismatch")
    init=pool["states"][int(init_state_id)]
    if init["init_state_id"]!=init_state_id:raise RuntimeError("pool id mismatch")
    spec=TASK_SPECS[task];eid=f"{task}__init{init_state_id:03d}__actor{actor_seed}"
    env,_=make_env(task,seed=0)
    output_root=Path(output_root)
    trace_path=output_root/"contact_topology"/task/f"{eid}.jsonl.gz"
    try:
        obs=restore_state(env,np.asarray(init["state"],dtype=np.float64))
        if state_sha256(env.get_sim_state())!=init["state_hash"]:raise RuntimeError("init restore hash mismatch")
        history=ActorHistory.initial(obs);bundle=ActorBundle.load(actor_seed,device)
        initial=joint_qpos(env,spec.manipulated_joint)
        event=None;stable=0;actions=[];previous=0.;ever=False;success=False
        recorder=SimulationStepRecorder(env,task,event_instance_id=eid,prefix_k=0,
                    operator="clean_actor",actor_seed=actor_seed,initial_object_z=float(initial[2]),
                    trace_path=str(trace_path))
        with recorder:
            for step in range(spec.horizon):
                guard_execution()
                qpos=joint_qpos(env,spec.manipulated_joint)
                lifted=float(qpos[2]-initial[2])>=float(spec.lift_delta)
                stable=stable+1 if lifted else 0;ever|=lifted
                chunk=np.ascontiguousarray(bundle.predict(history.state_array(),history.action_array(),task),dtype=np.float32)
                if chunk.shape!=(16,7) or not np.isfinite(chunk).all():raise RuntimeError("invalid actor chunk")
                grasp=bool(env.env._check_grasp(env.env.robots[0].gripper,env.env.objects_dict[spec.manipulated_joint.removesuffix("_joint0")]))
                if event is None and grasp and stable>=2 and previous>0 and not env.check_success():
                    state=np.asarray(env.get_sim_state(),dtype=np.float64)
                    pre=np.asarray(actions,dtype=np.float32).reshape(-1,7)
                    states=history.state_array();ah=history.action_array()
                    event=dict(schema_version=1,event_id=eid,event_instance_id=eid,task=task,
                      split=init["split"],actor_seed=actor_seed,generator_actor_seed=actor_seed,
                      checkpoint=str(bundle.checkpoint.relative_to(ROOT)),checkpoint_sha256=bundle.checkpoint_sha256,
                      actor_run_id=f"stage2a_shared_multitask_seed_{actor_seed}",init_state_id=init_state_id,
                      init_pool_file_sha256=file_sha(pool_path),init_state=init["state"],init_state_hash=sha256_array(init["state"],np.float64),pool_state_raw_hash=init["state_hash"],
                      pre_anchor_actions=pre.tolist(),pre_anchor_actions_hash=sha256_array(pre,np.float32),
                      anchor_global_step=step,anchor_state=state.tolist(),anchor_state_hash=state_sha256(state),
                      state_history=states.tolist(),state_history_hash=sha256_array(states,np.float32),
                      action_history=ah.tolist(),action_history_hash=sha256_array(ah,np.float32),
                      original_chunk=chunk.tolist(),original_chunk_hash=chunk_hash(chunk),chunk_length=16,
                      initial_manipulated_qpos=initial.tolist(),anchor_manipulated_qpos=qpos.tolist(),
                      anchor_target_qpos=joint_qpos(env,spec.target_joint).tolist(),
                      anchor_obstacle_qpos=None,anchor_contacts=[list(x) for x in contact_pairs(env)],
                      task_phase=dict(stable_lift_two_steps=True,stable_lift_count=stable,grasp=True,
                                      gripper_closed=True,task_success=False,ever_stably_lifted=True),
                      source_is_actor_generated_chunk=True,source_is_demonstration_chunk=False,
                      global_step_fallback_used=False)
                if event is not None:recorder.baseline_contacts=event["anchor_contacts"]
                action=chunk[0].copy()
                recorder.set_action(step=step,action=action,previous_gripper=previous,ever_lifted=ever)
                obs,_,_,_=env.step(action)
                recorder.finish_action()
                actions.append(action);history.update(obs,action);previous=float(action[-1])
                success=bool(env.check_success())
                if success:break
        qualified=event is not None and not success
        if not recorder.physics_instrumented or recorder.physics_step_count<len(actions):
            raise RuntimeError("D1 physics-step instrumentation unavailable")
        cause_records=recorder.records if event is None else [row for row in recorder.records if row["step"]>=event["anchor_global_step"]]
        labels=label_trace(cause_records,task)
        env_hash=digest({"task":task,"reset_seed":0,"init_hash":init["state_hash"],
                         "camera":False,"fresh_environment":True})
        result=dict(schema_version=1,runtime_receipt_sha256=os.environ.get("S1_RUNTIME_RECEIPT_SHA256"),elapsed_seconds=time.monotonic()-started,source_commit=os.environ.get("S1_SOURCE_COMMIT"),pai_run_id=os.environ.get("PAI_CANARY_RUN_ID"),job_id=os.environ.get("S1_JOB_ID"),event_instance_id=eid,task=task,init_state_id=init_state_id,
          actor_seed=actor_seed,split=init["split"],pid=os.getpid(),env_hash=env_hash,
          chunk_hash=chunk_hash(chunk) if event is None else event["original_chunk_hash"],clean_success=success,
          steps=len(actions),structural_anchor_found=event is not None,qualified_natural_failure=qualified,
          event=event if qualified else None,labels=labels,trace_path=str(trace_path),
          trace_sha256=file_sha(trace_path),status="COMPLETE",zero_injection=True)
        return result
    finally:env.close()

def init_pool_worker(task,output_root):
    out=Path(output_root)/"init_pool"/f"{task}.json"
    if out.exists():return str(out)
    shards=Path(output_root)/"init_pool"/"shards"/task
    states=[]
    for i in range(100):
        guard_execution();p=shards/f"{i:03d}.json"
        if not p.exists():atomic_json(p,make_init(task=task,init_state_id=i))
        states.append(json.loads(p.read_text()))
    hashes=[s["state_hash"] for s in states]
    if len(set(hashes))!=100:raise RuntimeError("duplicate reset states; resampling forbidden")
    atomic_json(out,dict(schema_version=1,task=task,states=states,seed_formula="2016214+10000*task_index+init_id"))
    return str(out)

def init_pool(task,output_root):
    return Path(spawn_call("experiments.r16_p14_stage2f.s1.collector","init_pool_worker",dict(task=task,output_root=str(output_root))))

def run_task(task,output_root,device="cpu",workers=1,infra_only=False):
    from concurrent.futures import ThreadPoolExecutor,as_completed
    out=Path(output_root);pool=init_pool(task,out)
    cases=[(seed,i) for i in (range(1) if infra_only else range(100)) for seed in (SEEDS[:1] if infra_only else SEEDS)]
    import threading
    cancelled=threading.Event()
    def run(case):
        if cancelled.is_set():return
        seed,i=case;split=split_for(i)
        target=out/("sealed_evaluation" if split=="evaluation" else "episodes")/task/f"init{i:03d}__actor{seed}.json"
        if target.exists():return
        row=spawn_call("experiments.r16_p14_stage2f.s1.collector","collect_episode",dict(task=task,init_state_id=i,actor_seed=seed,
              pool_path=str(pool),output_root=str(out),device=(f"cuda:{(i*3+SEEDS.index(seed))%2}" if device=="cuda" else device)),cancel_event=cancelled)
        atomic_json(target,row)
        if split=="evaluation":target.chmod(0o600)
        # Expose only qualification metadata before the selection receipt.
        atomic_json(out/"qualification"/task/f"init{i:03d}__actor{seed}.json",
            {k:row[k] for k in ("event_instance_id","task","init_state_id","actor_seed","split","qualified_natural_failure","status")})
        state_dir=os.environ.get("PAI_CANARY_RUN_DIR")
        if state_dir:
            first=Path(state_dir)/"pai_state/FIRST_REAL_WORK.json"
            import fcntl
            first.parent.mkdir(parents=True,exist_ok=True)
            with first.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if not first.exists():atomic_json(first,dict(uid=os.getuid(),gid=os.getgid(),shard=str(target),sha256=file_sha(target),task=task))
        print(f"persisted episode task={task} init={i} seed={seed}",flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures=[ex.submit(run,c) for c in cases]
        try:
            for f in as_completed(futures):f.result()
        except BaseException:
            cancelled.set()
            for f in futures:f.cancel()
            marker=os.environ.get("S1_STOP_FILE")
            if marker:Path(marker).write_text("S1_COLLECTION_FAILURE")
            raise
    counts={s:0 for s in ("infrastructure","calibration","evaluation","reserve")}
    for p in (out/"qualification"/task).glob("*.json"):
        row=json.loads(p.read_text());counts[row["split"]]+=int(row["qualified_natural_failure"])
    summary=dict(task=task,counts=counts,K1_pass=counts["calibration"]>=25 and counts["evaluation"]>=25,
                 expected_episodes=len(cases),infra_only=infra_only)
    atomic_json(out/f"summary_{task}.json",summary,immutable=False)
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--task",choices=TASKS,required=True)
    p.add_argument("--output-root",required=True);p.add_argument("--device",default="cpu")
    p.add_argument("--workers",type=int,default=1);p.add_argument("--infra-only",action="store_true");p.add_argument("--init-only",action="store_true")
    args=p.parse_args();init_only=vars(args).pop("init_only")
    if init_only:print(init_pool(args.task,args.output_root))
    else:print(json.dumps(run_task(**vars(args)),indent=2))
