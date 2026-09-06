"""Prepare an exact non-Git PAI payload and registered template, never submit."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,tarfile,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
REG=Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/pai-job-registry")
BASE=Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2f/s1-20260907")
ALIAS="exp-robot-oversold-r16p14-stage2f-2gpu"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(phase,task_index,run_id,source_commit,diagnostic_atlas=False):
    tasks=("put_the_cream_cheese_in_the_bowl","put_the_bowl_on_the_plate")
    if diagnostic_atlas and phase!="atlas":raise ValueError("diagnostic atlas requires phase=atlas")
    resolved_commit=subprocess.check_output(["git","rev-parse",source_commit+"^{commit}"],cwd=ROOT,text=True).strip()
    assert resolved_commit==source_commit and (os.getuid(),os.getgid())==(2254,2254)
    payload=Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r16p14-stage2f-payloads")/source_commit
    paths=[f"experiments/r16_p14_stage2{s}" for s in "abcdf"]+["experiments/r16_p14_libero_stage1/libero_config","artifacts/stage2a/actor/checkpoints","libero","artifacts/stage2f/preflight/runtime_compat/runtime_compat_receipt.json"]
    if not payload.exists():
        payload.mkdir(parents=True)
        tar=payload/"source.tar"
        with tar.open("wb") as f:subprocess.run(["git","archive",source_commit,*paths],cwd=ROOT,stdout=f,check=True)
        with tarfile.open(tar) as tf:tf.extractall(payload)
        tar.unlink()
        (payload/"SOURCE_COMMIT").write_text(source_commit+"\n")
    if (payload/"SOURCE_COMMIT").read_text().strip()!=source_commit:raise RuntimeError("payload provenance drift")
    manifest_path=payload/"SOURCE_MANIFEST.json"
    actual={str(p.relative_to(payload)):sha(p) for p in sorted(payload.rglob("*")) if p.is_file() and p!=manifest_path}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())!=actual:raise RuntimeError("source payload files changed")
    else:manifest_path.write_text(json.dumps(actual,indent=2)+"\n")
    source_manifest_sha=sha(manifest_path)
    output=BASE/"artifacts/stage2f";control=BASE/"control";control.mkdir(parents=True,exist_ok=True)
    (BASE/"pai_runs").mkdir(parents=True,exist_ok=True)
    if not (control/"jobs.json").exists():(control/"jobs.json").write_text(json.dumps({"lineage":"s1-20260907","jobs":[]},indent=2)+"\n")
    init=output/"phase0b/init_pool";init.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/"artifacts/stage2f/phase0b/init_pool").glob("*.json"):
        dest=init/p.name
        if dest.exists():
            if sha(dest)!=sha(p):raise RuntimeError("fixed pool differs")
        else:shutil.copy2(p,dest)
    launcher_suffix="_diagnostic" if diagnostic_atlas else ""
    launcher=REG/"launchers"/f"r16p14_stage2f_{phase}_{task_index}_{source_commit[:10]}{launcher_suffix}.sh"
    diagnostic_flag=" --diagnostic-atlas" if diagnostic_atlas else ""
    body=f"""#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=egl WANDB_PROJECT=r16-p14-stage2f-s1
export LIBERO_ASSETS_PATH=/mnt/cpfs/zbl-cpfs-new/dataset/leon/libero/assets/90001343cb134b7e26e18fde0fa2416f3ed6e6a3
export LIBERO_CONFIG_PATH={payload}/experiments/r16_p14_libero_stage1/libero_config
cd {payload}
exec /mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/r16p14_s1_cu124_20260907/bin/python -m experiments.r16_p14_stage2f.s1.pai_entry --phase {phase} --task {tasks[task_index]} --output-root {output} --control-root {control} --source-commit {source_commit} --source-manifest-sha256 {source_manifest_sha} --workers 24{diagnostic_flag}
"""
    if launcher.exists():
        if launcher.read_text()!=body:raise RuntimeError("immutable launcher differs")
    else:
        launcher.write_text(body);launcher.chmod(0o555)
    subprocess.run(["bash","-n",str(launcher)],check=True)
    resources=json.loads((REG/"config/resources.json").read_text())["resources"];resource=resources[ALIAS]
    import runpy
    controller=runpy.run_path(str(REG/"bin/pai-job"),run_name="s1_template_prepare")
    work="collection" if phase=="collect" else "evaluation"
    task_id=f"r16p14-s1-{phase}-task{task_index}"
    model_id="frozen-shared-multitask-act-seeds7-17-29-h16"
    template=dict(schema_version=2,kind="pytorchjob",name_prefix=f"r16p14-s1-{phase}",
      description=f"Stage2F zero-injection {phase} task={tasks[task_index]}; committed S1 protocol; immutable shards; no actor training",
      resource_alias=ALIAS,workspace_id="179169",
      provenance={"contract_source_job_id":resource["contract_source_job_id"],"resource_source_job_id":resource["source_job_id"],
        "submission_method":"cli_create","pai_clone_performed":False,"source_role":"readback_reference"},
      network={},storage={"data_sources":resource["authorized_data_sources"],"output_root":str(BASE/"pai_runs")},
      worker=resource["target_worker_contract"],
      runtime={"command_file":str(launcher),"identity_mechanism":"controller_inline_bootstrap_then_setpriv","uid":2254,"gid":2254,
        "pod_env":{},"secret_env_names":[],"write_paths":["{{ARTIFACT_DIR}}",str(output),str(control)],
        "output_mode":"resume","create_artifact_dir":True,"recursive_repair":False,"payload_transport":"base64"},
      fault_tolerance={"aimaster_args":controller["EXPECTED_IDLE_A800_AIMASTER_ARGS"],"maximum_platform_restarts":50,
        "launcher_attempts":1,"retry_sleep_seconds":30,"pai_automatic_fault_tolerance":True,"application_auto_resume":True},
      evidence={"kind":f"r16p14_stage2f_formal_{work}","idle_8gpu_contract":"generic_formal_quota_idle_2gpu_v1",
        "quota_idle_override":{"authorized":True,"pool":"exp-robot","resource_id":resource["resource_id"],
          "resource_source_job_id":resource["source_job_id"],"source_oversold_type":"ForbiddenQuotaOverSold","target_oversold_type":"AcceptQuotaOverSold"},
        "contract_ready":True,"workload_type":work,"task_id":task_id,"model_id":model_id,
        "success_gate":controller["GENERIC_FORMAL_IDLE_SUCCESS_GATES"][work],"require_actual_idle":True,
        "pai_probe_created":False,"validation_method":"static+dev14_smoke","validation_status":"passed",
        "validated_payload_sha256":sha(launcher),"expected_first_work_uid":2254,"expected_first_work_gid":2254,
        "first_work_evidence_path":"{{ARTIFACT_DIR}}/pai_state/FIRST_REAL_WORK.json",
        "completed_evidence_path":"{{ARTIFACT_DIR}}/pai_state/COMPLETED.json",
        "source_manifest_sha256":source_manifest_sha,"source_commit":source_commit,"source_tree":subprocess.check_output(["git","rev-parse",source_commit+"^{tree}"],cwd=ROOT,text=True).strip(),
        "resume_contract":"immutable completed JSON and gzip shards; exact source and request hash; failed attempts retained",
        "zero_injection":True,"training":False,"selection_open_deny":True,"budget_gpu_hours":20,
        "blackouts_beijing":["09:30-09:40","19:30-19:40"],"external_controller":str(control/"heartbeat.json")},
      submission={"priority":9,"disable_ecs_stock_check":True,"job_reserved_policy":"","job_reserved_minutes":0,
        "job_max_running_time_minutes":290,"tags":{"managed_by":"pai-job-registry","purpose":f"formal-{work}",
        "task":task_id,"model":model_id,"hardware":"2xa800-idle","resource_pool":"exp-robot","experiment_role":"stage2f-s1"}})
    if diagnostic_atlas:
        template["evidence"].update({
            "diagnostic_continuation": True,
            "kind": "r16p14_stage2f_diagnostic_atlas",
            "selection_source": "calibration_only",
            "diagnostic_selection_receipt_path": str(output/"phase1/diagnostic_selection_receipt.json"),
            "diagnostic_selection_authorization_path": str(output/"phase1/diagnostic_selection_authorization.json"),
            "diagnostic_protocol_sha256": "2d14427c2391422dd372b0528d6d784d1066d8d059e1405597cfb1954f260c6a",
            "evaluation_open_deny_until_diagnostic_proof": True,
        })
    target=REG/"templates"/f"{run_id}.json"
    if target.exists():raise RuntimeError("template already exists")
    target.write_text(json.dumps(template,indent=2)+"\n")
    repo_template=ROOT/"experiments/r16_p14_stage2f/pai"/target.name;shutil.copy2(target,repo_template)
    return {"template":str(target),"payload":str(payload),"source_commit":source_commit,"launcher_sha256":sha(launcher),"run_id":run_id}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=("collect","grid","atlas"),required=True)
    p.add_argument("--task-index",type=int,choices=(0,1),required=True);p.add_argument("--run-id",required=True);p.add_argument("--source-commit",required=True)
    p.add_argument("--diagnostic-atlas",action="store_true")
    print(json.dumps(build(**vars(p.parse_args())),indent=2))
