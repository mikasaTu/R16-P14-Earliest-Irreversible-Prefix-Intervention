#!/usr/bin/env python3
"""Verify Stage-2F's published artifacts, frozen protocol, and execution disposition.

Run in a complete clone; standard library only. This does not run a simulator.
"""
import argparse, hashlib, importlib.util, json, math, subprocess, sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BASE = "508e8a5c88780066a00d994c32d673d558594867"
BLOCK = "BLOCKED_BY_FROZEN_SPAWN_ZERO_INJECTION_CONTRACT"
def read(p): return json.loads((ROOT / p).read_text())
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def verify(skip_worktree=False):
    checks=[]
    manifest=ROOT/'artifacts/stage2f/SHA256SUMS'
    declared={}
    for line in manifest.read_text().splitlines():
        digest,rel=line.split("  ",1)
        path=Path(rel)
        require(not path.is_absolute() and ".." not in path.parts,"unsafe checksum path")
        require(rel not in declared,"duplicate checksum entry")
        require(len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'bad digest')
        require((ROOT/path).is_file(),f'missing {rel}')
        require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,f'hash mismatch: {rel}')
        declared[rel]=digest
    expected=set()
    for folder in ['artifacts/stage2f','experiments/r16_p14_stage2f']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p != manifest and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.pyo'}:
                expected.add(p.relative_to(ROOT).as_posix())
    expected.add('scripts/verify_r16p14_stage2f.py')
    require(set(declared)==expected,f'checksum coverage mismatch: {sorted(set(declared)^expected)}')
    checks.append({'check':'output_hashes_and_coverage','files':len(declared)})
    forbidden=[f'{base}/stage2{x}' if base=='artifacts' else f'{base}/r16_p14_stage2{x}' for base in ['artifacts','experiments'] for x in 'abcde']
    committed=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only',BASE,'HEAD','--',*forbidden],text=True).strip()
    require(not committed,f'frozen committed paths changed: {committed}')
    if not skip_worktree:
        dirty=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only','HEAD','--',*forbidden],text=True).strip()
        require(not dirty,f'frozen worktree paths changed: {dirty}')
    checks.append({'check':'frozen_predecessor_paths','base':BASE,'working_files_checked':not skip_worktree})
    p=ROOT/'experiments/r16_p14_stage2f/preflight.py'
    spec=importlib.util.spec_from_file_location('stage2f_static_preflight',p); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    live=module.inspect(ROOT); stored=read('artifacts/stage2f/preflight/zero_injection_audit.json')
    require(live==stored,'source audit does not reproduce')
    require(live['status']==BLOCK and not live['forbidden_function_executed'],'blocker disposition mismatch')
    checks.append({'check':'static_injection_call_chain','status':BLOCK})
    prereg=read('artifacts/stage2f/preflight/prereg_freeze.json')
    require(hashlib.sha256((ROOT/'experiments/r16_p14_stage2f/PREREG_S1.md').read_bytes()).hexdigest()==prereg['prereg_sha256'],'preregistered K criteria changed')
    amendment=read('artifacts/stage2f/preflight/execution_amendment_receipt.json')
    require(hashlib.sha256((ROOT/'experiments/r16_p14_stage2f/AMENDMENT_S1_EXECUTION.md').read_bytes()).hexdigest()==amendment['amendment_sha256'],'execution amendment changed')
    phase_status={}
    for phase,gate in [('phase0b','K1'),('phase1','K2'),('phase2','K3')]:
        summary=read(f'artifacts/stage2f/{phase}/summary.json')
        phase_status[gate]=summary.get('gate_status',summary.get('status',summary.get('execution_status')))
    pools=list((ROOT/'artifacts/stage2f/phase0b/init_pool').glob('*.json'))
    require(len(pools)==2,'two fixed init pools required')
    for pool in pools:
        d=json.loads(pool.read_text());states=d['states']
        require(len(states)==100 and len({x['state_hash'] for x in states})==100,'invalid or duplicate reset pool')
        require([x['init_state_id'] for x in states]==list(range(100)),'init ids changed')
        for item in states:
            i=item['init_state_id'];split='infrastructure' if i<10 else 'calibration' if i<50 else 'evaluation' if i<90 else 'reserve'
            require(item['split']==split,'split changed')
    jobs=read('experiments/r16_p14_stage2f/pai/jobs.json')
    require(jobs['max_concurrent_jobs']==2 and jobs['max_gpu_per_job']==2 and jobs['gpu_hours_limit']==20,'resource caps drifted')
    ids=[j['job_id'] for j in jobs['jobs']]
    require(len(ids)==len(set(ids)),'duplicate job identity')
    for job in jobs['jobs']:
        require(int(job.get('gpus',2))<=2,'per-job GPU cap exceeded')
        if job.get('status')=='Succeeded':require(job.get('persisted_completion_verified') is True,'Succeeded without artifact verification')
    receipt=ROOT/'artifacts/stage2f/phase1/selection_receipt.json'
    auth=ROOT/'artifacts/stage2f/phase1/selection_authorization.json'
    atlas=ROOT/'artifacts/stage2f/phase2/atlas_rows.jsonl'
    if atlas.exists():
        require(receipt.exists() and auth.exists(),'evaluation lacks committed calibration selection')
        authorization=json.loads(auth.read_text())
        require(authorization['selection_receipt_sha256']==hashlib.sha256(receipt.read_bytes()).hexdigest(),'selection receipt mismatch')
        require(len(authorization.get('git_commit',''))==40,'selection commit missing')
        proof_spec=importlib.util.spec_from_file_location("stage2f_selection_proof",ROOT/"experiments/r16_p14_stage2f/s1/selection.py")
        proof_module=importlib.util.module_from_spec(proof_spec);proof_spec.loader.exec_module(proof_module)
        require(proof_module.verify_commit_proof(receipt,authorization),"selection Git object proof invalid")
    checks.append({'check':'frozen_S1_protocol_and_available_execution_evidence','phase_status':phase_status,'PAI_jobs':len(ids)})
    summary=read('artifacts/stage2f/phase0a/summary.json')
    require(summary.get('diagnostic_only') is True,'Phase0A not diagnostic-only')
    require((ROOT/'artifacts/stage2f/phase0a/table.csv').stat().st_size>0,'missing Phase0A table')
    require(summary['control_candidates']==['fixed_delay_1','fixed_delay_2','fixed_delay_4','fixed_delay_8','immediate_fresh_h16'],'incorrect control expansion')
    require(summary['bootstrap_replicates']==10000 and summary['bootstrap_seed']==216214,'bootstrap contract drift')
    require(summary['A1_exact_reproduction'] is True,'A1 not reproduced')
    require(summary['G0_1']=='INCONCLUSIVE','unexpected G0 claim')
    require(len(summary['absolute_comparisons_same_support'])==4,'missing or duplicate paired absolute comparisons')
    for item in summary['source_inputs']['files'].values():
        require(hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'],'Phase0A source input drift')
    require(hashlib.sha256((ROOT/'scripts/run_r16p14_stage2e_s0.py').read_bytes()).hexdigest()==summary['source_script_sha256'],'frozen A source script drift')
    for c in summary['absolute_comparisons_same_support']:
        d=c['paired_cluster_delta']
        require(math.isclose(c['k_cluster_safe_success']-c['comparator_cluster_safe_success'],d['estimate'],abs_tol=1e-12),'absolute/delta estimand mismatch')
        require(d['replicates']==10000 and d['seed']==216214,'comparison bootstrap drift')
        require(d['ci95'][0]<0<d['ci95'][1],'unexpected significance claim')
    checks.append({'check':'phase0a_frozen_inputs_A1_controls_and_estimands','input_files':4,'A1_methods':9,'same_support_comparisons':4})
    checks.append({'check':'phase0a_outputs_present','diagnostic_only':True})
    return {'status':'PASS','scope':'integrity_of_published_evidence_not_a_claim_of_experiment_completion','scientific_S1_success':phase_status.get('K3')=='NONTRIVIAL_OPERATOR_RELATIVITY','phase_status':phase_status,'checks':checks}
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--skip-worktree',action='store_true',help='Only verify committed frozen-path trees; default also checks tracked working files')
    args=parser.parse_args()
    try: result=verify(args.skip_worktree)
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error':str(exc)},ensure_ascii=False));return 1
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0
if __name__=='__main__': sys.exit(main())
