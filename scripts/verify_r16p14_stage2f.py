#!/usr/bin/env python3
"""Verify Stage-2F's published CPU result and explicit blocked GPU disposition.

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
    for phase,gate in [('phase0b','K1'),('phase1','K2'),('phase2','K3')]:
        summary=read(f'artifacts/stage2f/{phase}/summary.json')
        require(summary['execution_status']=='NOT_RUN' and summary['gate_status']=='NOT_EVALUATED' and summary['gate']==gate and summary['observations'] is None and summary['gpu_hours']==0.0,f'fabricated {phase} result')
    for name in ['phase0b/events.jsonl','phase1/grid_rows.jsonl','phase1/selection_receipt.json','phase2/atlas_rows.jsonl','phase2/boundaries.jsonl','phase2/crossing.json','phase2/null_distribution.json']:
        require(not (ROOT/'artifacts/stage2f'/name).exists(),f'unexpected unexecuted output {name}')
    jobs=read('experiments/r16_p14_stage2f/pai/jobs.json')
    require(jobs['jobs']==[] and jobs['submission_attempts']==0 and jobs['total_gpu_hours']==0.0,'unexpected PAI use')
    checks.append({'check':'no_fabricated_gpu_work','K1':'NOT_EVALUATED','K2':'NOT_EVALUATED','K3':'NOT_EVALUATED','gpu_hours':0.0})
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
    return {'status':'PASS','scope':'published_cpu_analysis_and_blocked_gpu_disposition_only','scientific_S1_success':False,'checks':checks}
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--skip-worktree',action='store_true',help='Only verify committed frozen-path trees; default also checks tracked working files')
    args=parser.parse_args()
    try: result=verify(args.skip_worktree)
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error':str(exc)},ensure_ascii=False));return 1
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0
if __name__=='__main__': sys.exit(main())
