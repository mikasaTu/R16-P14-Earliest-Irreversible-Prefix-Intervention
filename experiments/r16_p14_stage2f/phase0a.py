#!/usr/bin/env python3
"""Reuse frozen S0 A in memory, extending only controls and output location.

CPU only. The frozen source and historical artifacts are never edited.
"""
from pathlib import Path
import ast, csv, hashlib, json, sys, types
sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'scripts/run_r16p14_stage2e_s0.py'
OUT=ROOT/'artifacts/stage2f/phase0a'
CONTROLS=['fixed_delay_1','fixed_delay_2','fixed_delay_4','fixed_delay_8','immediate_fresh_h16']
def adapted_module():
    tree=ast.parse(SOURCE.read_text(),filename=str(SOURCE))
    changes=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name=='_fixed_winner':
            candidates=[n for n in ast.walk(node) if isinstance(n,ast.Set) and {x.value for x in n.elts if isinstance(x,ast.Constant)}==set(CONTROLS[:-1])]
            assert len(candidates)==1
            candidates[0].elts.append(ast.Constant(value=CONTROLS[-1]));changes.append('add_immediate_fresh_h16_to_control_set')
        if isinstance(node,ast.FunctionDef) and node.name=='run_a':
            paths=[n for n in ast.walk(node) if isinstance(n,ast.Constant) and n.value=='artifacts/stage2e/s0/common_support']
            assert len(paths)==1
            paths[0].value='artifacts/stage2f/phase0a';changes.append('redirect_A_outputs_to_stage2f')
    assert len(changes)==2
    ast.fix_missing_locations(tree)
    m=types.ModuleType('_stage2f_frozen_A_adapter');m.__file__=str(SOURCE)
    exec(compile(tree,str(SOURCE),'exec'),m.__dict__)
    return m,changes
def run():
    m,changes=adapted_module();OUT.mkdir(parents=True,exist_ok=True)
    values={};receipt={'files':{},'issues':[],'scope':'only four frozen Stage2C A inputs; no S1 outcomes'}
    for name in ['c_recovery','c_boundaries','c_invalid','c_baseline']:
        path=ROOT/m.INPUT_PATHS[name];digest=m.sha256_file(path)
        if digest!=m.EXPECTED_HASHES[name]:raise RuntimeError('frozen input hash mismatch: '+name)
        values[name]=m.load_plain_jsonl(path);m.validate_rows(name,values[name])
        receipt['files'][name]={'path':m.INPUT_PATHS[name],'sha256':digest,'rows':len(values[name])}
    # A discriminating control-selection test: stronger immediate must win;
    # a still stronger ineligible heuristic must remain outside the control set.
    assert m._fixed_winner([{'method':x,'safe_success':.1} for x in CONTROLS[:-1]]+[{'method':CONTROLS[-1],'safe_success':.2},{'method':'action_disagreement','safe_success':.9}])==CONTROLS[-1]
    summary=m.run_a(ROOT,values,receipt,replicates=10000)
    if not summary['A1_exact_reproduction']:raise RuntimeError('published A1 mismatch')
    (OUT/'table.csv').replace(OUT/'source_method_table.csv')
    rows=[];comparisons=[]
    recovery_lookup=m._raw_recovery_lookup(values['c_recovery'])
    calibration=[row for row in values['c_baseline'] if row['split']=='calibration']
    for cohort,result in summary['cohorts'].items():
        events=m._cohort_events(values['c_boundaries'],values['c_invalid'],valid=cohort=='replay_valid_subset')
        for method in result['method_summaries']:
            rows.append({'cohort':cohort,'method':method['method'],'restricted_safe_success':method['restricted_support']['safe_success'],'full_safe_success':method['full_support']['safe_success'],'restricted_events':method['restricted_support']['event_count'],'full_events':method['full_support']['event_count'],'full_fallback_events':method['full_support']['fallback_event_count'],'estimand':'event mean of three-actor means; restricted is method-defined support','diagnostic_only':True})
        for restricted in [True,False]:
            ks,kr=m._method_summary('k_last_recoverable',events,calibration,recovery_lookup,restricted=restricted)
            ids={x['event_instance_id'] for x in kr}
            for comparator in dict.fromkeys([result['strongest_fixed_delay'],'immediate_fresh_h16']):
                _,cr=m._method_summary(comparator,events,calibration,recovery_lookup,restricted=False)
                cr=[x for x in cr if x['event_instance_id'] in ids]
                comparisons.append({'cohort':cohort,'support':'restricted' if restricted else 'full','comparator':comparator,'events':len(kr),'k_last_recoverable_safe_success':m.mean(x['safe_success'] for x in kr),'comparator_safe_success_same_events':m.mean(x['safe_success'] for x in cr),'absolute_level_estimator':'event mean of three-actor means','k_cluster_safe_success':m.cluster_bootstrap([{**x,'value':x['safe_success']} for x in kr],replicates=10000,seed=216214)['estimate'],'comparator_cluster_safe_success':m.cluster_bootstrap([{**x,'value':x['safe_success']} for x in cr],replicates=10000,seed=216214)['estimate'],'paired_cluster_delta':m.cluster_bootstrap(m._delta_rows(kr,cr),replicates=10000,seed=216214)})
        result['strongest_reference']=result['strongest_fixed_delay']
    m.write_csv(OUT/'table.csv',rows,list(rows[0]))
    summary.update({'phase':'0A','control_candidates':CONTROLS,'source_script_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'adapter_changes':changes,'bootstrap_replicates':10000,'bootstrap_seed':216214,'absolute_comparisons_same_support':comparisons,'source_inputs':receipt,'gpu_hours':0.0,'simulation_steps':0,'performance_advantage_claimed':False,'legacy_field_note':'strongest_fixed_delay is the frozen run_a key; its candidate set here includes immediate_fresh_h16 and is also named strongest_reference.'})
    m.safe_dump(OUT/'summary.json',summary)
    m.safe_dump(OUT/'tests.json',{'status':'PASS','tests':['stronger_immediate_is_eligible','non_control_heuristic_excluded','four_input_SHA256_bindings','frozen_input_schema_checks','nine_A1_counts_exactly_reproduced','same_event_absolute_comparisons'],'source_file_unchanged':hashlib.sha256(SOURCE.read_bytes()).hexdigest()==summary['source_script_sha256'],'bootstrap_replicates':10000,'bootstrap_seed':216214})
    print(json.dumps({'G0_1':summary['G0_1'],'cohorts':{name:{'reference':v['strongest_reference'],'delta':v['delta']} for name,v in summary['cohorts'].items()},'absolute_comparisons':comparisons},ensure_ascii=False))
if __name__=='__main__':run()
