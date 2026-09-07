"""Validate all sealed evaluation source traces after real diagnostic admission."""
import argparse,concurrent.futures,gzip,hashlib,json,sys,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/r16_p14_stage2f'))
from s1.matrix import diagnostic_selected_budget
from s1.measurement import label_trace
SOURCE='6b788a0764904e11e022c4330a74fa3e009c9a33'
CHECKPOINTS={7:'821177a82cc470e108082fd3c0f6913983236a2fdf142de2fe51fc37c44240ca',17:'83ee61e31ffdae6f2ef57203a2c0085df41e284039cd81c6ecf1210694521604',29:'0cf34a3e535525345306a2b322aae3b1bd6ebd6cd71dc653e2a91393e2b79d1a'}
def sha(blob):return hashlib.sha256(blob).hexdigest()
def check_one(path):
 p=Path(path);raw=p.read_bytes();d=json.loads(raw)
 assert p.stat().st_uid==p.stat().st_gid==2254
 assert d['status']=='COMPLETE' and d['split'] in {'infrastructure','calibration','evaluation','reserve'} and d['source_commit']==SOURCE and d['zero_injection'] is True
 assert 0<=d['init_state_id']<=99 and d['actor_seed'] in CHECKPOINTS and d['pid']>0 and d['env_hash'] and d['chunk_hash']
 assert d['job_id'] in {'dlc1hzmadm185c68','dlc1rz7o5mf1vajn'}
 tp=Path(d['trace_path']);assert tp.stat().st_uid==tp.stat().st_gid==2254
 compressed=tp.read_bytes();assert sha(compressed)==d['trace_sha256']
 content=gzip.decompress(compressed);records=[json.loads(l) for l in content.splitlines()]
 assert len(records)==26*d['steps']
 label_count=d['labels']['record_count'];assert 0<label_count<=len(records) and label_count%26==0
 label_start=(len(records)-label_count)//26
 if not d['structural_anchor_found']:assert label_start==0
 if d['event'] is not None:assert label_start==d['event']['anchor_global_step']
 for index,x in enumerate(records):
  step,within=divmod(index,26)
  assert x['control_step']==step and x['step']==step and x['actor_seed']==d['actor_seed'] and x['operator']=='clean_actor'
  assert x['event_instance_id']==d['event_instance_id'] and x['task_phase']['task']==d['task']
  assert x['missing']==[] and x['contact_stream_available'] is True
  assert x['step_kind']==('physics_step' if within<25 else 'action_step')
  assert x['substep']==(within+1 if within<25 else 0)
  assert x['physics_step_index']==(step*25+within+1 if within<25 else None)
  pairs=sorted(set(tuple(sorted((v['geom1_name'],v['geom2_name']))) for v in x['raw_contact_pairs']))
  assert pairs==[tuple(v) for v in x['normalized_contact_pairs']]
  assert x['contact_count']==len(x['raw_contact_pairs'])
  assert x['object_qpos'] is not None and x['target_qpos'] is not None
  if within<25:assert x['action_hash']==records[step*26+25]['action_hash']
 assert label_trace(records[label_start*26:],d['task'])==d['labels'],'offline labels differ'
 assert records[-1]['task_phase']['task_success'] is d['clean_success']
 assert any(x['task_phase']['task_success'] for x in records if x['step_kind']=='action_step') is d['clean_success']
 if d['qualified_natural_failure']:
  assert not d['clean_success'] and isinstance(d['event'],dict)
  event=d['event'];assert event['checkpoint_sha256']==CHECKPOINTS[d['actor_seed']]
  assert event['task']==d['task'] and event['split']==d['split'] and event['init_state_id']==d['init_state_id']
 return dict(path=str(p),sha256=sha(raw),trace_path=str(tp),trace_sha256=sha(compressed),trace_content_sha256=sha(content),task=d['task'],init_state_id=d['init_state_id'],actor_seed=d['actor_seed'],steps=d['steps'],trace_records=len(records),label_scope_start_control_step=label_start,label_scope='collector post-anchor suffix; full trace if no anchor; successful anchor index inferred from preserved label record count',qualified=d['qualified_natural_failure'],clean_success=d['clean_success'],job_id=d['job_id'],source_commit=d['source_commit'],split=d['split'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--split',choices=['evaluation','all'],default='evaluation');a=ap.parse_args()
 assert 1<=a.workers<=8
 budget=diagnostic_selected_budget(a.input_root)
 paths=sorted((a.input_root/'phase0b/sealed_evaluation').glob('*/*.json'))
 if a.split=='all':paths+=sorted((a.input_root/'phase0b/episodes').glob('*/*.json'))
 expected=240 if a.split=='evaluation' else 600
 assert len(paths)==expected
 rows=[];errors=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(check_one,str(p)):p for p in paths}
  for n,future in enumerate(concurrent.futures.as_completed(futures),1):
   try:rows.append(future.result())
   except Exception as exc:errors.append(dict(path=str(futures[future]),error=repr(exc)))
   if n%30==0:print(json.dumps(dict(done=n,total=expected,errors=len(errors))),flush=True)
 by_task={}
 for row in rows:
  t=by_task.setdefault(row['task'],dict(episodes=0,qualified=0,clean_success=0,steps=0,trace_records=0))
  t['episodes']+=1;t['qualified']+=int(row['qualified']);t['clean_success']+=int(row['clean_success']);t['steps']+=row['steps'];t['trace_records']+=row['trace_records']
 report=dict(status='PASS' if len(rows)==expected and not errors else 'BLOCKED',evaluation_read=True,diagnostic_admission_verified_before_evaluation=True,source_commit=SOURCE,budget=budget,validated_episodes=len(rows),requested_split=a.split,expected_episodes=expected,errors=errors,by_task=by_task,checks=['exact_job_source_owner_identity','compressed_SHA','25_physics_plus_action_step_order','normalized_complete_contact_pairs','per_control_action_hash','offline_labels_exact_reproduction','clean_success_control_end','qualified_event_checkpoint_binding'],non_event_checkpoint_scope='nonqualified metadata has no per-event checkpoint field; generator actor identity bound by seed, collection source and published checkpoint assets',script_sha256=sha(Path(__file__).read_bytes()),checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=sorted(rows,key=lambda r:r['path']))
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files'}));return 0 if report['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
