"""Reproduce six calibration-only matched traces at diagnostic candidate budget."""
from pathlib import Path
import json,gzip,hashlib,math
ROOT=Path(__file__).resolve().parents[3]
SPECS=json.loads("[[\"bowl_hold\",\"put_the_bowl_on_the_plate\",\"cd2911f0c476838d70587c701241b1cf3b53d3f4e9f8694e0f41e3d934da7b48\",\"23c6dfccb6ac29831fb1150df37c6981a09aa9b4843622ab5243f009f5e8e815\",true,629,24,6],[\"bowl_fresh4\",\"put_the_bowl_on_the_plate\",\"9cc121c6d3264a81aae05076c7a25b3fa9886dcb1b5450e277b4fef047e6e811\",\"241d245ebe509262d8bdb23567de5485d4dc5555be7acb6c7322b00f0677f904\",false,null,null,null],[\"cream_hold\",\"put_the_cream_cheese_in_the_bowl\",\"552be14b42ee562f1d0854f7d4de486875e3bb4f1f3fac4430a1e91199eca1a7\",\"5b01b774dc730d413fb902e93777619af30d832df5d152789fd780f2742c68c7\",false,null,null,null],[\"cream_fresh4\",\"put_the_cream_cheese_in_the_bowl\",\"e3a7ba9f20d7ecf36a63a5d665b27381c81abfd8a530bafdc6049185f4f2460b\",\"b890cc2c8a126303c1d6e1094e277dc491e6e5bda6e97903ef52b0a0c1b06741\",true,527,20,8],[\"cream_rollback\",\"put_the_cream_cheese_in_the_bowl\",\"534c79fd8e982b20606dc735e1146676b0646fcaae931ebbde3d7653971dc210\",\"8f858a51fa6cba0a9cebcfb95e0a582f80aeafa5308e1b7ec518e25b60b9df53\",false,null,null,null],[\"cream_fresh16\",\"put_the_cream_cheese_in_the_bowl\",\"1cee34fc7cffffa3f955d76bbd598d874b281bb24f1236a959867693022d0018\",\"da365c08681b60bd073391962427b7cba78965b929a6e939e9bed4f35e2f4240\",true,476,18,9]]")
def sha(b):return hashlib.sha256(b).hexdigest()
def compact(x,i):
 d={k:x.get(k) for k in ['step_kind','control_step','substep','physics_step_index','action','action_hash','object_qpos','target_qpos','normalized_contact_pairs','task_phase','current_gripper','previous_gripper']}
 d['trace_record_index']=i
 d['object_target_xy_m']=math.dist(x['object_qpos'][:2],x['target_qpos'][:2])
 return d
def main():
 result=[];pairs=[]
 for name,task,key,expected,safe,first_idx,step,substep in SPECS:
  rel=f'artifacts/stage2f/phase1/shards/{task}/{key}.json';p=ROOT/rel;raw=json.loads(p.read_text())
  trace_rel=f'artifacts/stage2f/phase1/contact_topology/{task}/{key}.jsonl.gz';tb=(ROOT/trace_rel).read_bytes();content=gzip.decompress(tb)
  assert sha(tb)==expected==raw['trace_sha256'];assert sha(content)==raw['trace_content_sha256']
  assert raw['split']=='calibration' and raw['status']=='COMPLETE' and raw['safe_success'] is safe and raw['release_based_violation'] is False
  assert (raw['prefix_k'],raw['tail_horizon'],raw['action_budget'],raw['policy_call_cap'])==(2,16,32,8)
  records=[json.loads(l) for l in content.splitlines()];actions=[(i,x) for i,x in enumerate(records) if x['step_kind']=='action_step']
  assert len(records)==raw['trace_records'] and len(records)==26*len(actions)
  for start in range(0,len(records),26):
   block=records[start:start+26];assert [x['substep'] for x in block[:25]]==list(range(1,26));assert block[-1]['step_kind']=='action_step'
   for x in block:
    normalized=sorted(set(tuple(sorted((v['geom1_name'],v['geom2_name']))) for v in x['raw_contact_pairs']))
    assert normalized==[tuple(v) for v in x['normalized_contact_pairs']] and not x['missing']
  success=[(i,x) for i,x in enumerate(records) if x['task_phase']['task_success']]
  assert (success[0][0] if success else None)==first_idx
  if success:assert (success[0][1]['control_step'],success[0][1]['substep'])==(step,substep)
  assert actions[-1][1]['task_phase']['task_success'] is safe
  result.append(dict(name=name,raw_shard_path=rel,raw_shard_sha256=sha(p.read_bytes()),trace_path=trace_rel,trace_sha256=sha(tb),trace_content_sha256=sha(content),
   binding={k:raw[k] for k in ['event_instance_id','init_state_id','source_commit','generator_checkpoint_sha256','recovery_checkpoint_sha256','generator_actor_seed','recovery_actor_seed','operator','prefix_k','tail_horizon','action_budget','policy_call_cap']},
   d4={k:raw['d4_signatures'][k]['complete_signature_hash'] for k in ['detection','pre_tail']},
   safe_success=safe,release_based_violation=False,first_success=compact(success[0][1],success[0][0]) if success else None,
   first_new_action=compact(actions[2][1],actions[2][0]),final_control=compact(actions[-1][1],actions[-1][0]),
   release_transitions=[compact(x,i) for i,x in actions if x['task_phase']['release_transition']],
   contacts_union=sorted(set(tuple(v) for x in records for v in x['normalized_contact_pairs'])),
   budget={k:raw[k] for k in ['effective_execution_horizon','actual_new_recovery_actions','actual_policy_calls','total_policy_calls','validation_policy_calls','task_horizon_remaining_after_prefix']},
   prefix_hashes=[x['action_hash'] for _,x in actions[:2]]))
 for i in [0,2,4]:
  a,b=result[i:i+2]
  for k in ['event_instance_id','init_state_id','source_commit','generator_checkpoint_sha256','recovery_checkpoint_sha256','generator_actor_seed','recovery_actor_seed','prefix_k','tail_horizon','action_budget','policy_call_cap']:assert a['binding'][k]==b['binding'][k]
  assert a['d4']==b['d4'] and a['prefix_hashes']==b['prefix_hashes']
  assert a['first_new_action']['action_hash']!=b['first_new_action']['action_hash']
  assert a['first_new_action']['normalized_contact_pairs']==b['first_new_action']['normalized_contact_pairs']
  pairs.append(dict(arms=[a['name'],b['name']],matched_binding=True,matched_prefix=True,first_divergent_action_index=2))
 out=ROOT/'artifacts/stage2f/phase1/mechanism_selected_budget_parent_acceptance.json'
 out.write_text(json.dumps(dict(status='PASS',scope='calibration_descriptive_matched_cases_no_unmeasured_force_claim',evaluation_read=False,trace_count=6,cases=result,pairs=pairs),indent=2,sort_keys=True)+'\n')
 print(json.dumps(dict(status='PASS',trace_count=6,pairs=pairs)))
if __name__=='__main__':main()
