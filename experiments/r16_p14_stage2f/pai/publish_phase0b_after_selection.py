"""Publish sealed Phase0B sources only after real diagnostic Git admission."""
from pathlib import Path
import sys,json,hashlib,shutil,datetime
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/r16_p14_stage2f'))
from s1.matrix import diagnostic_selected_budget
SOURCE=Path('/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2f/s1-20260907/artifacts/stage2f')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 budget=diagnostic_selected_budget(SOURCE)
 dst=ROOT/'artifacts/stage2f/phase0b';src=SOURCE/'phase0b';manifest=dst/'source_file_manifest.json'
 before=manifest.read_bytes();data=json.loads(before)
 def copy(item):
  source=src/item['path'];target=dst/item['path']
  assert source.stat().st_uid==source.stat().st_gid==2254
  assert source.stat().st_size==item['bytes'] and sha(source)==item['sha256']
  if target.exists():assert sha(target)==item['sha256']
  else:
   target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
  assert target.stat().st_size==item['bytes'] and sha(target)==item['sha256']
  return dict(item,published=True)
 with ThreadPoolExecutor(max_workers=8) as pool:data['files']=list(pool.map(copy,data['files']))
 episodes=[];events=[]
 for entry in data['files']:
  if entry['path'].startswith(('episodes/','sealed_evaluation/')):
   row=json.loads((dst/entry['path']).read_text());assert row['status']=='COMPLETE'
   episodes.append(row)
   if row['qualified_natural_failure']:assert row['event'] is not None;events.append(row['event'])
 assert len(episodes)==600 and len(events)==79 and sum(e['split']=='evaluation' for e in events)==36
 events=sorted(events,key=lambda e:(e['task'],e['init_state_id'],e['generator_actor_seed']))
 (dst/'events.jsonl').write_text(''.join(json.dumps(e,sort_keys=True)+'\n' for e in events))
 data.update(published_files=1200,sealed_files=0,evaluation_outcome_read=True,sealed_evaluation_kept_on_cpfs=True,diagnostic_selection_authorization_sha256=sha(SOURCE/'phase1/diagnostic_selection_authorization.json'))
 manifest.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
 receipt=dict(status='PASS',evaluation_read=True,diagnostic_admission_checked_before_source_reads=True,budget=budget,prior_manifest_sha256=hashlib.sha256(before).hexdigest(),current_manifest_sha256=sha(manifest),published_source_files=1200,episode_count=600,qualified_events=79,evaluation_events=36,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),retained_cpfs_source=True)
 (ROOT/'artifacts/stage2f/preflight/phase0b_evaluation_publication_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
if __name__=='__main__':main()
