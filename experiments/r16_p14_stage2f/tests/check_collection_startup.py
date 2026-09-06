"""CPU-only exact init-state reproduction in the formal spawn bootstrap."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.r16_p14_stage2f.s1.common import spawn_call,TASKS
from experiments.r16_p14_stage2f.s1.assets import configure_assets
def main():
    started=time.monotonic();rows=[]
    for task in TASKS:
        actual=spawn_call("experiments.r16_p14_stage2f.s1.collector","make_init",dict(task=task,init_state_id=0))
        expected=json.loads((ROOT/"artifacts/stage2f/phase0b/init_pool"/f"{task}.json").read_text())["states"][0]
        assert actual==expected, "frozen reset state differs under formal bootstrap"
        rows.append({"task":task,"init_state_id":0,"state_hash":actual["state_hash"],"exact_reproduction":True})
    result={"scope":"CPU only; reset reconstruction; no actor/GPU/calibration/evaluation","passed":True,
       "rows":rows,"assets":configure_assets(),"elapsed_seconds":time.monotonic()-started}
    path=ROOT/"artifacts/stage2f/preflight/collection_startup.json"
    path.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
