"""Offline tests of the exact installed Stage2F pre-CreateJob admission."""
import ast,datetime as dt,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import fcntl
CANON=Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/pai-job-registry/bin/pai-job")
class Refused(RuntimeError):pass
class Admission(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
  self.expected="/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2f/s1-20260907/control/heartbeat.json"
  node=next(n for n in ast.parse(CANON.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="stage2f_external_admission")
  env={"Path":lambda s:self.root/Path(s).name,"Refused":Refused,"fcntl":fcntl,
       "load_json":lambda p:json.loads(p.read_text()),"atomic_json":lambda p,v:p.write_text(json.dumps(v))}
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(CANON),"exec"),env);self.fn=env[node.name]
  self.res={"resource_alias":"exp-robot-oversold-r16p14-stage2f-2gpu","run_id":"test-s1","evidence":{"external_controller":str(self.root/"heartbeat.json")}}
 def tearDown(self):self.temp.cleanup()
 def at(self,h,m,age=0,resume=True,jobs=None,claims=None):
  fixed=dt.datetime(2026,9,7,h,m,tzinfo=dt.timezone(dt.timedelta(hours=8)))
  (self.root/"heartbeat.json").write_text(json.dumps({"time":(fixed-dt.timedelta(seconds=age)).isoformat(),"resume_allowed":resume}))
  (self.root/"jobs.json").write_text(json.dumps({"jobs":jobs or [],"submission_claims":claims or {}}))
  class Clock(dt.datetime):
   @classmethod
   def now(cls,tz=None):return fixed.astimezone(tz)
  with patch("datetime.datetime",Clock):self.fn(self.res)
 def test_admission_records_claim(self):
  self.at(9,24);self.assertEqual(json.loads((self.root/"jobs.json").read_text())["submission_claims"]["test-s1"]["gpus"],2)
 def test_blackout_boundaries(self):
  for h,m in [(9,25),(9,30),(9,39),(19,25),(19,39)]:
   with self.assertRaises(Refused):self.at(h,m)
  self.at(9,40);self.at(19,40)
 def test_stale_or_denied(self):
  with self.assertRaises(Refused):self.at(8,0,age=61)
  with self.assertRaises(Refused):self.at(8,0,resume=False)
 def test_active_and_pending_cap(self):
  with self.assertRaises(Refused):self.at(8,0,jobs=[{"status":"Running","run_id":"other"}],claims={"pending":{"state":"precreate"}})
  self.at(8,0,jobs=[{"status":"Stopped","run_id":"old"}],claims={"old":{"job_id":"dlcold"}})
 def test_exact_controller_binding(self):
  self.res["evidence"]["external_controller"]="/tmp/fake"
  with self.assertRaises(Refused):self.at(8,0)
if __name__=="__main__":unittest.main()
