"""Synthetic unit tests for the narrowly registered S1 idle 2-A800 profile."""
import copy,json,runpy,unittest
from pathlib import Path
R=Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/pai-job-registry")
P=runpy.run_path(str(R/"bin/pai-job"),run_name="s1_registry_test")
O=runpy.run_path(str(R/"bin/record-openapi-idle-source"),run_name="s1_openapi_test")
RES=json.loads((R/"config/resources.json").read_text())["resources"]
A="exp-robot-oversold-r16p14-stage2f-2gpu"
class Contract(unittest.TestCase):
 def fixture(self):
  r=copy.deepcopy(RES[A]);art=Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2f/pai/unit")
  return dict(run_id="s1-unit",resource_alias=A,resource=r,worker=r["target_worker_contract"],network={},
   storage={"data_sources":r["authorized_data_sources"]},runtime={"payload_sha256":"a"*64,"recursive_repair":False,"create_artifact_dir":True},
   fault_tolerance={"aimaster_args":P["EXPECTED_IDLE_A800_AIMASTER_ARGS"],"maximum_platform_restarts":50,
    "launcher_attempts":1,"pai_automatic_fault_tolerance":True},
   evidence={"kind":"r16p14_stage2f_formal_collection","idle_8gpu_contract":"generic_formal_quota_idle_2gpu_v1",
    "quota_idle_override":{"authorized":True,"pool":"exp-robot","resource_id":r["resource_id"],
     "resource_source_job_id":r["source_job_id"],"source_oversold_type":"ForbiddenQuotaOverSold","target_oversold_type":"AcceptQuotaOverSold"},
    "contract_ready":True,"workload_type":"collection","task_id":"s1","model_id":"frozen-act",
    "success_gate":"persisted_committed_sample_or_shard","require_actual_idle":True,"pai_probe_created":False,
    "validation_method":"static","validation_status":"passed","validated_payload_sha256":"a"*64,
    "expected_first_work_uid":2254,"expected_first_work_gid":2254,"first_work_evidence_path":str(art/"first.json")},
   submission={"disable_ecs_stock_check":True,"tags":{"purpose":"formal-collection","task":"s1","model":"frozen-act",
    "managed_by":"pai-job-registry","resource_pool":"exp-robot","hardware":"2xa800-idle"}})
 def test_valid_exact_profile(self):
  d=self.fixture()
  self.assertTrue(P["is_valid_efficiency_idle_override"](d))
  P["validate_resource_fault_tolerance_policy"](d)
  self.assertTrue(P["validate_disable_ecs_stock_check"](d))
  art=Path(d["evidence"]["first_work_evidence_path"]).parent
  P["validate_generic_formal_idle_8gpu_profile"](resolved=d,runtime=d["runtime"],write_paths=[str(art)],artifact_dir=art,output_mode="resume")
  self.assertTrue(O["is_registered_generic_efficiency_idle_8gpu"](d,RES))
 def test_worker_expansion_rejected(self):
  for key,value in [("gpu",8),("cpu",88),("memory","1525Gi"),("count",2)]:
   d=self.fixture();d["worker"][key]=value
   with self.assertRaises(P["Refused"]):P["validate_disable_ecs_stock_check"](d)
   self.assertFalse(O["is_registered_generic_efficiency_idle_8gpu"](d,RES))
 def test_no_unregistered_alias_or_cross_contract(self):
  for alias,contract in [(A,"generic_formal_quota_idle_8gpu_v1"),("exp-robot-oversold-other-2gpu","generic_formal_quota_idle_2gpu_v1"),
      ("exp-robot-oversold-other-8gpu","generic_formal_quota_idle_2gpu_v1")]:
   d=self.fixture();d["resource_alias"]=alias;d["evidence"]["idle_8gpu_contract"]=contract
   self.assertFalse(P["is_valid_efficiency_idle_override"](d))
 def test_dedicated_or_no_idle_rejected(self):
  d=self.fixture();d["resource"]["oversold_type"]="ForbiddenQuotaOverSold"
  self.assertFalse(P["is_valid_efficiency_idle_override"](d))
  d=self.fixture();d["evidence"]["require_actual_idle"]=False
  art=Path(d["evidence"]["first_work_evidence_path"]).parent
  with self.assertRaises(P["Refused"]):P["validate_generic_formal_idle_8gpu_profile"](resolved=d,runtime=d["runtime"],write_paths=[str(art)],artifact_dir=art,output_mode="resume")
if __name__=="__main__":unittest.main()
