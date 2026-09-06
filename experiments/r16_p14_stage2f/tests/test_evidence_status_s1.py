from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from s1.consolidate import _canonical_row, _validate_structural_exclusions, _event_key, TASKS
from s1.statistics import _error_status

def branch(**extra):
    return dict(event_instance_id="bowl-event", task=TASKS[1], init_state_id=10,
                split="calibration", generator_actor_seed=7, recovery_actor_seed=7,
                operator="fresh_h4", prefix_k=8, tail_horizon=4, action_budget=8,
                policy_call_cap=8, safe_success=False, pid=123, env_hash="env",
                chunk_hash="chunk", status="COMPLETE", **extra)

@pytest.mark.parametrize("extra", [
    {"error_type":"PrefixOutsideTaskHorizon"}, {"error_type":"UnexpectedRuntimeError"},
    {"structurally_excluded":True},
])
def test_completed_error_or_forged_exclusion_is_not_an_observation(extra):
    row,reasons=_canonical_row(branch(**extra),"phase1")
    assert row is None and reasons

@pytest.mark.parametrize("anchor",[-1,320,9999])
def test_out_of_range_anchor_cannot_prove_exclusion(anchor):
    raw=branch(error_type="PrefixOutsideTaskHorizon");raw["status"]="BLOCKED"
    row,reasons=_canonical_row(raw,"phase1")
    assert row and not reasons
    _validate_structural_exclusions([row],{_event_key(row):{"anchor_global_step":anchor,"task_horizon":320}},reasons)
    assert any("outside" in x for x in reasons)

def test_true_horizon_exclusion_is_retained():
    raw=branch(error_type="PrefixOutsideTaskHorizon");raw["status"]="BLOCKED"
    row,reasons=_canonical_row(raw,"phase1")
    _validate_structural_exclusions([row],{_event_key(row):{"anchor_global_step":316,"task_horizon":320}},reasons)
    assert not reasons and row["structurally_excluded"]

@pytest.mark.parametrize("status",["BLOCKED_BY_RUNTIME_ERROR","ERROR:crash","FAILED_WITH_OUTPUT"])
def test_statistics_failure_variants(status):
    assert _error_status(status)
