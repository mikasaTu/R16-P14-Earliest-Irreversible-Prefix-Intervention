from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest


EXPERIMENTS = Path(__file__).resolve().parents[2]
for _path in (
    EXPERIMENTS,
    EXPERIMENTS / "r16_p14_stage2a",
    EXPERIMENTS / "r16_p14_stage2b",
    EXPERIMENTS / "r16_p14_stage2c",
    EXPERIMENTS / "r16_p14_stage2d",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from r16_p14_stage2f.s1 import dispatch, measurement, runtime  # noqa: E402


TASK = "put_the_cream_cheese_in_the_bowl"


def _phase(*, lifted: bool, closed: bool, success: bool = False, release: bool = False) -> dict:
    return {
        "task": TASK,
        "task_success": success,
        "lifted_now": lifted,
        "ever_lifted": lifted,
        "gripper_closed": closed,
        "release_transition": release,
        "object_height_delta_m": 0.1 if lifted else 0.0,
    }


def _row(index: int, *, phase_name: str = "post_anchor", lifted: bool, closed: bool, release: bool = False, object_xy=(0.5, 0.5), target_xy=(0.0, 0.0), current_pairs=None) -> dict:
    action = [0.0] * 6 + ([-1.0] if release else [1.0])
    return {
        "step": index,
        "step_kind": "action_step",
        "phase": phase_name,
        "action": action,
        "previous_gripper": 1.0,
        "current_gripper": action[-1],
        "object_qpos": [object_xy[0], object_xy[1], 0.2 if lifted else 0.0],
        "target_qpos": [target_xy[0], target_xy[1], 0.0],
        "task_phase": _phase(lifted=lifted, closed=closed, release=release),
        "normalized_contact_pairs": current_pairs if current_pairs is not None else [["bowl_geom", "cream_geom"]],
        "baseline_contact_pairs": [["bowl_geom", "cream_geom"]],
        "geometry_names": {"object": ["cream_geom"], "target": ["bowl_geom"], "robot": ["robot_finger"]},
        "object_geometry_names": ["cream_geom"],
        "target_geometry_names": ["bowl_geom"],
        "robot_geometry_names": ["robot_finger"],
        "missing": [],
    }


def test_operator_contract_exact_and_task_horizon_clips_new_actions() -> None:
    assert runtime.OPERATORS == ("fresh_h4", "fresh_h16", "hold_1+fresh_h4", "rollback_1+fresh_h16")
    assert runtime.REFERENCE_OPERATORS == ("immediate_fresh", "fixed_delay_2", "fixed_delay_4", "fixed_delay_8")
    contract = runtime.execution_contract(TASK, "hold_1+fresh_h4", 8, 16, 32, 8, anchor_global_step=100)
    assert contract["configured_operator_horizon"] == 4
    assert contract["effective_execution_horizon"] == 4
    assert contract["effective_action_budget"] == 32
    clipped = runtime.execution_contract(TASK, "fresh_h16", 16, 16, 32, 8, anchor_global_step=350)
    assert clipped["task_horizon_remaining_after_prefix"] == 0
    assert clipped["effective_action_budget"] == 0
    assert clipped["effective_execution_horizon"] == 16
    assert clipped["task_horizon_budget_clipped"] is True
    with pytest.raises(ValueError):
        runtime.execution_contract(TASK, "fresh_h4", 2, 4, 8, 9)


def test_label_trace_ignores_early_release_and_starts_topology_at_anchor() -> None:
    rows = [
        _row(0, phase_name="pre_anchor", lifted=False, closed=False, release=True, object_xy=(0.0, 0.0)),
        _row(
            1,
            phase_name="pre_anchor",
            lifted=True,
            closed=True,
            current_pairs=[["bowl_geom", "cream_geom"], ["cream_geom", "table_geom"]],
        ),
        _row(
            2,
            lifted=True,
            closed=True,
            current_pairs=[["bowl_geom", "cream_geom"], ["cream_geom", "table_geom"]],
        ),
        _row(3, lifted=True, closed=False, release=True, object_xy=(0.5, 0.5), target_xy=(0.0, 0.0)),
    ]
    result = measurement.label_trace(rows, TASK)
    release = result["release_based"]
    candidate = result["prerelease_topology_candidate"]
    assert result["ignored_pre_lift_release_indices"] == [0]
    assert release["release_indices"] == [3]
    assert release["release_violation_indices"] == [3]
    assert result["label_complete"] is True
    assert candidate["first_index"] == 2
    assert candidate["ignored_pre_anchor_indices"] == [1]
    assert result["candidate_precedes_release"] is True
    assert candidate["missing"] == []
    assert result["time_order"] == "candidate_before_release"


class _FakeSim:
    def __init__(self, owner: "_FakeEnv") -> None:
        self.owner = owner
        self.calls = 0
        self.model = None

    def step(self) -> None:
        self.calls += 1
        self.owner.state += 1


class _FakeEnv:
    def __init__(self) -> None:
        self.state = 0
        self.qpos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.sim = _FakeSim(self)
        self.contacts = [("cream_geom", "bowl_geom"), ("bowl_geom", "cream_geom")]
        self._r16_object_geometry_names = ["cream_geom"]
        self._r16_target_geometry_names = ["bowl_geom"]
        self._r16_robot_geometry_names = ["robot_finger"]

    def check_success(self) -> bool:
        return False

    def step(self, action: np.ndarray):
        self.sim.step()
        self.qpos[2] += 0.1
        return {}, 0.0, False, {}


def test_record_step_is_json_safe_and_keeps_unique_and_raw_contacts(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _FakeEnv()
    monkeypatch.setattr(measurement, "joint_qpos", lambda _env, name: env.qpos.copy() if "cream" in name else np.zeros(3))
    row = measurement.record_step(
        env,
        TASK,
        event_instance_id="evt",
        prefix_k=2,
        operator="fresh_h4",
        actor_seed=7,
        step=0,
        action=np.arange(7, dtype=np.float32),
        previous_gripper=1.0,
        initial_object_z=0.0,
        ever_lifted=False,
        baseline_contacts=[["bowl_geom", "cream_geom"]],
    )
    json.dumps(row, allow_nan=False)
    assert row["normalized_contact_pairs"] == [["bowl_geom", "cream_geom"]]
    assert row["contact_pairs"] == row["normalized_contact_pairs"]
    assert len(row["raw_contact_pairs"]) == 2
    assert row["object_qpos"] == [0.0, 0.0, 0.0]
    assert row["target_qpos"] == [0.0, 0.0, 0.0]
    assert row["geometry_names"]["robot"] == ["robot_finger"]


def test_recorder_physics_probe_does_not_change_sim_and_writes_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(measurement, "joint_qpos", lambda env, name: env.qpos.copy())
    plain = _FakeEnv()
    measured = _FakeEnv()
    action = np.ones(7, dtype=np.float32)
    plain.step(action)
    trace_path = tmp_path / "trace.jsonl.gz"
    recorder = measurement.SimulationStepRecorder(
        measured,
        TASK,
        event_instance_id="evt",
        initial_object_z=0.0,
        trace_path=trace_path,
        baseline_contacts=[["bowl_geom", "cream_geom"]],
    )
    before = (measured.state, measured.sim.calls, measured.qpos.copy())
    recorder.set_action(0, action, 1.0, False)
    measured.step(action)
    recorder.finish_action()
    after_action = (measured.state, measured.sim.calls, measured.qpos.copy())
    digest = recorder.close()
    bytes_after_first_close = trace_path.read_bytes()
    assert recorder.physics_instrumented is True
    assert recorder.physics_step_count == 1
    assert len(recorder.physics_records) == 1
    assert len(recorder.action_records) == 1
    assert after_action[0:2] == (plain.state, plain.sim.calls)
    assert np.array_equal(after_action[2], plain.qpos)
    assert before[0:2] == (0, 0)
    assert np.array_equal(before[2], np.zeros(3))
    assert digest == recorder.close()
    assert trace_path.read_bytes() == bytes_after_first_close
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        trace_rows = [json.loads(line) for line in stream]
    assert [row["step_kind"] for row in trace_rows] == ["physics_step", "action_step"]
    assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == digest


def test_dispatch_fail_closed_contract_and_spawn_source() -> None:
    event = {"event_id": "e", "event_instance_id": "e"}
    row = dispatch._blocked_row(event, 7, "fresh_h4", "missing", repeat=0, trace_path=None)
    assert row["status"] == "BLOCKED_BY_RUNTIME_ERROR"
    assert row["safe_success"] is False
    assert row["env_hash"] is None
    assert "env_hash" in row["missing_provenance"]
    assert 'mp.get_context("spawn")' in inspect.getsource(dispatch.run_spawned_branch)
    assert "apply_perturbation" not in inspect.getsource(runtime)

class _FakeHistory:
    def __init__(self) -> None:
        self.states = [np.zeros(4, dtype=np.float32) for _ in range(4)]
        self.actions = [np.zeros(7, dtype=np.float32) for _ in range(3)]

    def state_array(self) -> np.ndarray:
        return np.asarray(self.states, dtype=np.float32)

    def action_array(self) -> np.ndarray:
        return np.asarray(self.actions, dtype=np.float32)

    def update(self, _observation: dict, action: np.ndarray) -> None:
        self.states = self.states[1:] + [np.zeros(4, dtype=np.float32)]
        self.actions = self.actions[1:] + [np.asarray(action, dtype=np.float32).copy()]


class _FakeRuntimeEnv(_FakeEnv):
    def get_sim_state(self) -> np.ndarray:
        return np.asarray([self.state, *self.qpos], dtype=np.float64)

    def close(self) -> None:
        return None


class _FakeBundle:
    loads: list[int] = []

    @classmethod
    def load(cls, seed: int, _device: str):
        cls.loads.append(int(seed))
        return cls()

    def predict(self, _states: np.ndarray, _actions: np.ndarray, _task: str) -> np.ndarray:
        return np.zeros((5, 7), dtype=np.float32)


def test_execute_branch_stub_enforces_recovery_budget_and_call_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    event = {
        "event_id": "stub-event",
        "event_instance_id": "stub-event",
        "task": TASK,
        "actor_seed": 7,
        "checkpoint_sha256": "generator-checkpoint",
        "init_state": [0.0],
        "init_state_hash": "init",
        "pre_anchor_actions": [],
        "pre_anchor_actions_hash": "pre",
        "anchor_state": [0.0],
        "anchor_state_hash": "anchor",
        "state_history": [[0.0]],
        "state_history_hash": "states",
        "action_history": [[0.0] * 7] * 3,
        "action_history_hash": "actions",
        "original_chunk": [[0.0] * 7] * 16,
        "original_chunk_hash": "chunk",
        "anchor_global_step": 0,
        "initial_manipulated_qpos": [0.0, 0.0, 0.0],
        "task_phase": {"stable_lift_two_steps": True},
        "anchor_contacts": [["bowl_geom", "cream_geom"]],
        "source_is_actor_generated_chunk": True,
        "source_is_demonstration_chunk": False,
        "global_step_fallback_used": False,
    }
    monkeypatch.setattr(runtime, "ActorBundle", _FakeBundle)
    monkeypatch.setattr(runtime, "_frozen_reconstruct_anchor", lambda _event, _bundle: (_FakeRuntimeEnv(), _FakeHistory(), {
        "anchor_state_exact": True,
        "state_history_exact": True,
        "action_history_exact": True,
        "event_chunk_exact": True,
        "actor_inference_side_effect_free": True,
        "max_anchor_state_error": 0.0,
    }))
    monkeypatch.setattr(runtime, "_frozen_branch_signature", lambda _env, _history, _event, _tracker, prefix: {
        "complete_signature_hash": "stub",
        "executed_prefix_length": len(prefix),
    })
    monkeypatch.setattr(measurement, "joint_qpos", lambda env, _name: env.qpos.copy())
    _FakeBundle.loads.clear()
    row = runtime.execute_branch(
        event,
        recovery_actor_seed=17,
        operator="hold_1+fresh_h4",
        prefix_k=2,
        tail_horizon=4,
        action_budget=3,
        policy_call_cap=1,
    )
    assert _FakeBundle.loads == [7, 17]
    assert row["actual_prefix_actions"] == 2
    assert row["actual_new_recovery_actions"] == 3
    assert row["actual_total_action_steps"] == 5
    assert row["validation_policy_calls"] == 1
    assert row["recovery_policy_calls"] == 1
    assert row["total_policy_calls"] == 2
    assert row["policy_call_cap_respected"] is True
    assert row["action_budget_respected"] is True
    assert row["physics_instrumented"] is True
    assert row["status"] == "OK"
    assert row["safe_success"] is False
