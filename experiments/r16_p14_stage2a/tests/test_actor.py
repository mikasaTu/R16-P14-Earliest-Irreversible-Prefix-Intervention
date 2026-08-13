from __future__ import annotations

import json

import numpy as np
import torch

from r16_p14_stage2a.actor_data import Normalization
from r16_p14_stage2a.actor_mechanism_audit import condition_inputs
from r16_p14_stage2a.actor_model import (
    ACTConfig,
    HistoryConditionedStateACT,
    predict_chunk,
)


def test_numpy_phase_flags_are_explicitly_serializable() -> None:
    promoted = False
    promoted |= np.float32(1.0) > 0.0
    assert isinstance(promoted, np.bool_)
    payload = {"phase_reached": bool(promoted)}
    assert json.loads(json.dumps(payload)) == {"phase_reached": True}


def test_mechanism_audit_changes_one_input_factor() -> None:
    states = np.arange(2 * 4 * 95, dtype=np.float32).reshape(2, 4, 95)
    histories = np.ones((2, 3, 7), dtype=np.float32)
    task_ids = np.asarray([0, 5], dtype=np.int64)
    repeated, retained_history, retained_task = condition_inputs(
        states, histories, task_ids, "repeat_current_state"
    )
    assert np.all(repeated == states[:, -1:, :])
    assert np.array_equal(retained_history, histories)
    assert np.array_equal(retained_task, task_ids)
    retained_states, zeroed, retained_task = condition_inputs(
        states, histories, task_ids, "zero_action_history"
    )
    assert np.array_equal(retained_states, states)
    assert np.all(zeroed == 0.0)
    assert np.array_equal(retained_task, task_ids)
    retained_states, retained_history, cycled = condition_inputs(
        states, histories, task_ids, "cyclic_task_id"
    )
    assert np.array_equal(retained_states, states)
    assert np.array_equal(retained_history, histories)
    assert np.array_equal(cycled, np.asarray([1, 0]))


def test_actor_shape_and_parameter_budget() -> None:
    model = HistoryConditionedStateACT()
    output = model(
        torch.zeros(2, 4, 95),
        torch.zeros(2, 3, 7),
        torch.tensor([0, 5]),
    )
    assert output.shape == (2, 16, 7)
    assert model.parameter_count() == 3_146_503
    assert model.parameter_count() <= 10_000_000


def test_actor_has_no_rgb_or_privileged_input() -> None:
    config = ACTConfig()
    assert config.state_dim == 95
    assert config.state_history == 4
    assert config.action_history == 3
    assert config.chunk_length == 16


def test_predict_chunk_clips_and_binarizes_gripper() -> None:
    model = HistoryConditionedStateACT(ACTConfig(dropout=0.0)).eval()
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    normalization = Normalization(
        state_mean=np.zeros(95, dtype=np.float32),
        state_std=np.ones(95, dtype=np.float32),
        action_mean=np.zeros(7, dtype=np.float32),
        action_std=np.ones(7, dtype=np.float32),
    )
    chunk = predict_chunk(
        model,
        normalization,
        np.zeros((4, 95), dtype=np.float32),
        np.zeros((3, 7), dtype=np.float32),
        0,
        torch.device("cpu"),
    )
    assert chunk.shape == (16, 7)
    assert np.all(chunk[:, -1] == 1.0)
    assert np.max(np.abs(chunk)) <= 1.0
