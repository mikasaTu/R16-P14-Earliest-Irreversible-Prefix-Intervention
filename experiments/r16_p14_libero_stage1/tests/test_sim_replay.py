from __future__ import annotations

import hashlib

import h5py
import numpy as np

from r16_p14_stage1.envs import contact_pairs, make_env, restore_state
from r16_p14_stage1.settings import TASK_SPECS


def test_snapshot_suffix_replay_is_deterministic() -> None:
    spec = TASK_SPECS["open_the_middle_drawer_of_the_cabinet"]
    with h5py.File(spec.hdf5_path, "r") as dataset:
        demo = dataset["data/demo_0"]
        init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
        actions = np.asarray(demo["actions"], dtype=np.float32)
    env, _ = make_env(spec.name, horizon=1000)
    try:
        restore_state(env, init_state)
        for action in actions[:50]:
            env.step(action)
        snapshot = env.get_sim_state().copy()
        outputs = []
        for _ in range(3):
            restore_state(env, snapshot)
            contacts = []
            for action in actions[50:66]:
                env.step(action)
                contacts.append(contact_pairs(env))
            outputs.append(
                (
                    env.get_sim_state().copy(),
                    bool(env.check_success()),
                    hashlib.sha256(repr(contacts).encode("utf-8")).hexdigest(),
                )
            )
        for output in outputs[1:]:
            assert np.max(np.abs(outputs[0][0] - output[0])) < 1e-9
            assert outputs[0][1:] == output[1:]
    finally:
        env.close()
