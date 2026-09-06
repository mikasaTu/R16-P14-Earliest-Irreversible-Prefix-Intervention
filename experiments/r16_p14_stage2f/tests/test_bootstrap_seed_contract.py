from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from s1 import statistics  # noqa: E402


def _rows():
    rows = []
    for task in ("task_a", "task_b"):
        for event in range(2):
            for prefix in statistics.PHASE2_PREFIXES:
                for operator in statistics.OPERATORS:
                    bound = 16 if operator in statistics.FAMILY_A else 8
                    for recovery_seed in (7, 17, 29):
                        rows.append(
                            {
                                "event_instance_id": f"{task}-event-{event}",
                                "task": task,
                                "init_state_id": f"{task}-init-{event}",
                                "split": "evaluation",
                                "generator_actor_seed": 7,
                                "recovery_actor_seed": recovery_seed,
                                "operator": operator,
                                "prefix_k": prefix,
                                "tail_horizon": 4,
                                "action_budget": 8,
                                "policy_call_cap": 8,
                                "safe_success": float(prefix <= bound),
                                "pid": 1000 + event,
                                "env_hash": "env",
                                "chunk_hash": "chunk",
                                "status": "ok",
                            }
                        )
    return rows


def _reported_bootstrap_seeds(value):
    found = []

    def visit(node):
        if isinstance(node, dict):
            for key, item in node.items():
                if key in {"seed", "bootstrap_seed"}:
                    found.append(item)
                visit(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item)

    visit(value)
    return found


def test_every_bootstrap_rng_uses_declared_seed_and_reports_it(monkeypatch):
    declared_seed = 123456
    calls = []
    original = statistics.np.random.default_rng

    def recording_default_rng(seed=None):
        calls.append(seed)
        return original(seed)

    monkeypatch.setattr(statistics.np.random, "default_rng", recording_default_rng)
    first = statistics.analyze_crossing(_rows(), replicates=5, seed=declared_seed)
    second = statistics.analyze_crossing(_rows(), replicates=5, seed=declared_seed)

    # There are 3 primary scopes and 9 per-family split-half scopes; each
    # scope has one crossing and one Spearman bootstrap. Two identical
    # calls are made to check reproducibility. The three partitions per
    # family are retained without adding reverse-direction streams.
    assert len(calls) == 2 * 2 * (3 + 9 + 9)
    assert set(calls) == {declared_seed}
    assert set(_reported_bootstrap_seeds(first)) == {declared_seed}
    assert first == second
    assert len(first["split_half_null"]["family_A"]["partitions"]) == 3
    assert len(first["split_half_null"]["family_B"]["partitions"]) == 3
    assert len(first["split_half_null"]["by_task"]["task_a"]["bootstrap_null_draws"]) == 6 * 5


def test_custom_seed_is_not_rewritten_by_nested_scope_offsets(monkeypatch):
    declared_seed = 216214
    calls = []
    original = statistics.np.random.default_rng

    def recording_default_rng(seed=None):
        calls.append(seed)
        return original(seed)

    monkeypatch.setattr(statistics.np.random, "default_rng", recording_default_rng)
    result = statistics.analyze_crossing(_rows(), replicates=2, seed=declared_seed)

    assert calls
    assert all(seed == declared_seed for seed in calls)
    assert result["bootstrap_seed"] == declared_seed
    assert result["crossing"]["bootstrap_seed"] == declared_seed
    assert result["split_half_null"]["bootstrap_seed"] == declared_seed
    assert result["split_half_null"]["family_A"]["bootstrap_seed"] == declared_seed
    assert result["split_half_null"]["family_B"]["bootstrap_seed"] == declared_seed
    for task_metrics in result["crossing"]["by_task"].values():
        assert task_metrics["bootstrap_seed"] == declared_seed
        assert task_metrics["bootstrap"]["seed"] == declared_seed
        assert task_metrics["spearman_bootstrap"]["seed"] == declared_seed


def test_family_success_estimand_describes_actual_aggregation():
    text = statistics._family_success(
        [
            {
                "task": "task_a",
                "init_state_id": "init_a",
                "event_instance_id": "event_a",
                "values": {
                    2: {
                        operator: {"7": 1.0, "17": 0.0, "29": 0.0}
                        for operator in statistics.FAMILY_A
                    },
                    4: {
                        operator: {"7": 0.0, "17": 1.0, "29": 1.0}
                        for operator in statistics.FAMILY_A
                    },
                },
            }
        ],
        statistics.FAMILY_A,
    )
    estimand = text["by_task"]["task_a"]["estimand"]
    assert "per-prefix max operator" in estimand
    assert "equal cluster weight" in estimand
    assert "after actor aggregation" in estimand
