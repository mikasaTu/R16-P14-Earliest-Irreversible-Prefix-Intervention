from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .io_utils import atomic_write_json, atomic_write_jsonl, load_jsonl, sha256_json
from .settings import ACTOR_SEEDS, ARTIFACT_ROOT, EXPERIMENT_ROOT, MIRROR_EXPERIMENT_OUTPUTS, TASKS


def create_formal_event_pool() -> dict[str, Any]:
    qualification = json.loads(
        (ARTIFACT_ROOT / "perturbation_qualification/frozen_parameters.json").read_text()
    )
    events = load_jsonl(ARTIFACT_ROOT / "actor_events/events.jsonl")
    pool: list[dict[str, Any]] = []
    for split in ("calibration", "evaluation"):
        for task in TASKS:
            parameters = qualification["tasks"][task]["parameters"]
            choices = sorted(
                [event for event in events if event["split"] == split and event["task"] == task],
                key=lambda event: (
                    int(event["actor_seed"]),
                    int(event["init_state_id"]),
                    event["event_id"],
                ),
            )
            per_seed_index = Counter()
            for event in choices:
                seed = int(event["actor_seed"])
                parameter = parameters[per_seed_index[seed] % len(parameters)]
                per_seed_index[seed] += 1
                pool.append(
                    {
                        "event_instance_id": f"{event['event_id']}__{parameter['parameter_id']}",
                        "event_id": event["event_id"],
                        "source_event_sha256": sha256_json(event),
                        "source_checkpoint": event["checkpoint"],
                        "source_checkpoint_sha256": event["checkpoint_sha256"],
                        "source_original_chunk_sha256": event["original_chunk_hash"],
                        "source_is_actor_generated_chunk": bool(
                            event["source_is_actor_generated_chunk"]
                        ),
                        "source_is_demonstration_chunk": bool(
                            event["source_is_demonstration_chunk"]
                        ),
                        "task": task,
                        "split": split,
                        "actor_seed": seed,
                        "init_state_id": int(event["init_state_id"]),
                        "parameter": {
                            key: value
                            for key, value in parameter.items()
                            if key
                            not in {
                                "qualified",
                                "diagnostic_fallback",
                                "gate_distance",
                            }
                        },
                        "parameter_id": parameter["parameter_id"],
                        "qualified_parameter": bool(parameter["qualified"]),
                        "diagnostic_parameter_fallback": bool(parameter["diagnostic_fallback"]),
                    }
                )
    summary = {
        "schema_version": 1,
        "counts": {
            split: {
                task: {
                    "events": sum(item["split"] == split and item["task"] == task for item in pool),
                    "by_seed": {
                        str(seed): sum(
                            item["split"] == split
                            and item["task"] == task
                            and item["actor_seed"] == seed
                            for item in pool
                        )
                        for seed in ACTOR_SEEDS
                    },
                    "by_severity": dict(
                        Counter(
                            item["parameter_id"]
                            for item in pool
                            if item["split"] == split and item["task"] == task
                        )
                    ),
                    "distinct_init_ids": len(
                        {
                            item["init_state_id"]
                            for item in pool
                            if item["split"] == split and item["task"] == task
                        }
                    ),
                }
                for task in TASKS
            }
            for split in ("calibration", "evaluation")
        },
        "one_event_per_actor_rollout": True,
        "severity_assignment_rule": "alternate frozen severities independently within each actor seed",
        "evaluation_outcomes_read": False,
        "diagnostic_only": qualification["status"] != "PASS",
    }
    output = ARTIFACT_ROOT / "actor_events"
    atomic_write_jsonl(output / "formal_event_pool.jsonl", pool)
    atomic_write_json(output / "formal_event_pool_summary.json", summary)
    if MIRROR_EXPERIMENT_OUTPUTS:
        mirror = EXPERIMENT_ROOT / "actor_events"
        mirror.mkdir(parents=True, exist_ok=True)
        (mirror / "formal_event_pool_summary.json").write_bytes(
            (output / "formal_event_pool_summary.json").read_bytes()
        )
    print(json.dumps(summary, sort_keys=True))
    return summary


if __name__ == "__main__":
    create_formal_event_pool()
