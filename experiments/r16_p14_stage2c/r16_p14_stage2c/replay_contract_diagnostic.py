from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from r16_p14_stage2b.io_utils import atomic_write_json, atomic_write_text, load_jsonl

from .settings import ARTIFACT_ROOT, PREFIX_INDICES


def _group(rows: Iterable[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def _sequence_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["prefix_k"]))
    indices = [int(row["prefix_k"]) for row in ordered]
    values = [int(bool(row["S_obs_at_k"])) for row in ordered]
    transitions = [
        {"from_k": indices[index - 1], "to_k": indices[index]}
        for index in range(1, len(values))
        if values[index - 1] == 0 and values[index] == 1
    ]
    return {
        "complete_prefix_grid": indices == list(PREFIX_INDICES),
        "zero_to_one_transitions": transitions,
        "monotone": indices == list(PREFIX_INDICES) and not transitions,
    }


def _breakdown(event_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task": dict(sorted(Counter(str(row["task"]) for row in event_rows).items())),
        "split": dict(sorted(Counter(str(row["split"]) for row in event_rows).items())),
        "task_split": dict(
            sorted(Counter(f"{row['task']}/{row['split']}" for row in event_rows).items())
        ),
        "generator_actor_seed": {
            str(key): value
            for key, value in sorted(
                Counter(int(row["generator_actor_seed"]) for row in event_rows).items()
            )
        },
        "parameter_id": dict(
            sorted(Counter(str(row["parameter_id"]) for row in event_rows).items())
        ),
    }


def build_replay_contract_audit(
    matched: list[dict[str, Any]], recovery: list[dict[str, Any]]
) -> dict[str, Any]:
    recovery_cells = _group(recovery, "event_instance_id", "prefix_k")
    inconsistent_recovery_cells = []
    for (event_instance_id, prefix_k), rows in sorted(recovery_cells.items()):
        values = sorted({bool(row["S_obs_at_k"]) for row in rows})
        if len(values) != 1:
            first = rows[0]
            inconsistent_recovery_cells.append(
                {
                    "event_instance_id": event_instance_id,
                    "task": first["task"],
                    "split": first["split"],
                    "init_state_id": int(first["init_state_id"]),
                    "generator_actor_seed": int(first["generator_actor_seed"]),
                    "parameter_id": first["parameter_id"],
                    "prefix_k": int(prefix_k),
                    "S_obs_values": values,
                    "row_count": len(rows),
                }
            )

    inconsistent_event_ids = {
        row["event_instance_id"] for row in inconsistent_recovery_cells
    }
    event_metadata = {}
    for row in recovery:
        event_metadata.setdefault(
            row["event_instance_id"],
            {
                "event_instance_id": row["event_instance_id"],
                "task": row["task"],
                "split": row["split"],
                "init_state_id": int(row["init_state_id"]),
                "generator_actor_seed": int(row["generator_actor_seed"]),
                "parameter_id": row["parameter_id"],
            },
        )
    inconsistent_events = [event_metadata[key] for key in sorted(inconsistent_event_ids)]

    nominal_pair_cells = _group(
        [
            row
            for row in matched
            if row["branch"] in {"CACHED_MATCHED", "CACHED_NOQUERY"}
        ],
        "event_instance_id",
        "prefix_k",
    )
    nominal_pair_disagreements = []
    for (event_instance_id, prefix_k), rows in sorted(nominal_pair_cells.items()):
        by_branch = {row["branch"]: row for row in rows}
        if set(by_branch) != {"CACHED_MATCHED", "CACHED_NOQUERY"}:
            continue
        left = by_branch["CACHED_MATCHED"]
        right = by_branch["CACHED_NOQUERY"]
        if bool(left["S_obs_at_k"]) != bool(right["S_obs_at_k"]):
            nominal_pair_disagreements.append(
                {
                    "event_instance_id": event_instance_id,
                    "task": left["task"],
                    "split": left["split"],
                    "init_state_id": int(left["init_state_id"]),
                    "generator_actor_seed": int(left["generator_actor_seed"]),
                    "parameter_id": left["parameter_id"],
                    "prefix_k": int(prefix_k),
                    "cached_matched_S_obs": bool(left["S_obs_at_k"]),
                    "cached_noquery_S_obs": bool(right["S_obs_at_k"]),
                }
            )

    matched_sequences = _group(
        [
            row
            for row in matched
            if row["branch"] in {"CACHED_MATCHED", "CACHED_NOQUERY"}
        ],
        "event_instance_id",
        "branch",
    )
    matched_nonmonotone = [
        {
            "event_instance_id": key[0],
            "branch": key[1],
            **_sequence_audit(rows),
        }
        for key, rows in sorted(matched_sequences.items())
        if not _sequence_audit(rows)["monotone"]
    ]

    recovery_sequences = _group(
        recovery,
        "event_instance_id",
        "recovery_actor_seed",
        "operator",
    )
    recovery_nonmonotone = [
        {
            "event_instance_id": key[0],
            "recovery_actor_seed": int(key[1]),
            "operator": key[2],
            **_sequence_audit(rows),
        }
        for key, rows in sorted(recovery_sequences.items())
        if not _sequence_audit(rows)["monotone"]
    ]

    expected_recovery_cells = len(event_metadata) * len(PREFIX_INDICES)
    expected_matched_cells = len(event_metadata) * len(PREFIX_INDICES) * 4
    formal_execution_complete = (
        len(recovery_cells) == expected_recovery_cells
        and all(len(rows) == 12 for rows in recovery_cells.values())
        and len(matched) == expected_matched_cells
    )
    blocked = bool(
        inconsistent_recovery_cells
        or matched_nonmonotone
        or recovery_nonmonotone
    )
    return {
        "schema_version": 1,
        "status": "BLOCKED" if blocked else "PASS",
        "classification": (
            "SHARED_RUNTIME_RESET_INCOMPLETE_OR_ORDER_DEPENDENT" if blocked else None
        ),
        "formal_execution_complete": formal_execution_complete,
        "recovery_prefix_cells": {
            "total": len(recovery_cells),
            "expected_rows_per_cell": 12,
            "all_have_expected_rows": all(len(rows) == 12 for rows in recovery_cells.values()),
            "inconsistent_S_obs_cells": len(inconsistent_recovery_cells),
            "inconsistent_S_obs_fraction": (
                len(inconsistent_recovery_cells) / len(recovery_cells) if recovery_cells else None
            ),
        },
        "events": {
            "total": len(event_metadata),
            "with_inconsistent_recovery_S_obs": len(inconsistent_events),
            "inconsistent_fraction": (
                len(inconsistent_events) / len(event_metadata) if event_metadata else None
            ),
            "breakdown": _breakdown(inconsistent_events),
        },
        "same_nominal_prefix_control": {
            "pair_cells": len(nominal_pair_cells),
            "S_obs_disagreement_cells": len(nominal_pair_disagreements),
            "S_obs_disagreement_fraction": (
                len(nominal_pair_disagreements) / len(nominal_pair_cells)
                if nominal_pair_cells
                else None
            ),
        },
        "monotonicity": {
            "matched_nominal_sequences": len(matched_sequences),
            "matched_nominal_nonmonotone_sequences": len(matched_nonmonotone),
            "recovery_sequences": len(recovery_sequences),
            "recovery_nonmonotone_sequences": len(recovery_nonmonotone),
        },
        "code_localization": {
            "frozen_prefix": "All recovery rows execute the same immutable A[d:k] before any recovery actor or operator action.",
            "recording_order": "prefix_cause_violation and S_obs_at_k are captured before bundle.predict and before operator-specific prelude/tail execution.",
            "runtime_lifecycle": "run_event constructs one EventRuntime/environment per event and reuses it for every ordered branch; EventRuntime.reset restores only enumerated simulator/controller/RNG state instead of reconstructing a fresh environment.",
            "eliminated_explanations": [
                "recovery actor identity",
                "recovery operator action",
                "tail policy outcome",
                "missing formal rows",
            ],
            "remaining_boundary": "The raw evidence localizes the failure to residual mutable state or order dependence in reset/detection/injection/prefix rollout. It does not uniquely identify the hidden simulator, controller, or observable-cache field.",
        },
        "interpretation": {
            "invalid_event_reason": "INCOMPLETE_PREFIX_GRID is an aggregation consequence of mutually inconsistent S_obs values, not missing execution rows.",
            "downstream_scope": "Raw arm outcomes and all planned statistics are retained, but cached/fresh gains, recoverability boundaries, and cross-fitted selections are not valid causal or deployable evidence under this failed replay contract.",
            "repair_applied": False,
            "new_idea_generated": False,
        },
        "inconsistent_recovery_cells": inconsistent_recovery_cells,
        "nominal_pair_disagreements": nominal_pair_disagreements,
        "matched_nonmonotone_sequences": matched_nonmonotone,
        "recovery_nonmonotone_sequences": recovery_nonmonotone,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    cells = payload["recovery_prefix_cells"]
    events = payload["events"]
    control = payload["same_nominal_prefix_control"]
    mono = payload["monotonicity"]
    return "\n".join(
        [
            "# Stage-2C replay-contract diagnostic",
            "",
            f"Status: **{payload['status']}** (`{payload['classification']}`).",
            "",
            f"All {events['total']} formal events and all planned rows completed. Nevertheless, {cells['inconsistent_S_obs_cells']}/{cells['total']} recovery prefix cells returned both true and false for `S_obs(k)`, affecting {events['with_inconsistent_recovery_S_obs']}/{events['total']} events.",
            f"The same-action `CACHED_MATCHED`/`CACHED_NOQUERY` control disagreed on pre-tail `S_obs(k)` in {control['S_obs_disagreement_cells']}/{control['pair_cells']} cells. Independently, {mono['matched_nominal_nonmonotone_sequences']}/{mono['matched_nominal_sequences']} matched nominal sequences and {mono['recovery_nonmonotone_sequences']}/{mono['recovery_sequences']} recovery sequences contained a 0→1 transition.",
            "",
            "The recovery code records `prefix_cause_violation` before actor inference and before operator-specific actions. Those rows also execute the same frozen cached prefix. Recovery actor, operator and tail outcome therefore cannot explain the disagreement.",
            "",
            "The formal worker reuses one mutable LIBERO environment per event and restores an enumerated state snapshot between branches. The evidence localizes the failure to residual reset/order-dependent runtime state, but does not identify one hidden field strongly enough to claim a narrower root cause.",
            "",
            "No raw row was changed or discarded. `INCOMPLETE_PREFIX_GRID` is the fail-closed aggregation label created when contradictory `S_obs(k)` values are set to null; it is not a missing-execution label. All downstream gains/losses remain descriptive only, and no new idea is introduced.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    matched = load_jsonl(root / "formal_matrix/matched_prefix_rows.jsonl")
    recovery = load_jsonl(root / "formal_matrix/recovery_operator_rows.jsonl")
    payload = build_replay_contract_audit(matched, recovery)
    atomic_write_json(root / "replay_contract_diagnostic.json", payload)
    atomic_write_text(root / "replay_contract_diagnostic.md", render_markdown(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "classification": payload["classification"],
                "output": str(root / "replay_contract_diagnostic.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
