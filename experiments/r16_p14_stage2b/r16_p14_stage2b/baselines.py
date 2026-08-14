from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def validate_prefix_index(k: int, d: int, horizon: int) -> None:
    if not d <= k <= horizon:
        raise ValueError(f"prefix k={k} outside [{d}, {horizon}]")


def k_last_safe(branches: Iterable[dict[str, Any]]) -> int | None:
    safe = [int(branch["prefix_k"]) for branch in branches if branch.get("S_k") == 1]
    return max(safe) if safe else None


def k_best(branches: Iterable[dict[str, Any]]) -> int | None:
    candidates = [
        branch
        for branch in branches
        if branch.get("error") is None and int(branch.get("S_k", 0)) == 1
    ]
    if not candidates:
        return None
    winner = max(
        candidates,
        key=lambda branch: (
            int(bool(branch["safe_success"])),
            -int(branch["new_non_nominal_actions"]),
            -int(branch["policy_calls"]),
            -int(branch["prefix_k"]),
        ),
    )
    return int(winner["prefix_k"])


def velocity_phase_prefix(chunk: Any, d: int, horizon: int) -> int:
    actions = np.asarray(chunk, dtype=np.float32)
    speeds = np.linalg.norm(actions[:, :3], axis=1)
    for k in range(d + 1, horizon):
        index = k - 1
        if speeds[index] <= speeds[index - 1] and speeds[index] <= speeds[index + 1]:
            return k
    previous = float(actions[d - 1, -1])
    for k in range(d + 1, horizon + 1):
        current = float(actions[k - 1, -1])
        if (previous > 0.0) != (current > 0.0):
            return k
        previous = current
    return horizon


def disagreement_prefix(branches: Iterable[dict[str, Any]], threshold: float, horizon: int) -> int:
    for branch in sorted(branches, key=lambda item: int(item["prefix_k"])):
        value = branch.get("cached_fresh_normalized_disagreement")
        if branch.get("S_k") == 1 and value is not None and float(value) > threshold:
            return int(branch["prefix_k"])
    return horizon


def method_positions(
    branches: list[dict[str, Any]],
    *,
    chunk: Any,
    d: int,
    horizon: int,
    disagreement_threshold: float,
) -> dict[str, int | None]:
    available = {int(branch["prefix_k"]) for branch in branches}
    proposed = {
        "N0": horizon,
        "B0": d,
        "B1": d + 1,
        "B2": d + 2,
        "B3": d + 4,
        "B4": d + 8 if d + 8 <= horizon else None,
        "B5": velocity_phase_prefix(chunk, d, horizon),
        "B6": disagreement_prefix(branches, disagreement_threshold, horizon),
        "A_original": k_last_safe(branches),
        "N_oracle": k_best(branches),
    }
    return {
        name: (position if position is not None and position in available else None)
        for name, position in proposed.items()
    }


def strongest_baseline(method_rows: list[dict[str, Any]]) -> str | None:
    names = ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "A_original")
    candidates = []
    for name in names:
        rows = [row for row in method_rows if row["method"] == name and row.get("available")]
        if not rows:
            continue
        candidates.append(
            (
                float(np.mean([row["safe_success"] for row in rows])),
                -float(np.mean([row["cause_violation"] for row in rows])),
                -float(np.mean([row["new_non_nominal_actions"] for row in rows])),
                -float(np.mean([row["policy_calls"] for row in rows])),
                -names.index(name),
                name,
            )
        )
    return max(candidates)[-1] if candidates else None
