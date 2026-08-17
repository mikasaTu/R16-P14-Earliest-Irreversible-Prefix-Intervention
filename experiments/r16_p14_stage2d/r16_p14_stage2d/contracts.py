from __future__ import annotations

from typing import Iterable

from .settings import DETECTION_PREFIX, H_VALID, TAIL_HORIZON


def has_zero_to_one(values: Iterable[bool]) -> bool:
    sequence = list(values)
    return any(not left and right for left, right in zip(sequence, sequence[1:]))


def prefix_length(prefix_k: int) -> int:
    if not DETECTION_PREFIX <= prefix_k <= H_VALID:
        raise ValueError(prefix_k)
    return prefix_k - DETECTION_PREFIX


def required_detection_calls(arm: str) -> int:
    return int(
        arm
        in {
            "CACHED_MATCHED",
            "FRESH_MATCHED",
            "HOLD_MATCHED",
            "EVENT_ALIGNED_CACHED",
            "FRESH_MATCHED_AT_RULE_K",
            "HOLD_MATCHED_AT_RULE_K",
        }
    )


def maximum_budgets() -> dict[str, int]:
    return {
        "tail_horizon": TAIL_HORIZON,
        "maximum_action_budget": H_VALID - DETECTION_PREFIX + TAIL_HORIZON,
        "maximum_policy_call_budget": 2,
    }


def outcome_label_allowed(upstream_pass: bool, diagnostic_only: bool) -> bool:
    return bool(upstream_pass and not diagnostic_only)


def splits_disjoint(*splits: Iterable[int]) -> bool:
    materialized = [set(split) for split in splits]
    return all(
        materialized[left].isdisjoint(materialized[right])
        for left in range(len(materialized))
        for right in range(left + 1, len(materialized))
    )
