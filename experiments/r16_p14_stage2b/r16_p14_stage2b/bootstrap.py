from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np


def percentile_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def paired_event_bootstrap(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    *,
    resamples: int,
    seed: int,
    stratum_key: str = "task",
) -> dict[str, Any]:
    if not rows:
        return {"estimate": None, "ci95": [None, None], "resamples": resamples, "event_count": 0}
    if all("event_id" in row for row in rows):
        event_ids = [str(row["event_id"]) for row in rows]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("bootstrap input must contain one paired row per independent event")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[stratum_key])].append(row)
    estimate = float(np.mean([np.mean([value(row) for row in group]) for group in groups.values()]))
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        stratum_means = []
        for group in groups.values():
            sampled = rng.integers(0, len(group), size=len(group))
            stratum_means.append(float(np.mean([value(group[item]) for item in sampled])))
        draws[index] = float(np.mean(stratum_means))
    lower, upper = percentile_interval(draws)
    return {
        "estimate": estimate,
        "ci95": [lower, upper],
        "resamples": resamples,
        "seed": seed,
        "event_count": len(rows),
        "strata": {name: len(group) for name, group in groups.items()},
        "unit": "event",
    }


def grouped_readout(rows: list[dict[str, Any]], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        record = {key: value for key, value in zip(keys, values)}
        record["event_count"] = len(group)
        for metric in metrics:
            record[metric] = float(np.mean([float(item[metric]) for item in group]))
        output.append(record)
    return output
