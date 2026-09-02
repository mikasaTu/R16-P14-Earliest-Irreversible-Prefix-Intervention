#!/usr/bin/env python3
"""CPU-only Stage-2E/S0 offline reanalysis.

This module deliberately contains no simulator, model, GPU, or PAI imports.
It reads the frozen Stage-2C/2D evidence, validates the declared contracts,
and produces the diagnostic-only A/B/C/G0 outputs.  Stage-2D's official
``load_jsonl`` is used for the sharded calibration atlas; no local shard
ordering is inferred here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGE2E_EXP = ROOT / "experiments" / "r16_p14_stage2e"
STAGE2E_ART = ROOT / "artifacts" / "stage2e"
STAGE2C_ART = ROOT / "artifacts" / "stage2c"
STAGE2D_ART = ROOT / "artifacts" / "stage2d"
STAGE2D_CODE = ROOT / "experiments" / "r16_p14_stage2d"
IMMUTABLE_PARENT_COMMIT = "edebdfc64576129d994535dacb76de930f493c8d"
IMMUTABLE_PARENT_TREE = "fa0a951abc1aa778cfa76663679173557e6c9b96"
IMMUTABLE_OLD_PATHS = (
    "artifacts/stage2c", "artifacts/stage2d",
    "experiments/r16_p14_stage2a", "experiments/r16_p14_stage2b",
    "experiments/r16_p14_stage2c", "experiments/r16_p14_stage2d",
    "docs/feishu",
)

# The Stage-2D loader is the frozen project loader.  Importing this small
# utility imports only stdlib/numpy; it does not construct an environment or
# load an actor.  Failure is a B-task prerequisite failure, not a fallback to
# a hand-written shard globber.
if str(STAGE2D_CODE) not in sys.path:
    sys.path.insert(0, str(STAGE2D_CODE))
try:
    from r16_p14_stage2d.io_utils import load_jsonl as official_load_jsonl
except Exception as exc:  # pragma: no cover - exercised only on bad installs
    official_load_jsonl = None
    _OFFICIAL_LOADER_IMPORT_ERROR = repr(exc)
else:
    _OFFICIAL_LOADER_IMPORT_ERROR = None


DIAGNOSTIC = True
FORMAL_POSITIVE = False
NEW_IDEA = False
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 216214
DETECTION_PREFIX = 2
H_VALID = 16
PREFIXES = tuple(range(DETECTION_PREFIX, H_VALID + 1))
ACTOR_SEEDS = (7, 17, 29)
OPERATORS = (
    "fresh_h4",
    "fresh_h16",
    "hold_one_step_then_fresh_h16",
    "rollback_one_step_then_fresh_h16",
)
BASELINE_METHODS = (
    "action_disagreement",
    "fixed_delay_1",
    "fixed_delay_2",
    "fixed_delay_4",
    "fixed_delay_8",
    "immediate_fresh_h16",
    "k_last_observed_safe",
    "k_last_recoverable",
    "velocity_phase",
)
EXPECTED_A1 = {
    "action_disagreement": (10, 144, 48),
    "fixed_delay_1": (8, 144, 48),
    "fixed_delay_2": (13, 144, 48),
    "fixed_delay_4": (15, 144, 48),
    "fixed_delay_8": (16, 144, 48),
    "immediate_fresh_h16": (23, 144, 48),
    "k_last_observed_safe": (3, 72, 24),
    "k_last_recoverable": (20, 54, 18),
    "velocity_phase": (16, 144, 48),
}
EXPECTED_HASHES = {
    "c_recovery": "21bd70d4dd258dd4e4b52a453ca137b2552800a40654bb03db0bbdb344e47efa",
    "c_atlas": "2eb4f0482b4363e5e562119466f48d92682b090602be32cc2e6b87574e37c5fe",
    "c_boundaries": "d2c93d6027419a850220b1c1fc644579d4f1387633099c34129a9643305c7ff9",
    "c_invalid": "eae0ce6f2804a3628c46bdd55d1f5abcae6b16dab35f8e194914f9fd9ec02f13",
    "c_baseline": "2d212f4a7f0a4062866feed39482f67a9622cd2b8aaaabff332874552d3a7626",
    "c_crossfit": "a6b6bc82748516acb02fafd577a4be0fa60eae2e1afa7b87c8557525380db6a5",
    "d_rows_pointer": "490c715119ae22b24980db15fd3611e31879e641ee8c877b0e83bf55611eab71",
    "d_rows_part_000": "5f6b819a423053fdbf3dc50d59e4bf311bf7a4d952043bf4420cb7bdf0ee89e7",
    "d_rows_part_001": "8479815df89a7f7f5776a455b86ea3fd02afa6f605472dbd102a058e088917c5",
    "d_rows_part_002": "0575ea50cc2332204417e20afbb2ca8a14b31df94b6233e50761f8b4fa91600b",
    "d_rows_part_003": "4ee2386d8d5e5b8bfb086677c75ea62b94c10c9dcba74cb34909e2f7084de7ef",
    "d_baselines": "cc8bf0e78946b8430037e9df4562a3ed14495bf86eca12b32f9fc497eecb87e9",
    "d_summary": "377825d0a16f10c35c9d7608e2df28e68b99cbcc07a7426faaea512dbd8cf8c4",
    "d_method": "3e518cbc8460276393a6084fe140a98004a5d531d41f4ed93b1607b33c928fff",
    "d_paired": "6951c6570d8f0002d4e249ef4482fd064f73f33a0942d0334e084e141b79dc0e",
}
INPUT_PATHS = {
    "c_recovery": "artifacts/stage2c/formal_matrix/recovery_operator_rows.jsonl",
    "c_atlas": "artifacts/stage2c/recoverability_atlas/rows.jsonl",
    "c_boundaries": "artifacts/stage2c/recoverability_atlas/event_boundaries.jsonl",
    "c_invalid": "artifacts/stage2c/recoverability_atlas/invalid_events.jsonl",
    "c_baseline": "artifacts/stage2c/crossfit_replanability/baseline_rows.jsonl",
    "c_crossfit": "artifacts/stage2c/crossfit_replanability/crossfit_rows.jsonl",
    "d_rows_pointer": "artifacts/stage2d/calibration_atlas/rows.jsonl",
    "d_baselines": "artifacts/stage2d/frozen_rule/baselines.json",
    "d_summary": "artifacts/stage2d/calibration_atlas/summary.json",
    "d_method": "artifacts/stage2d/confirmatory_evaluation/method_rows.jsonl",
    "d_paired": "artifacts/stage2d/confirmatory_evaluation/paired_rows.jsonl",
}
EXPECTED_ROWS = {
    "c_recovery": 17280,
    "c_atlas": 1440,
    "c_boundaries": 96,
    "c_invalid": 33,
    "c_baseline": 2289,
    "c_crossfit": 117,
    "d_rows_pointer": 10620,
    "d_method": 1386,
    "d_paired": 154,
}
REQUIRED_KEYS = {
    "c_recovery": {
        "S_obs_at_k", "actor_calls", "allocated_actor_call_budget", "branch",
        "cause_violation", "cause_violation_type", "completion_steps",
        "error", "event_id", "event_instance_id", "generator_actor_seed",
        "init_state_id", "new_non_nominal_actions", "operator", "parameter_id",
        "prefix_cause_violation", "prefix_k", "recovery_actor_seed",
        "safe_success", "split", "tail_execution_horizon", "task",
        "task_success", "total_post_detection_action_budget",
    },
    "c_atlas": {"event_instance_id", "event_id", "operators", "prefix_k", "split", "task", "R_U"},
    "c_boundaries": {"event_instance_id", "event_id", "k_last_observed_safe", "k_last_recoverable", "k_irrev_U", "prefix_safety_valid", "split", "task"},
    "c_invalid": {"event_instance_id", "reason"},
    "c_baseline": {"event_instance_id", "heldout_actor_seed", "method", "prefix_k", "safe_success", "split", "task", "init_state_id"},
    "c_crossfit": {"event_instance_id", "heldout_actor_seed", "selected_prefix", "split", "task"},
    "d_rows_pointer": {"S_obs_at_k", "actor_seed", "actual_actor_calls", "actual_post_detection_actions", "cause_violation", "error", "event_instance_id", "effective_arm", "prefix_k", "requested_prefix_k", "maximum_action_budget", "maximum_policy_call_budget", "requested_arm", "safe_success", "split", "tail_horizon", "task", "unique_process_contract"},
    "d_method": {"actor_seed", "actual_actor_calls", "actual_post_detection_actions", "cause_violation", "error", "event_instance_id", "requested_arm", "prefix_k", "safe_success", "split", "tail_horizon", "task", "unique_process_contract"},
    "d_paired": {"event_instance_id", "actor_seed", "methods", "rule_k", "task", "replay_admitted_3_of_3"},
}


class ContractError(RuntimeError):
    """A frozen-input contract failure; never silently repaired."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> Any:
        raise ContractError(f"non-finite JSON constant {value}")

    return json.loads(text, parse_constant=reject_constant)


def assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"non-finite value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")


def load_plain_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = strict_json_loads(line)
            if not isinstance(item, dict):
                raise ContractError(f"{path}: line {line_no} is not an object")
            assert_finite(item, f"line{line_no}")
            rows.append(item)
    return rows


def load_object(path: Path) -> Any:
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    assert_finite(value)
    return value


def key_unique(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], label: str) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if key in seen:
            raise ContractError(f"{label}: duplicate key {fields}={key}")
        seen.add(key)


def validate_rows(name: str, rows: list[dict[str, Any]]) -> None:
    if name in EXPECTED_ROWS and len(rows) != EXPECTED_ROWS[name]:
        raise ContractError(f"{name}: row count {len(rows)} != {EXPECTED_ROWS[name]}")
    required = REQUIRED_KEYS.get(name, set())
    for index, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            raise ContractError(f"{name}: row {index} missing {sorted(missing)}")
        if not isinstance(row, dict):
            raise ContractError(f"{name}: row {index} not object")
        assert_finite(row, f"{name}[{index}]")
    if name == "c_recovery":
        key_unique(rows, ("event_instance_id", "prefix_k", "operator", "recovery_actor_seed"), name)
        events = {row["event_instance_id"] for row in rows}
        if len(events) != 96:
            raise ContractError(f"{name}: events={len(events)} != 96")
        counts = Counter(row["event_instance_id"] for row in rows)
        if set(counts.values()) != {180}:
            raise ContractError(f"{name}: per-event counts={sorted(set(counts.values()))}")
        if set(row["operator"] for row in rows) != set(OPERATORS):
            raise ContractError(f"{name}: operator set mismatch")
        if set(int(row["prefix_k"]) for row in rows) != set(PREFIXES):
            raise ContractError(f"{name}: prefix set mismatch")
        if set(int(row["recovery_actor_seed"]) for row in rows) != set(ACTOR_SEEDS):
            raise ContractError(f"{name}: actor seed set mismatch")
    elif name == "c_atlas":
        key_unique(rows, ("event_instance_id", "prefix_k"), name)
        if len({row["event_instance_id"] for row in rows}) != 96:
            raise ContractError(f"{name}: event count mismatch")
    elif name == "c_boundaries":
        key_unique(rows, ("event_instance_id",), name)
        if len({row["event_instance_id"] for row in rows}) != 96:
            raise ContractError(f"{name}: event count mismatch")
    elif name == "c_invalid":
        key_unique(rows, ("event_instance_id",), name)
    elif name == "c_baseline":
        key_unique(rows, ("event_instance_id", "heldout_actor_seed", "method"), name)
    elif name == "c_crossfit":
        key_unique(rows, ("event_instance_id", "heldout_actor_seed"), name)
    elif name == "d_rows_pointer":
        # The official atlas stores one row per requested prefix.  Immediate
        # arms execute at prefix_k=d for every requested prefix, so prefix_k
        # alone is intentionally not the unique key.
        key_unique(rows, ("event_instance_id", "requested_prefix_k", "requested_arm"), name)
        if any(row.get("split") != "calibration" for row in rows):
            raise ContractError(f"{name}: non-calibration row in calibration atlas")
        if any(int(row["init_state_id"]) >= 40 for row in rows if "init_state_id" in row):
            raise ContractError(f"{name}: evaluation/reserve init id in calibration atlas")
        if any(int(row.get("init_state_id", 0)) >= 80 for row in rows):
            raise ContractError(f"{name}: reserve id 80-99 read")
        if set(row["requested_arm"] for row in rows) != {"CACHED_MATCHED", "CACHED_NOQUERY", "FRESH_MATCHED", "FULL_OLD_CHUNK", "HOLD_MATCHED", "IMMEDIATE_FRESH"}:
            raise ContractError(f"{name}: arm set mismatch")
    elif name == "d_method":
        key_unique(rows, ("event_instance_id", "requested_arm"), name)
        if any(row.get("split") != "evaluation" for row in rows):
            raise ContractError(f"{name}: non-evaluation row")
        if any(not 40 <= int(row.get("init_state_id", -1)) <= 79 for row in rows):
            raise ContractError(f"{name}: method row outside evaluation ids 40-79")
        if any(80 <= int(row.get("init_state_id", -1)) <= 99 for row in rows):
            raise ContractError(f"{name}: reserve row read")
    elif name == "d_paired":
        key_unique(rows, ("event_instance_id",), name)
        if any(row.get("split", "evaluation") != "evaluation" for row in rows):
            raise ContractError(f"{name}: non-evaluation row")


def safe_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = clean_json(value)
    path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def clean_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def stable_json(value: Any) -> str:
    return json.dumps(clean_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: clean_csv(row.get(column)) for column in columns})


def clean_csv(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return stable_json(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "NA"
    if value is None:
        return "NA"
    return value


def pct_half_up(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "NA"
    value = (Decimal(numerator) / Decimal(denominator) * Decimal("100")).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{value:.1f}%"


def mean(values: Iterable[Any]) -> float | None:
    vals = [float(value) for value in values if value is not None]
    return float(np.mean(vals)) if vals else None


def bool_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    return mean([bool(row.get(key)) for row in rows])


def git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def load_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read and validate immutable inputs; evaluation rows are deliberately deferred."""
    values: dict[str, Any] = {}
    receipt: dict[str, Any] = {"files": {}, "issues": [], "evaluation_rows_deferred": True}
    for name, rel in INPUT_PATHS.items():
        path = root / rel
        expected_hash = EXPECTED_HASHES.get(name)
        info: dict[str, Any] = {"path": rel, "exists": path.is_file(), "expected_sha256": expected_hash}
        if not path.is_file():
            receipt["issues"].append(f"missing:{name}")
            receipt["files"][name] = info
            continue
        actual = sha256_file(path)
        info["actual_sha256"] = actual
        info["sha256_match"] = expected_hash is None or actual == expected_hash
        if expected_hash is not None and actual != expected_hash:
            receipt["issues"].append(f"sha256:{name}:{actual}!={expected_hash}")
        # The evaluation-only Stage-2D rows are opened only after the B2
        # selection receipt is frozen.  Their hashes are still recorded now.
        if name in {"d_method", "d_paired"}:
            receipt["files"][name] = info
            continue
        try:
            if name == "d_rows_pointer":
                if official_load_jsonl is None:
                    raise ContractError(f"official Stage-2D loader unavailable: {_OFFICIAL_LOADER_IMPORT_ERROR}")
                pointer = load_object(path)
                if pointer.get("_sharded_jsonl") is not True:
                    raise ContractError("Stage-2D rows pointer is not a sharded pointer")
                if int(pointer.get("row_count", -1)) != EXPECTED_ROWS[name]:
                    raise ContractError("Stage-2D pointer row_count mismatch")
                values[name] = official_load_jsonl(path)
                # Hash every declared part without using a local loader/glob.
                parts_dir = path.parent / str(pointer["parts_dir"])
                for index in range(int(pointer.get("shard_count", 0))):
                    part_name = f"part-{index:03d}.jsonl"
                    part_path = parts_dir / part_name
                    part_key = f"d_rows_part_{index:03d}"
                    part_info = {"path": str(part_path.relative_to(root)), "exists": part_path.is_file(), "expected_sha256": EXPECTED_HASHES.get(part_key)}
                    if part_path.is_file():
                        part_info["actual_sha256"] = sha256_file(part_path)
                        part_info["sha256_match"] = part_info["actual_sha256"] == part_info["expected_sha256"]
                    else:
                        part_info["sha256_match"] = False
                    receipt["files"][part_key] = part_info
                    if not part_info["sha256_match"]:
                        receipt["issues"].append(f"part_sha256:{part_key}")
            elif path.suffix == ".jsonl":
                values[name] = load_plain_jsonl(path)
            else:
                values[name] = load_object(path)
            if name in EXPECTED_ROWS and isinstance(values.get(name), list):
                validate_rows(name, values[name])
            receipt["files"][name] = info
        except Exception as exc:
            receipt["issues"].append(f"schema:{name}:{type(exc).__name__}:{exc}")
            receipt["files"][name] = info
    # Validate metadata that is JSON rather than row-wise data.
    baselines = values.get("d_baselines")
    if isinstance(baselines, dict) and baselines.get("evaluation_outcomes_read") is not False:
        receipt["issues"].append("d_baselines:evaluation_outcomes_read_not_false")
    safe_dump(root / "artifacts/stage2e/source_freeze/input_receipt.json", receipt)
    return values, receipt


def rows_by(rows: Iterable[Mapping[str, Any]], *fields: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    """Deterministic grouping helper; it never silently overwrites a row."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in fields)].append(dict(row))
    return grouped


def event_cluster(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["task"]), int(row["init_state_id"])


def cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], value_key: str = "value", *,
    replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Cluster bootstrap over (task, init_state_id), never over prefix rows.

    Multiple event instances in a cluster are averaged first.  The returned
    point estimate therefore has the same equal-cluster weighting as every
    bootstrap draw.  This is intentionally a small local implementation so
    the receipt can state exactly which unit was resampled.
    """
    # ``rows`` is already in the frozen event-id order produced by
    # ``_method_summary``/``_delta_rows``.  Keep the first-appearance order
    # of clusters instead of sorting the cluster labels here.  The bootstrap
    # distribution is exchangeable in the population limit, but a finite,
    # seeded resample is not invariant to a permutation of the input vector.
    # The preregistration therefore freezes this order as part of the
    # reproducibility contract: event-level actor means first, then the
    # equally weighted clusters in first event-id appearance order.
    clusters = rows_by(rows, "task", "init_state_id")
    cluster_values = [
        float(np.mean([float(row[value_key]) for row in group]))
        for _, group in clusters.items()
        if all(row.get(value_key) is not None and math.isfinite(float(row[value_key])) for row in group)
    ]
    if not cluster_values:
        return {
            "estimate": None, "ci95": [None, None], "clusters": 0,
            "replicates": int(replicates), "seed": int(seed),
            "cluster_unit": "(task,init_state_id)",
            "cluster_aggregation": "event_actor_mean_then_equal_cluster_mean",
            "cluster_order": "first_appearance_in_stable_event_id_order",
            "percentile_method": "linear",
        }
    arr = np.asarray(cluster_values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    # The point estimate is cluster-weighted, and every draw samples clusters.
    draws = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        draws[index] = float(np.mean(arr[rng.integers(0, len(arr), size=len(arr))]))
    return {
        "estimate": float(np.mean(arr)),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "clusters": int(len(arr)), "replicates": int(replicates), "seed": int(seed),
        "cluster_unit": "(task,init_state_id)",
        "cluster_aggregation": "event_actor_mean_then_equal_cluster_mean",
        "cluster_order": "first_appearance_in_stable_event_id_order",
        "percentile_method": "linear",
    }


def selection_guard(rows: Iterable[Mapping[str, Any]], *, allowed_split: str = "calibration") -> None:
    """Open-deny guard used by every calibration-only selector.

    It rejects evaluation rows, reserve IDs and init IDs from the evaluation
    range before a caller can inspect any outcome field for selection.
    """
    for row in rows:
        split = str(row.get("split", ""))
        init_id = row.get("init_state_id")
        if split != allowed_split:
            raise ContractError(f"selection guard denied split={split!r}")
        if init_id is not None and int(init_id) >= 40:
            raise ContractError(f"selection guard denied init_state_id={init_id}")
        if init_id is not None and 80 <= int(init_id) <= 99:
            raise ContractError(f"selection guard denied reserve init_state_id={init_id}")


def _event_actor_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _cohort_events(boundaries: Sequence[Mapping[str, Any]], invalid: Sequence[Mapping[str, Any]], *, valid: bool) -> list[dict[str, Any]]:
    invalid_ids = {str(row["event_instance_id"]) for row in invalid}
    rows = [dict(row) for row in boundaries if row.get("split") == "calibration"]
    if valid:
        rows = [row for row in rows if str(row["event_instance_id"]) not in invalid_ids]
    return sorted(rows, key=lambda row: str(row["event_instance_id"]))


def _raw_recovery_lookup(recovery: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int, str, int], dict[str, Any]]:
    return {
        (str(row["event_instance_id"]), int(row["prefix_k"]), str(row["operator"]), int(row["recovery_actor_seed"])): dict(row)
        for row in recovery
    }


def _baseline_event_rows(baseline: Sequence[Mapping[str, Any]], method: str, instance: str) -> list[dict[str, Any]]:
    return [
        dict(row) for row in baseline
        if row.get("split") == "calibration" and str(row.get("method")) == method
        and str(row.get("event_instance_id")) == instance
    ]


def _fresh_fallback_rows(
    recovery_lookup: Mapping[tuple[str, int, str, int], Mapping[str, Any]], instance: str, k: int = DETECTION_PREFIX,
) -> list[dict[str, Any]]:
    return [
        dict(recovery_lookup[(instance, int(k), "fresh_h16", int(seed))])
        for seed in ACTOR_SEEDS
        if (instance, int(k), "fresh_h16", int(seed)) in recovery_lookup
    ]


def _event_method_rows(
    method: str, instance: str, baseline: Sequence[Mapping[str, Any]],
    recovery_lookup: Mapping[tuple[str, int, str, int], Mapping[str, Any]],
    boundary: Mapping[str, Any], *, restricted: bool,
) -> list[dict[str, Any]]:
    """Return the actor rows for a method, with the only allowed None fallback."""
    original = _baseline_event_rows(baseline, method, instance)
    if method not in {"k_last_observed_safe", "k_last_recoverable"}:
        return original
    if original:
        return original
    if restricted:
        return []
    return _fresh_fallback_rows(recovery_lookup, instance, DETECTION_PREFIX)


def _summarise_method_event_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "safe_success": _event_actor_mean(rows, "safe_success"),
        "new_non_nominal_actions": _event_actor_mean(rows, "new_non_nominal_actions"),
        "actor_calls": _event_actor_mean(rows, "actor_calls"),
        "action_budget": _event_actor_mean(rows, "action_budget") if rows and "action_budget" in rows[0] else _event_actor_mean(rows, "total_post_detection_action_budget"),
        "prefix_k": _event_actor_mean(rows, "prefix_k"),
        "actor_count": len(rows),
    }


def _method_summary(
    method: str, events: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]],
    recovery_lookup: Mapping[tuple[str, int, str, int], Mapping[str, Any]], *, restricted: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event_summaries: list[dict[str, Any]] = []
    for boundary in events:
        instance = str(boundary["event_instance_id"])
        rows = _event_method_rows(method, instance, baseline, recovery_lookup, boundary, restricted=restricted)
        if len(rows) != 0 and len(rows) != len(ACTOR_SEEDS):
            raise ContractError(f"{method}: event {instance} actor rows={len(rows)} not 0/3")
        if not rows:
            continue
        summary = _summarise_method_event_rows(rows)
        event_summaries.append({
            "event_instance_id": instance, "task": boundary["task"],
            "init_state_id": int(boundary["init_state_id"]), "parameter_id": boundary["parameter_id"],
            "method": method, "fallback": not bool(_baseline_event_rows(baseline, method, instance)),
            **summary,
        })
    summary = {
        "method": method, "event_count": len(event_summaries),
        "actor_row_count": int(sum(row["actor_count"] for row in event_summaries)),
        "safe_success": mean(row["safe_success"] for row in event_summaries),
        "new_non_nominal_actions": mean(row["new_non_nominal_actions"] for row in event_summaries),
        "actor_calls": mean(row["actor_calls"] for row in event_summaries),
        "mean_prefix_k": mean(row["prefix_k"] for row in event_summaries),
        "fallback_event_count": int(sum(bool(row["fallback"]) for row in event_summaries)),
    }
    return summary, event_summaries


def _fixed_winner(method_summaries: Sequence[Mapping[str, Any]]) -> str | None:
    candidates = [row for row in method_summaries if str(row.get("method")) in {"fixed_delay_1", "fixed_delay_2", "fixed_delay_4", "fixed_delay_8"}]
    candidates = [row for row in candidates if row.get("safe_success") is not None]
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda row: (
        -float(row["safe_success"]),
        float(row.get("new_non_nominal_actions") if row.get("new_non_nominal_actions") is not None else float("inf")),
        float(row.get("actor_calls") if row.get("actor_calls") is not None else float("inf")),
        str(row["method"]),
    ))
    return str(ranked[0]["method"])


def _delta_rows(k_rows: Sequence[Mapping[str, Any]], winner_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    right = {str(row["event_instance_id"]): row for row in winner_rows}
    out = []
    for left in k_rows:
        match = right.get(str(left["event_instance_id"]))
        if match is None or left.get("safe_success") is None or match.get("safe_success") is None:
            continue
        out.append({
            "event_instance_id": left["event_instance_id"], "task": left["task"],
            "init_state_id": left["init_state_id"], "parameter_id": left["parameter_id"],
            "value": float(left["safe_success"]) - float(match["safe_success"]),
            "k_lr_safe_success": left["safe_success"], "winner_safe_success": match["safe_success"],
        })
    return out


def run_a(
    root: Path, values: Mapping[str, Any], receipt: Mapping[str, Any], *, replicates: int,
) -> dict[str, Any]:
    """Reproduce A1 and perform calibration-only common-support analysis."""
    out_dir = root / "artifacts/stage2e/s0/common_support"
    baseline = list(values.get("c_baseline", []))
    recovery = list(values.get("c_recovery", []))
    boundaries = list(values.get("c_boundaries", []))
    invalid = list(values.get("c_invalid", []))
    if not baseline or not recovery or not boundaries:
        summary = {"task": "A", "status": "BLOCKED", "reason": "missing_or_invalid_stage2c_input", "diagnostic_only": DIAGNOSTIC, "formal_positive_evidence_allowed": FORMAL_POSITIVE, "new_idea_generated": NEW_IDEA}
        safe_dump(out_dir / "summary.json", summary)
        return summary

    # A1 is intentionally the published evaluation table only.  It never
    # participates in calibration selection.
    eval_baseline = [row for row in baseline if row.get("split") == "evaluation"]
    a1_rows: list[dict[str, Any]] = []
    a1_ok = True
    for method in BASELINE_METHODS:
        rows = [row for row in eval_baseline if str(row.get("method")) == method]
        numerator = int(sum(bool(row.get("safe_success")) for row in rows))
        denominator = len(rows)
        event_count = len({str(row["event_instance_id"]) for row in rows})
        expected_n, expected_d, expected_events = EXPECTED_A1[method]
        matches = (numerator, denominator, event_count) == (expected_n, expected_d, expected_events)
        a1_ok = a1_ok and matches
        a1_rows.append({
            "baseline": method, "numerator": numerator, "denominator": denominator,
            "percentage": pct_half_up(numerator, denominator), "support_events": event_count,
            "support_denominator": 48, "expected_numerator": expected_n,
            "expected_denominator": expected_d, "expected_support_events": expected_events,
            "exact_match": matches, "diagnostic_only": DIAGNOSTIC,
        })
    write_csv(out_dir / "a1_published_evaluation.csv", a1_rows, [
        "baseline", "numerator", "denominator", "percentage", "support_events", "support_denominator",
        "expected_numerator", "expected_denominator", "expected_support_events", "exact_match", "diagnostic_only",
    ])

    cohort_specs = [("all_contaminated", False), ("replay_valid_subset", True)]
    recovery_lookup = _raw_recovery_lookup(recovery)
    cohort_results: dict[str, Any] = {}
    all_table_rows: list[dict[str, Any]] = []
    winner_by_cohort: dict[str, str | None] = {}
    delta_by_cohort: dict[str, dict[str, Any]] = {}
    for cohort, valid in cohort_specs:
        cohort_events = _cohort_events(boundaries, invalid, valid=valid)
        # All boundary methods and baseline methods are calibration-only here.
        calibration_baseline = [row for row in baseline if row.get("split") == "calibration"]
        methods_for_cohort: list[dict[str, Any]] = []
        event_rows_for_method: dict[str, list[dict[str, Any]]] = {}
        for method in BASELINE_METHODS:
            method_summary, event_rows = _method_summary(
                method, cohort_events, calibration_baseline, recovery_lookup,
                restricted=False,
            )
            restricted_summary, restricted_rows = _method_summary(
                method, cohort_events, calibration_baseline, recovery_lookup,
                restricted=True,
            )
            method_summary["cohort"] = cohort
            method_summary["support_coverage"] = {
                "defined_events": restricted_summary["event_count"],
                "defined_rows": restricted_summary["actor_row_count"],
                "cohort_events": len(cohort_events), "global_events": 48,
                "coverage_of_cohort": restricted_summary["event_count"] / len(cohort_events) if cohort_events else None,
                "coverage_of_global_48": restricted_summary["event_count"] / 48.0,
            }
            method_summary["restricted_support"] = restricted_summary
            method_summary["full_support"] = {key: value for key, value in method_summary.items() if key not in {"restricted_support", "full_support"}}
            methods_for_cohort.append(method_summary)
            event_rows_for_method[method] = event_rows
            for support_name, support_summary, rows in (
                ("restricted_support", restricted_summary, restricted_rows),
                ("full_support", method_summary, event_rows),
            ):
                all_table_rows.append({
                    "cohort": cohort, "support": support_name, "method": method,
                    "defined_events": restricted_summary["event_count"],
                    "defined_actor_rows": restricted_summary["actor_row_count"],
                    "cohort_events": len(cohort_events), "global_events": 48,
                    "coverage_of_cohort": restricted_summary["event_count"] / len(cohort_events) if cohort_events else None,
                    "coverage_of_global_48": restricted_summary["event_count"] / 48.0,
                    "safe_success": support_summary.get("safe_success"),
                    "new_non_nominal_actions": support_summary.get("new_non_nominal_actions"),
                    "actor_calls": support_summary.get("actor_calls"),
                    "mean_prefix_k": support_summary.get("mean_prefix_k"),
                    "fallback_event_count": support_summary.get("fallback_event_count"),
                    "diagnostic_only": DIAGNOSTIC,
                    "formal_positive_evidence_allowed": FORMAL_POSITIVE,
                    "new_idea_generated": NEW_IDEA,
                })
        # The winner is frozen from calibration full-support summaries only.
        winner = _fixed_winner(methods_for_cohort)
        winner_by_cohort[cohort] = winner
        fixed_summary = next((row for row in methods_for_cohort if row["method"] == winner), None)
        k_summary = next((row for row in methods_for_cohort if row["method"] == "k_last_recoverable"), None)
        k_full_rows = event_rows_for_method.get("k_last_recoverable", [])
        # Build the comparable full-support event delta against the selected
        # fixed delay.  Both are event-first actor means.
        fixed_event_rows = event_rows_for_method.get(str(winner), []) if winner else []
        full_delta_rows = _delta_rows(k_full_rows, fixed_event_rows)
        # Restricted kLR excludes fallback events, while the winner is measured
        # on exactly that same event set.
        k_restricted_summary, k_restricted_rows = _method_summary(
            "k_last_recoverable", cohort_events, calibration_baseline, recovery_lookup, restricted=True,
        )
        fixed_restricted_rows = [
            row for row in event_rows_for_method.get(str(winner), [])
            if str(row["event_instance_id"]) in {str(item["event_instance_id"]) for item in k_restricted_rows}
        ] if winner else []
        restricted_delta_rows = _delta_rows(k_restricted_rows, fixed_restricted_rows)
        full_stats = cluster_bootstrap(full_delta_rows, replicates=replicates, seed=BOOTSTRAP_SEED)
        restricted_stats = cluster_bootstrap(restricted_delta_rows, replicates=replicates, seed=BOOTSTRAP_SEED)
        # The preregistered point is the equal-cluster estimate, not a pooled
        # event-row mean.  This matters in the valid subset because a cluster
        # can contain more than one event instance.
        full_delta = full_stats.get("estimate")
        restricted_delta = restricted_stats.get("estimate")
        absolute_inflation = None if restricted_delta is None or full_delta is None else restricted_delta - full_delta
        explained_fraction = None if restricted_delta is None or restricted_delta <= 0 or absolute_inflation is None else absolute_inflation / restricted_delta
        delta_by_cohort[cohort] = {
            "winner": winner, "restricted_delta": restricted_delta, "full_delta": full_delta,
            "absolute_inflation": absolute_inflation, "explained_fraction": explained_fraction,
            "restricted_bootstrap": restricted_stats, "full_bootstrap": full_stats,
            "restricted_event_count": len(restricted_delta_rows), "full_event_count": len(full_delta_rows),
            "historical_heldout_track_b_safe_success_difference": -0.09444444444444444,
            "diagnostic_only": DIAGNOSTIC, "formal_positive_evidence_allowed": FORMAL_POSITIVE,
            "new_idea_generated": NEW_IDEA,
        }
        cohort_results[cohort] = {
            "event_count": len(cohort_events), "invalid_excluded": len(invalid) if valid else 0,
            "method_summaries": methods_for_cohort, "strongest_fixed_delay": winner,
            "delta": delta_by_cohort[cohort],
        }

    write_csv(out_dir / "table.csv", all_table_rows, [
        "cohort", "support", "method", "defined_events", "defined_actor_rows", "cohort_events", "global_events",
        "coverage_of_cohort", "coverage_of_global_48", "safe_success", "new_non_nominal_actions", "actor_calls",
        "mean_prefix_k", "fallback_event_count", "diagnostic_only", "formal_positive_evidence_allowed", "new_idea_generated",
    ])
    # A5 is kept in a separate machine table to make the raw-prefix field audit
    # impossible to confuse with final cause_violation.
    a5_rows: list[dict[str, Any]] = []
    recovery_by_event = rows_by(recovery, "event_instance_id")
    boundary_lookup = {str(row["event_instance_id"]): row for row in boundaries}
    invalid_ids = {str(row["event_instance_id"]) for row in invalid}
    for cohort, valid in cohort_specs:
        events = [str(row["event_instance_id"]) for row in _cohort_events(boundaries, invalid, valid=valid)]
        for instance in events:
            raw_rows = recovery_by_event.get((instance,), [])
            if len(raw_rows) != 180:
                raise ContractError(f"A5 {instance}: expected 180 recovery rows, got {len(raw_rows)}")
            boundary = boundary_lookup[instance]
            covered = boundary.get("k_last_recoverable") is not None
            type_set = sorted({str(row.get("cause_violation_type")) for row in raw_rows if row.get("cause_violation_type") not in (None, "")})
            prefix_values = [bool(row.get("prefix_cause_violation")) for row in raw_rows]
            a5_rows.append({
                "cohort": cohort, "event_instance_id": instance, "task": boundary["task"],
                "parameter_id": boundary["parameter_id"], "support_group": "covered" if covered else "uncovered",
                "covered_k_last_recoverable": boundary.get("k_last_recoverable"),
                "cause_violation_type_set": type_set, "prefix_cause_violation_any": any(prefix_values),
                "prefix_cause_violation_fraction": float(np.mean(prefix_values)), "raw_row_count": len(raw_rows),
                "diagnostic_only": DIAGNOSTIC, "formal_positive_evidence_allowed": FORMAL_POSITIVE,
                "new_idea_generated": NEW_IDEA,
            })
    write_csv(out_dir / "a5_event_rows.csv", a5_rows, [
        "cohort", "event_instance_id", "task", "parameter_id", "support_group", "covered_k_last_recoverable",
        "cause_violation_type_set", "prefix_cause_violation_any", "prefix_cause_violation_fraction", "raw_row_count",
        "diagnostic_only", "formal_positive_evidence_allowed", "new_idea_generated",
    ])
    a5_table: list[dict[str, Any]] = []
    for key, group in rows_by(a5_rows, "cohort", "support_group", "task", "parameter_id").items():
        cohort, support_group, task, parameter = key
        a5_table.append({
            "cohort": cohort, "support_group": support_group, "task": task, "parameter_id": parameter,
            "events": len(group), "prefix_cause_violation_any_rate": mean(bool(row["prefix_cause_violation_any"]) for row in group),
            "prefix_cause_violation_fraction_mean": mean(row["prefix_cause_violation_fraction"] for row in group),
            "cause_violation_type_set": sorted({item for row in group for item in row["cause_violation_type_set"]}),
            "diagnostic_only": DIAGNOSTIC, "formal_positive_evidence_allowed": FORMAL_POSITIVE,
            "new_idea_generated": NEW_IDEA,
        })
    write_csv(out_dir / "a5_table.csv", a5_table, [
        "cohort", "support_group", "task", "parameter_id", "events", "prefix_cause_violation_any_rate",
        "prefix_cause_violation_fraction_mean", "cause_violation_type_set", "diagnostic_only",
        "formal_positive_evidence_allowed", "new_idea_generated",
    ])
    full_delta_values = [delta_by_cohort[name].get("full_delta") for name in ("all_contaminated", "replay_valid_subset")]
    full_lowers = [delta_by_cohort[name]["full_bootstrap"]["ci95"][0] for name in ("all_contaminated", "replay_valid_subset")]
    direction_consistent = (full_delta_values[0] is not None and full_delta_values[1] is not None and (full_delta_values[0] > 0) == (full_delta_values[1] > 0))
    full_advantage = direction_consistent and all(lower > -0.05 for lower in full_lowers) and all(delta > 0 for delta in full_delta_values)
    if not a1_ok:
        status, g0 = "BLOCKED", "BLOCKED_A1_REPRODUCTION"
    elif not direction_consistent:
        status, g0 = "INCONCLUSIVE", "INCONCLUSIVE"
    elif full_advantage:
        status, g0 = "PASS", "FULL_SUPPORT_ADVANTAGE"
    elif all(abs(float(delta or 0.0)) <= 1e-12 for delta in full_delta_values):
        status, g0 = "INCONCLUSIVE", "ZERO_FULL_SUPPORT_ADVANTAGE"
    elif all(float(delta or 0.0) < -1e-12 for delta in full_delta_values):
        status, g0 = "INCONCLUSIVE", "NEGATIVE_FULL_SUPPORT_ADVANTAGE"
    else:
        status, g0 = "INCONCLUSIVE", "INCONCLUSIVE"
    summary = {
        "task": "A", "status": status, "G0_1": g0, "A1_exact_reproduction": a1_ok,
        "A1_rows": a1_rows, "cohorts": cohort_results, "a5_event_rows": len(a5_rows),
        "a5_aggregation_fields": {"any": "raw prefix_cause_violation", "fraction": "raw prefix_cause_violation", "type_set": "raw cause_violation_type"},
        "direction_consistent_full_support": direction_consistent,
        "positive_label_allowed": False,
        "diagnostic_only": DIAGNOSTIC, "formal_positive_evidence_allowed": FORMAL_POSITIVE,
        "new_idea_generated": NEW_IDEA, "planned_pai_jobs": 0, "submitted_pai_jobs": 0,
    }
    safe_dump(out_dir / "summary.json", summary)
    return summary


def _semantic_family(task: str, parameter_id: str | None = None) -> str:
    text = f"{task} {parameter_id or ''}".lower()
    if "obstacle" in text or "future_" in text:
        return "path_obstacle"
    if "cream_cheese" in text or "shift_" in text:
        return "target_shift"
    return "unknown"


def _runtime_cohort(stage: str, cohort: str | None = None) -> str:
    if stage == "stage2d":
        return "fresh_process_verified"
    return str(cohort or "all_contaminated")


def _b1_rows_stage2c(recovery: Sequence[Mapping[str, Any]], invalid: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    invalid_ids = {str(row["event_instance_id"]) for row in invalid}
    result: list[dict[str, Any]] = []
    for cohort, excluded in (("all_contaminated", set()), ("replay_valid_subset", invalid_ids)):
        selected = [row for row in recovery if str(row["event_instance_id"]) not in excluded]
        groups = rows_by(selected, "task", "parameter_id", "operator")
        for (task, parameter, operator), group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
            result.append({
                "stage": "stage2c", "split": "calibration+evaluation",
                "task": task, "semantic_family": _semantic_family(str(task), str(parameter)),
                "original_parameter": parameter, "runtime_purity": "shared_runtime_known_invalid" if cohort == "all_contaminated" else "shared_runtime_no_detected_violation",
                "event": f"{len({row['event_instance_id'] for row in group})}_event_instances",
                "prefix": "k=2..16", "operator_arm": operator, "tail_horizon": "operator-defined",
                "configured_action_budget": "event_remaining_horizon (not a preset grid)",
                "configured_call_budget": "padded event budget (not a preset grid)",
                "actual_usage": {"mean_post_detection_actions": mean(row.get("total_post_detection_action_budget") for row in group), "mean_actor_calls": mean(row.get("actor_calls") for row in group)},
                "safe_success": mean(bool(row.get("safe_success")) for row in group), "cohort": cohort,
                "diagnostic_only": DIAGNOSTIC, "formal_positive_evidence_allowed": FORMAL_POSITIVE, "new_idea_generated": NEW_IDEA,
            })
    return result


def _b1_rows_stage2d(rows: Sequence[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    # This function is called after B2's selection receipt is committed.  The
    # rows are descriptive only and are never passed to B2/B3 selectors.
    selected = [row for row in rows if str(row.get("split")) == split]
    result: list[dict[str, Any]] = []
    groups = rows_by(selected, "task", "parameter_id", "requested_arm")
    for (task, parameter, arm), group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        result.append({
            "stage": "stage2d", "split": split, "task": task,
            "semantic_family": _semantic_family(str(task), str(parameter)), "original_parameter": parameter,
            "runtime_purity": "fresh_process_verified", "event": f"{len({row['event_instance_id'] for row in group})}_event_instances",
            "prefix": "requested k", "operator_arm": arm, "tail_horizon": mean(row.get("tail_horizon") for row in group),
            "configured_action_budget": mean(row.get("maximum_action_budget") for row in group),
            "configured_call_budget": mean(row.get("maximum_policy_call_budget") for row in group),
            "actual_usage": {"mean_post_detection_actions": mean(row.get("actual_post_detection_actions") for row in group), "mean_actor_calls": mean(row.get("actual_actor_calls") for row in group)},
            "safe_success": mean(bool(row.get("safe_success")) for row in group),
            "diagnostic_only": DIAGNOSTIC, "formal_positive_evidence_allowed": FORMAL_POSITIVE, "new_idea_generated": NEW_IDEA,
        })
    return result


def _stage2c_b2_cells(recovery: Sequence[Mapping[str, Any]], invalid: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return every descriptive stage-2C candidate cell; no budget inference."""
    cells: list[dict[str, Any]] = []
    invalid_ids = {str(row["event_instance_id"]) for row in invalid}
    # Use calibration only and event-first actor means before cluster means.
    for cohort, excluded in (("all_contaminated", set()), ("replay_valid_subset", invalid_ids)):
        selected = [row for row in recovery if row.get("split") == "calibration" and str(row["event_instance_id"]) not in excluded]
        grouped = rows_by(selected, "task", "parameter_id")
        for (task, parameter), cell_rows in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
            event_operator = rows_by(cell_rows, "event_instance_id", "operator")
            arm_event_values: dict[str, list[float]] = defaultdict(list)
            for (instance, operator), arm_rows in event_operator.items():
                if len({int(row["recovery_actor_seed"]) for row in arm_rows}) < 3:
                    continue
                arm_event_values[str(operator)].append(float(np.mean([bool(row["safe_success"]) for row in arm_rows])))
            arm_means = {arm: float(np.mean(vals)) for arm, vals in arm_event_values.items() if vals}
            # In this historical artifact all action/call budgets are event
            # residual/padded values, not an assigned grid. Keep this explicit.
            oracle = max(arm_means.values()) if arm_means else None
            weakest = min(arm_means.values()) if arm_means else None
            cells.append({
                "stage": "stage2c", "cohort": cohort, "task": task, "parameter_id": parameter,
                "configured_action_budget": "event_remaining_horizon", "tail_horizon": "operator-defined",
                "configured_call_budget": "padded_event_budget", "budget_levels": 1, "controlled_grid": False,
                "arm_count": len(arm_means), "arms": sorted(arm_means), "arm_event_means": arm_means,
                "oracle_best": oracle, "weakest_arm": weakest, "gap": None if oracle is None else oracle - weakest,
                "clusters": len({event_cluster(row) for row in cell_rows}), "criterion_oracle_range": oracle is not None and 0.25 <= oracle <= 0.85,
                "criterion_gap": oracle is not None and (oracle - weakest) >= 0.15 if oracle is not None else False,
                "criterion_clusters": len({event_cluster(row) for row in cell_rows}) >= 6,
                "criterion_arms": len(arm_means) >= 2, "diagnostic_only": DIAGNOSTIC,
                "formal_positive_evidence_allowed": FORMAL_POSITIVE, "new_idea_generated": NEW_IDEA,
            })
    return cells


def _stage2d_b2_cells(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    selected = [row for row in rows if row.get("split") == "calibration"]
    grouped = rows_by(selected, "task", "parameter_id")
    for (task, parameter), cell_rows in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        by_event_arm = rows_by(cell_rows, "event_instance_id", "requested_arm")
        arm_values: dict[str, list[float]] = defaultdict(list)
        for (instance, arm), arm_rows in by_event_arm.items():
            # There are repeated prefixes and repeats in the atlas. First take
            # an event/arm mean, then the cluster-level arm mean.
            arm_values[str(arm)].append(float(np.mean([bool(row["safe_success"]) for row in arm_rows])))
        arm_means = {arm: float(np.mean(vals)) for arm, vals in arm_values.items() if vals}
        oracle = max(arm_means.values()) if arm_means else None
        weakest = min(arm_means.values()) if arm_means else None
        clusters = len({event_cluster(row) for row in cell_rows})
        cells.append({
            "stage": "stage2d", "cohort": "fresh_process_verified", "task": task, "parameter_id": parameter,
            "configured_action_budget": 18, "tail_horizon": 4, "configured_call_budget": 2,
            "budget_levels": 1, "controlled_grid": False, "arm_count": len(arm_means), "arms": sorted(arm_means),
            "arm_event_means": arm_means, "oracle_best": oracle, "weakest_arm": weakest,
            "gap": None if oracle is None else oracle - weakest, "clusters": clusters,
            "criterion_oracle_range": oracle is not None and 0.25 <= oracle <= 0.85,
            "criterion_gap": oracle is not None and (oracle - weakest) >= 0.15 if oracle is not None else False,
            "criterion_clusters": clusters >= 6, "criterion_arms": len(arm_means) >= 2,
            "diagnostic_only": DIAGNOSTIC, "formal_positive_evidence_allowed": FORMAL_POSITIVE, "new_idea_generated": NEW_IDEA,
        })
    return cells


def _b1_axis_audit(
    baseline: Sequence[Mapping[str, Any]],
    stage2c_rows: Sequence[Mapping[str, Any]],
    stage2d_baselines: Mapping[str, Any],
    invalid: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Materialize the three confounded B1 axes and the only safe sensitivity.

    The commonly quoted Stage-2C ``10--16%`` band is the published
    calibration-frozen baseline band from fixed delay 4/8 through immediate;
    the exact fixed-delay-1/2 values are retained so this audit cannot hide
    an out-of-band baseline.  Stage-2D's ``0.85--3.39%`` band is copied from
    its frozen fixed-delay receipt, not recomputed from evaluation outcomes.
    No factor is treated as causal because budget, severity, and runtime
    purity change together across the two stages.
    """
    eval_baseline = [row for row in baseline if row.get("split") == "evaluation"]
    published_methods = (
        "fixed_delay_1", "fixed_delay_2", "fixed_delay_4", "fixed_delay_8",
        "immediate_fresh_h16",
    )
    published_rates = {
        method: float(np.mean([bool(row.get("safe_success")) for row in eval_baseline if row.get("method") == method]))
        for method in published_methods
    }
    published_band_methods = ("fixed_delay_4", "fixed_delay_8", "immediate_fresh_h16")
    d_fixed = []
    for row in stage2d_baselines.get("candidate_fixed_delays", []):
        d_fixed.append({
            "method": row.get("method"), "delay": row.get("delay"),
            "safe_success": row.get("safe_success"), "events": row.get("events"),
            "actual_post_detection_actions": row.get("actual_post_detection_actions"),
            "cause_violation": row.get("cause_violation"),
        })
    fixed_rates = [float(row["safe_success"]) for row in d_fixed if row.get("safe_success") is not None]

    # Stage-2C all -> replay-valid is an invalid-event-exclusion subset
    # sensitivity calculation, not a decomposition or a paired event effect.
    # The two cohorts contain different event sets; alignment is only by the
    # same-stage/task/parameter/operator row label, with no event join across
    # cohorts or stages. Rebuild from the full row list so values are auditable.
    cohort_aligned: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in stage2c_rows:
        if row.get("stage") != "stage2c":
            continue
        key = (str(row.get("task")), str(row.get("original_parameter")), str(row.get("operator_arm")))
        cohort = "all_contaminated" if row.get("runtime_purity") == "shared_runtime_known_invalid" else "replay_valid_subset"
        cohort_aligned.setdefault(key, {})[cohort] = float(row["safe_success"])
    sensitivity_rows = []
    for key in sorted(cohort_aligned):
        values = cohort_aligned[key]
        if set(values) == {"all_contaminated", "replay_valid_subset"}:
            sensitivity_rows.append({
                "task": key[0], "parameter_id": key[1], "operator": key[2],
                "all_contaminated": values["all_contaminated"],
                "replay_valid_subset": values["replay_valid_subset"],
                "valid_minus_all": values["replay_valid_subset"] - values["all_contaminated"],
            })
    stage2c_rates = [float(row["safe_success"]) for row in stage2c_rows if row.get("stage") == "stage2c"]
    return {
        "schema_version": 1,
        "published_stage2c_baseline_rates": published_rates,
        "published_stage2c_10_to_16_band": {
            "methods": list(published_band_methods),
            "exact_range": [min(published_rates[m] for m in published_band_methods), max(published_rates[m] for m in published_band_methods)],
            "display_range": ["10.4%", "16.0%"],
            "note": "The exact 5.6% fixed-delay-1 and 9.0% fixed-delay-2 rates remain in published_stage2c_baseline_rates.",
        },
        "stage2d_frozen_fixed_delay_rates": {
            "rows": d_fixed,
            "exact_range": [min(fixed_rates), max(fixed_rates)] if fixed_rates else [None, None],
            "display_range": ["0.85%", "3.39%"],
        },
        "stage2c_b1_operator_rate_range": [min(stage2c_rates), max(stage2c_rates)] if stage2c_rates else [None, None],
        "stage2c_invalid_events": {"invalid": len(invalid), "total": 96, "valid": 96 - len(invalid)},
        "stage2c_all_to_valid_sensitivity": sensitivity_rows,
        "axis_contract": {
            "budget": {
                "stage2c": "event_remaining_horizon plus padded_event_budget; not a preset grid",
                "stage2d": "maximum_action_budget=18, maximum_policy_call_budget=2, tail_horizon=4",
                "causal_decomposition": "NOT_IDENTIFIABLE",
            },
            "severity": {
                "stage2c": ["future_06_lateral_040mm", "future_14_lateral_020mm", "shift_040mm", "shift_060mm"],
                "stage2d": ["future_09__clearance_m010mm", "future_09__clearance_p000mm", "shift_060mm", "shift_080mm"],
                "causal_decomposition": "NOT_IDENTIFIABLE",
            },
            "runtime_purity": {
                "stage2c": ["shared_runtime_known_invalid", "shared_runtime_no_detected_violation"],
                "stage2d": ["fresh_process_verified"],
                "causal_decomposition": "NOT_IDENTIFIABLE",
            },
        },
        "causal_contribution": "NOT_IDENTIFIABLE_BUDGET_SEVERITY_RUNTIME_CO_VARY",
        "only_quantifiable_sensitivity": "stage2c_all_to_valid_invalid_event_exclusion_subset_sensitivity",
        "diagnostic_only": DIAGNOSTIC,
        "formal_positive_evidence_allowed": FORMAL_POSITIVE,
        "new_idea_generated": NEW_IDEA,
        "planned_pai_jobs": 0,
        "submitted_pai_jobs": 0,
    }


def run_b(
    root: Path, values: Mapping[str, Any], receipt: dict[str, Any], *, replicates: int,
) -> dict[str, Any]:
    out_dir = root / "artifacts/stage2e/s0/headroom"
    recovery = list(values.get("c_recovery", []))
    invalid = list(values.get("c_invalid", []))
    if not recovery:
        summary = {"task": "B", "status": "BLOCKED", "G0_2": "BUDGET_SCAN_FIRST", "reason": "missing_stage2c_recovery", "diagnostic_only": DIAGNOSTIC, "formal_positive_evidence_allowed": FORMAL_POSITIVE, "new_idea_generated": NEW_IDEA}
        safe_dump(out_dir / "summary.json", summary)
        return summary
    # B1 Stage-2C is descriptive and can be built before the B2 receipt.
    b1_rows = _b1_rows_stage2c(recovery, invalid)
    cells_c = _stage2c_b2_cells(recovery, invalid)
    d_rows = list(values.get("d_rows_pointer", []))
    cells_d = _stage2d_b2_cells(d_rows) if d_rows else []
    candidate_rows = cells_c + cells_d
    write_csv(out_dir / "table.csv", b1_rows + ([{"stage": row["stage"], "split": "calibration", "task": row["task"], "semantic_family": _semantic_family(str(row["task"]), str(row.get("parameter_id"))), "original_parameter": row.get("parameter_id"), "runtime_purity": row["cohort"], "event": "structural cell", "prefix": "calibration", "operator_arm": "|".join(row["arms"]), "tail_horizon": row["tail_horizon"], "configured_action_budget": row["configured_action_budget"], "configured_call_budget": row["configured_call_budget"], "actual_usage": {"arm_event_means": row["arm_event_means"]}, "safe_success": row["oracle_best"], "diagnostic_only": DIAGNOSTIC, "formal_positive_evidence_allowed": FORMAL_POSITIVE, "new_idea_generated": NEW_IDEA} for row in candidate_rows] if candidate_rows else []), [
        "stage", "split", "task", "semantic_family", "original_parameter", "runtime_purity", "event", "prefix",
        "operator_arm", "tail_horizon", "configured_action_budget", "configured_call_budget", "actual_usage", "safe_success",
        "diagnostic_only", "formal_positive_evidence_allowed", "new_idea_generated",
    ])
    axis_audit = _b1_axis_audit(
        list(values.get("c_baseline", [])), b1_rows,
        values.get("d_baselines", {}) if isinstance(values.get("d_baselines", {}), dict) else {},
        invalid,
    )
    safe_dump(out_dir / "axis_audit.json", axis_audit)
    # B2 selection is explicitly calibration-only and is frozen before the
    # deferred Stage-2D evaluation files are opened.
    selection_receipt = {
        "schema_version": 1, "selection_stage": "B2/B3", "input_split": "calibration",
        "evaluation_rows_opened_before_receipt": False, "reserve_ids_read": False,
        "stage2d_evaluation_read_for_selection": False,
        "candidate_count_all_listed": len(candidate_rows),
        "controlled_candidate_count": int(sum(bool(row.get("controlled_grid")) and int(row.get("budget_levels", 0)) >= 2 for row in candidate_rows)),
        "candidate_rows_hash_basis": [stable_json(row) for row in candidate_rows],
        "diagnostic_only": DIAGNOSTIC, "formal_positive_evidence_allowed": FORMAL_POSITIVE, "new_idea_generated": NEW_IDEA,
    }
    safe_dump(out_dir / "selection_receipt.json", selection_receipt)
    receipt["evaluation_selection_receipt_sha256"] = sha256_file(out_dir / "selection_receipt.json")
    safe_dump(root / "artifacts/stage2e/source_freeze/input_receipt.json", receipt)

    # Only now may Stage-2D evaluation be read, and only for a descriptive B1
    # appendix. It never enters candidate_rows or the G0-2 decision.
    eval_rows: list[dict[str, Any]] = []
    eval_paired: list[dict[str, Any]] = []
    eval_read_error: str | None = None
    try:
        d_method_path = root / INPUT_PATHS["d_method"]
        d_paired_path = root / INPUT_PATHS["d_paired"]
        eval_rows = load_plain_jsonl(d_method_path)
        validate_rows("d_method", eval_rows)
        eval_paired = load_plain_jsonl(d_paired_path)
        validate_rows("d_paired", eval_paired)
        receipt["evaluation_rows_deferred"] = False
        receipt["evaluation_rows_read_after_selection_receipt"] = True
        receipt["files"]["d_method"]["schema_validated_after_selection_receipt"] = True
        receipt["files"]["d_paired"]["schema_validated_after_selection_receipt"] = True
        safe_dump(root / "artifacts/stage2e/source_freeze/input_receipt.json", receipt)
    except Exception as exc:  # A/B selection remains valid; descriptive B1 is blocked.
        eval_read_error = f"{type(exc).__name__}: {exc}"
    b1_eval = _b1_rows_stage2d(eval_rows, "evaluation") if eval_rows else []
    if b1_eval:
        write_csv(out_dir / "stage2d_evaluation_descriptive.csv", b1_eval, [
            "stage", "split", "task", "semantic_family", "original_parameter", "runtime_purity", "event", "prefix",
            "operator_arm", "tail_horizon", "configured_action_budget", "configured_call_budget", "actual_usage", "safe_success",
            "diagnostic_only", "formal_positive_evidence_allowed", "new_idea_generated",
        ])
    controlled = [row for row in candidate_rows if bool(row.get("controlled_grid")) and int(row.get("budget_levels", 0)) >= 2 and bool(row.get("criterion_oracle_range")) and bool(row.get("criterion_gap")) and bool(row.get("criterion_clusters")) and bool(row.get("criterion_arms"))]
    # Per prereg, list all candidate tuples and never pick among them by an
    # outcome. The current artifacts expose only one configured level.
    summary = {
        "task": "B", "status": "PASS" if controlled else "BLOCKED", "G0_2": "CONTROLLED_HEADROOM_FOUND" if controlled else "BUDGET_SCAN_FIRST",
        "b1_rows": len(b1_rows), "b1_stage2d_evaluation_rows": len(b1_eval), "b1_stage2d_evaluation_read_error": eval_read_error,
        "b2_all_candidate_cells": candidate_rows, "b2_controlled_candidates": controlled,
        "b2_reason": None if controlled else "BLOCKED_BY_NO_CONFIGURED_BUDGET_GRID",
        "b3": {"S1_FIRST_ACTION": "SELECT_CONFIGURED_TUPLE" if controlled else "BUDGET_SCAN_FIRST", "selected_tuple": controlled[0] if controlled else None, "tie_break": ["max_policy_calls", "configured_action_budget", "tail_horizon", "canonical_severity_key", "canonical_operator_key"]},
        "budget_severity_runtime_causal_contribution": "NOT_IDENTIFIABLE/BLOCKED_BY_NO_COMMON_SUPPORT",
        "stage2c_all_to_valid_sensitivity_only": True,
        "selection_receipt": selection_receipt,
        "positive_label_allowed": False, "diagnostic_only": DIAGNOSTIC,
        "formal_positive_evidence_allowed": FORMAL_POSITIVE, "new_idea_generated": NEW_IDEA,
        "planned_pai_jobs": 0, "submitted_pai_jobs": 0,
    }
    safe_dump(out_dir / "summary.json", summary)
    return summary


def _last_boundary(values: Mapping[int, bool]) -> int | None:
    return max((int(k) for k, value in values.items() if bool(value)), default=None)


def _persistent_zero_boundary(values: Mapping[int, bool]) -> int | None:
    for k in PREFIXES:
        if all(not bool(values.get(index, False)) for index in PREFIXES if index >= k):
            return int(k)
    return None


def _ladder_for_event(event_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_cell = rows_by(event_rows, "prefix_k", "operator")
    complete = True
    family_c: dict[str, dict[str, float | None]] = {str(k): {} for k in PREFIXES}
    cell_raw: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for k in PREFIXES:
        key = str(k)
        cell_raw[key] = {}
        for op in OPERATORS:
            rows = by_cell.get((int(k), op), [])
            cell_raw[key][op] = rows
            seeds = {int(row.get("recovery_actor_seed")) for row in rows}
            if len(rows) != len(ACTOR_SEEDS) or seeds != set(ACTOR_SEEDS):
                complete = False
                family_c[key][op] = None
            else:
                values = [float(bool(row["safe_success"])) for row in rows]
                if any(not math.isfinite(value) for value in values):
                    complete = False
                    family_c[key][op] = None
                else:
                    family_c[key][op] = float(np.mean(values))
    family_members = {
        "U1": ("fresh_h4",),
        "U2": ("fresh_h4", "fresh_h16"),
        "U3": ("fresh_h4", "fresh_h16", "hold_one_step_then_fresh_h16"),
        "U4": OPERATORS,
    }
    r_by_family: dict[str, dict[int, bool | None]] = {}
    last_by_family: dict[str, int | None] = {}
    irrev_by_family: dict[str, int | None] = {}
    for family, members in family_members.items():
        r_values: dict[int, bool | None] = {}
        for k in PREFIXES:
            values = [family_c[str(k)].get(op) for op in members]
            r_values[int(k)] = None if any(value is None for value in values) else any(float(value) >= (2.0 / 3.0) for value in values)
        r_by_family[family] = r_values
        last_by_family[family] = _last_boundary({k: bool(v) for k, v in r_values.items()}) if complete else None
        irrev_by_family[family] = _persistent_zero_boundary({k: bool(v) for k, v in r_values.items()}) if complete else None
    violations: list[dict[str, Any]] = []
    if complete:
        for left, right in (("U1", "U2"), ("U2", "U3"), ("U3", "U4"), ("U1", "U3"), ("U1", "U4"), ("U2", "U4")):
            left_value = last_by_family[left] if last_by_family[left] is not None else DETECTION_PREFIX - 1
            right_value = last_by_family[right] if last_by_family[right] is not None else DETECTION_PREFIX - 1
            if left_value > right_value:
                violations.append({"left": left, "right": right, "left_last": last_by_family[left], "right_last": last_by_family[right], "sentinel": DETECTION_PREFIX - 1})
    return {
        "complete": complete, "family_C": family_c, "R_U": {family: {str(k): value for k, value in values.items()} for family, values in r_by_family.items()},
        "k_last_recoverable": last_by_family, "k_irrev_operator_relative": irrev_by_family,
        "monotonicity_violations": violations, "cell_raw_counts": {k: {op: len(rows) for op, rows in ops.items()} for k, ops in cell_raw.items()},
    }


def _c_cohort_events(recovery: Sequence[Mapping[str, Any]], invalid: Sequence[Mapping[str, Any]], *, valid: bool) -> list[str]:
    instances = sorted({str(row["event_instance_id"]) for row in recovery})
    if valid:
        invalid_ids = {str(row["event_instance_id"]) for row in invalid}
        instances = [instance for instance in instances if instance not in invalid_ids]
    return instances


def run_c(root: Path, values: Mapping[str, Any], *, replicates: int) -> dict[str, Any]:
    out_dir = root / "artifacts/stage2e/s0/operator_ladder"
    recovery = list(values.get("c_recovery", []))
    boundaries = list(values.get("c_boundaries", []))
    invalid = list(values.get("c_invalid", []))
    if not recovery:
        summary = {"task": "C", "status": "BLOCKED", "G0_3": "EXPAND_OPERATOR_FAMILY_FIRST", "reason": "missing_stage2c_recovery", "diagnostic_only": DIAGNOSTIC, "formal_positive_evidence_allowed": FORMAL_POSITIVE, "new_idea_generated": NEW_IDEA}
        safe_dump(out_dir / "summary.json", summary)
        return summary
    grouped = rows_by(recovery, "event_instance_id")
    boundary_lookup = {str(row["event_instance_id"]): dict(row) for row in boundaries}
    output_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for cohort, valid in (("all_contaminated", False), ("replay_valid_subset", True)):
        event_ids = _c_cohort_events(recovery, invalid, valid=valid)
        cohort_ladder: list[dict[str, Any]] = []
        for instance in event_ids:
            event_rows = grouped.get((instance,), [])
            ladder = _ladder_for_event(event_rows)
            boundary = boundary_lookup.get(instance, {})
            row = {
                "schema_version": 1, "cohort": cohort, "event_instance_id": instance,
                "event_id": boundary.get("event_id") or (event_rows[0].get("event_id") if event_rows else None),
                "task": boundary.get("task") or (event_rows[0].get("task") if event_rows else None),
                "split": boundary.get("split") or (event_rows[0].get("split") if event_rows else None),
                "init_state_id": boundary.get("init_state_id") or (event_rows[0].get("init_state_id") if event_rows else None),
                "parameter_id": boundary.get("parameter_id") or (event_rows[0].get("parameter_id") if event_rows else None),
                "complete_4op_3actor_k2_16": ladder["complete"],
                "family_C": ladder["family_C"], "R_U": ladder["R_U"],
                "k_last_recoverable": ladder["k_last_recoverable"],
                "k_irrev_operator_relative": ladder["k_irrev_operator_relative"],
                "monotonicity_violations": ladder["monotonicity_violations"],
                "u4_existing_k_last_recoverable": boundary.get("k_last_recoverable"),
                "u4_exact_crosscheck": ladder["k_last_recoverable"].get("U4") == boundary.get("k_last_recoverable"),
                "diagnostic_only": DIAGNOSTIC, "formal_positive_evidence_allowed": FORMAL_POSITIVE, "new_idea_generated": NEW_IDEA,
            }
            cohort_ladder.append(row)
            output_rows.append(row)
        u1_defined = [row for row in cohort_ladder if row["k_last_recoverable"].get("U1") is not None]
        u4_defined = [row for row in cohort_ladder if row["k_last_recoverable"].get("U4") is not None]
        both = [row for row in cohort_ladder if row["k_last_recoverable"].get("U1") is not None and row["k_last_recoverable"].get("U4") is not None]
        shifts = [abs(int(row["k_last_recoverable"]["U4"]) - int(row["k_last_recoverable"]["U1"])) for row in both]
        quantiles = [float(x) for x in np.percentile(np.asarray(shifts, dtype=np.float64), [0, 25, 50, 75, 100], method="linear")] if shifts else [None] * 5
        violation_count = sum(len(row["monotonicity_violations"]) for row in cohort_ladder)
        u4_mismatch = sum(not bool(row["u4_exact_crosscheck"]) for row in cohort_ladder)
        summaries[cohort] = {
            "events": len(cohort_ladder), "complete_events": sum(bool(row["complete_4op_3actor_k2_16"]) for row in cohort_ladder),
            "u1_defined_events": len(u1_defined), "u2_defined_events": sum(row["k_last_recoverable"].get("U2") is not None for row in cohort_ladder),
            "u3_defined_events": sum(row["k_last_recoverable"].get("U3") is not None for row in cohort_ladder),
            "u4_defined_events": len(u4_defined), "both_defined_events": len(both),
            "absolute_shift_type7_linear": quantiles,
            "absolute_shift_ge_1_fraction": (sum(value >= 1 for value in shifts) / len(shifts)) if shifts else None,
            "support_gain_u1_to_u4": len(u4_defined) - len(u1_defined),
            "monotonicity_violation_count": violation_count, "u4_exact_crosscheck_mismatch_count": u4_mismatch,
            "u1_support_ge_10": len(u1_defined) >= 10,
            "diagnostic_only": DIAGNOSTIC, "formal_positive_evidence_allowed": FORMAL_POSITIVE, "new_idea_generated": NEW_IDEA,
        }
    per_event_path = out_dir / "per_event.jsonl"
    per_event_path.parent.mkdir(parents=True, exist_ok=True)
    with per_event_path.open("w", encoding="utf-8") as handle:
        for row in sorted(output_rows, key=lambda item: (str(item["cohort"]), str(item["event_instance_id"]))):
            handle.write(stable_json(row) + "\n")
    full = summaries["all_contaminated"]
    valid = summaries["replay_valid_subset"]
    direction_consistent = ((full["absolute_shift_ge_1_fraction"] is not None and valid["absolute_shift_ge_1_fraction"] is not None) and ((full["absolute_shift_ge_1_fraction"] > 0) == (valid["absolute_shift_ge_1_fraction"] > 0)))
    g0_pass = all(summaries[name]["both_defined_events"] > 0 and summaries[name]["absolute_shift_ge_1_fraction"] is not None and summaries[name]["absolute_shift_ge_1_fraction"] >= 0.30 and summaries[name]["monotonicity_violation_count"] == 0 and summaries[name]["u1_support_ge_10"] for name in ("all_contaminated", "replay_valid_subset")) and direction_consistent
    summary = {
        "task": "C", "status": "PASS" if g0_pass else "INCONCLUSIVE", "G0_3": "OPERATOR_LADDER_USABLE" if g0_pass else "EXPAND_OPERATOR_FAMILY_FIRST",
        "cohorts": summaries, "direction_consistent": direction_consistent,
        "u4_crosscheck_exact": all(summaries[name]["u4_exact_crosscheck_mismatch_count"] == 0 for name in summaries),
        "none_sentinel_for_monotonicity_only": DETECTION_PREFIX - 1,
        "nested_family_theory": "monotonicity is algebraically guaranteed; violations indicate implementation/missing/NaN issues, not direct contamination proof",
        "positive_label_allowed": False, "diagnostic_only": DIAGNOSTIC,
        "formal_positive_evidence_allowed": FORMAL_POSITIVE, "new_idea_generated": NEW_IDEA,
        "planned_pai_jobs": 0, "submitted_pai_jobs": 0,
    }
    safe_dump(out_dir / "summary.json", summary)
    return summary


def write_instrumentation_contract(root: Path) -> None:
    text = """# Stage-2E S1 instrumentation contract (not started)

本文件只冻结未来 S1 的测量合同；S0 是零 rollout、零模型的离线重分析，未创建环境、未加载模型、未提交 PAI，也未启动 S1。

## D1 接触拓扑

每个 simulation step 必须记录规范化、排序后的完整 contact geom-name pair 集合和原始 pair 数据。`contact_count` 不能替代拓扑。记录必须带 `event_instance_id, prefix_k, operator, actor_seed, step` 唯一键。

## D2 可替换 cause 函数

cause 判据必须是离线纯函数，输入为原始接触拓扑、对象/目标几何和 task phase，输出 `(violation, violation_type)`。运行时不得把一个标签写死为唯一判据；既有 release-based label 必须原样保留，新增候选量只能并列记录，不能覆盖既有 label。

## D3 release 前候选观测

S1 必须同时记录一个 release 前可触发的候选量（例如 manipulated object 与非目标 geom 的接触拓扑越界），并明确它与既有 release-based cause 的时间关系、缺失状态和阈值来源。此合同不定义新 idea，也不把候选量当作验证结论。

## D4 fresh-process provenance

每条 branch 记录 `pid`、environment construction hash、actor chunk bytes hash，并沿用 Stage-2D isolation signature 字段：simulator/observation/state-history/action-history/robot qpos-qvel/contact/object/target/gripper/task-phase/CauseTracker/executed-prefix hashes。`pid/env_hash/chunk_hash` 任一缺失时 branch/event 必须 BLOCKED。每个 `(event,prefix,operator,actor_seed,repeat)` 必须是 `multiprocessing.get_context("spawn")` 的独立进程，不能共享 env、wrapper、RNG、cache 或 runtime。

## D5 严格边界

S1 的 actors 固定为 7/17/29；calibration 与 evaluation 严格分离；任何选择器/预算选择必须在 calibration receipt 冻结后才能读取 evaluation 的描述性结果。不得加入 world model、RGB/VLA、训练、actor retraining、learned head 或闭环结果。

`diagnostic_only=true`, `formal_positive_evidence_allowed=false`, `new_idea_generated=false`。
"""
    path = root / "experiments/r16_p14_stage2e/INSTRUMENTATION_S1.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_metric_contract(root: Path) -> None:
    text = """# Stage-2E S0 metric contract

S0 is CPU-only, zero-rollout, diagnostic-only offline reanalysis. Independent unit is an event cluster `(task, init_state_id)`; prefix rows are never independent observations. Stage-2C numbers are always reported as `all_contaminated` (96 events) and `replay_valid_subset` (63 events). Bootstrap is 10,000 resamples, seed 216214, percentile 95% CI and cluster-first weighting. For a paired delta, the three actor rows are averaged within each event first; repeated event instances are then averaged with equal weight within each `(task, init_state_id)` cluster. Clusters are kept in their first-appearance order after the input event IDs have been stably sorted; this order is part of the finite-seed reproducibility contract (permuting a seeded vector can change a finite percentile even though the population bootstrap is exchangeable). The `all_contaminated` → `replay_valid_subset` comparison is instead an invalid-event-exclusion subset sensitivity: the event sets differ, so it is not a paired event effect and is not causal.

A1 reproduces the frozen evaluation machine table exactly. A2/A3/A4 use calibration only, boundary `None` falls back to `d=2` with `fresh_h16`, and fixed winner candidates are only fixed delays 1/2/4/8 with the preregistered lexicographic tie-break. A5 aggregates raw `prefix_cause_violation` for `any` and `fraction`; only `cause_violation_type` supplies the type set.

B1 is within-stage descriptive alignment only. Stage-2C runtime labels are `shared_runtime_known_invalid`/`shared_runtime_no_detected_violation`; Stage-2D calibration is `fresh_process_verified`. B2 requires at least two configured preset budget levels; event residual horizon or actual usage is not a budget grid. Evaluation files are open-deny until the B2 selection receipt is written, and afterwards are descriptive only.

C recomputes U1/U2/U3/U4 from the complete 4-operator × 3-actor × k=2..16 raw grid. `None` uses sentinel `d-1=1` only for monotonicity. U4 is cross-checked against the existing atlas. D only writes an S1 contract and does not execute it.

All outputs carry `diagnostic_only=true`, `formal_positive_evidence_allowed=false`, `new_idea_generated=false`; PAI planned/submitted jobs are `0/0`.
"""
    path = root / "experiments/r16_p14_stage2e/metric_contract.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_commands(root: Path) -> None:
    path = root / "experiments/r16_p14_stage2e/commands.sh"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
python "$ROOT/scripts/run_r16p14_stage2e_s0.py" --root "$ROOT"
python -m unittest discover -s "$ROOT/experiments/r16_p14_stage2e/tests" -v
python "$ROOT/scripts/verify_r16p14_stage2e_s0.py" --root "$ROOT"
""", encoding="utf-8")
    path.chmod(0o755)


def write_source_manifest(root: Path) -> None:
    files = {name: {"path": rel, "sha256": sha256_file(root / rel) if (root / rel).is_file() else None} for name, rel in INPUT_PATHS.items()}
    files.update({key: info for key, info in json.loads((root / "artifacts/stage2e/source_freeze/input_receipt.json").read_text(encoding="utf-8")).get("files", {}).items() if key.startswith("d_rows_part_")})
    manifest = {
        "schema_version": 1, "stage": "R16-P14 Stage-2E/S0", "analysis_mode": "CPU offline; zero rollout/model/GPU/PAI",
        "immutable_parent_commit": "edebdfc64576129d994535dacb76de930f493c8d", "immutable_parent_tree": "fa0a951abc1aa778cfa76663679173557e6c9b96",
        "preregistration_sha256": sha256_file(root / "experiments/r16_p14_stage2e/PREREG_S0.md"),
        "input_files": files,
        "official_sharded_loader": "experiments/r16_p14_stage2d/r16_p14_stage2d/io_utils.py:load_jsonl",
        "old_tree_read_only": ["artifacts/stage2c", "artifacts/stage2d", "experiments/r16_p14_stage2a", "experiments/r16_p14_stage2b", "experiments/r16_p14_stage2c", "experiments/r16_p14_stage2d", "docs/feishu"],
        "deferred_stage2d_evaluation": "opened only after artifacts/stage2e/s0/headroom/selection_receipt.json; descriptive B1 only",
        "runtime_versions": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__},
        # Exclude generated/untracked Stage-2E outputs from this receipt.  The
        # tracked-worktree status remains an exact, rerun-stable source
        # freeze, whereas including the output files would make the receipt
        # differ between the first and second analysis invocation.
        "git": {"head": git_value(root, "rev-parse", "HEAD"), "tree": git_value(root, "rev-parse", "HEAD^{tree}"), "status": git_value(root, "status", "--short", "--untracked-files=no"), "status_scope": "tracked_only"},
        "read_order": ["validate paths/hash/schema/rows", "A1/A2/A3/A4/A5", "B2 selection receipt", "descriptive Stage2D evaluation B1", "C", "D"],
        "diagnostic_only": DIAGNOSTIC, "formal_positive_evidence_allowed": FORMAL_POSITIVE, "new_idea_generated": NEW_IDEA,
        "planned_pai_jobs": 0, "submitted_pai_jobs": 0,
    }
    safe_dump(root / "artifacts/stage2e/source_freeze/analysis_manifest.json", manifest)


def write_immutable_predecessor_receipt(root: Path) -> None:
    """Freeze tree-object evidence that all predecessor directories are unchanged.

    Tree object IDs are content-addressed hashes for the complete directory
    contents.  Recording both the immutable parent and current branch values
    makes the read-only boundary independently auditable without copying or
    rewriting any predecessor artifact.
    """
    entries: list[dict[str, Any]] = []
    for rel in IMMUTABLE_OLD_PATHS:
        parent_tree = git_value(root, "rev-parse", f"{IMMUTABLE_PARENT_COMMIT}:{rel}")
        current_tree = git_value(root, "rev-parse", f"HEAD:{rel}")
        entries.append({
            "path": rel,
            "parent_tree": parent_tree,
            "current_head_tree": current_tree,
            "equal": parent_tree is not None and parent_tree == current_tree,
        })
    receipt = {
        "schema_version": 1,
        "stage": "R16-P14 Stage-2E/S0",
        "immutable_parent_commit": IMMUTABLE_PARENT_COMMIT,
        "immutable_parent_tree": IMMUTABLE_PARENT_TREE,
        "old_paths": entries,
        "all_equal": bool(entries) and all(bool(item["equal"]) for item in entries),
        "read_only": True,
        "diagnostic_only": DIAGNOSTIC,
        "formal_positive_evidence_allowed": FORMAL_POSITIVE,
        "new_idea_generated": NEW_IDEA,
    }
    safe_dump(root / "artifacts/stage2e/source_freeze/immutable_predecessor_receipt.json", receipt)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(root: Path, a: Mapping[str, Any], b: Mapping[str, Any], c: Mapping[str, Any]) -> None:
    a_cohorts = a.get("cohorts", {})
    lines = [
        "# R16-P14 Stage-2E / step7 S0 实验报告",
        "",
        "> 本报告是零 rollout、零模型、零 GPU/PAI 的 CPU 离线诊断重分析。它不构成正式正面证据。",
        "",
        "## 结论先行",
        "",
        f"- `diagnostic_only=true`，`formal_positive_evidence_allowed=false`，`new_idea_generated=false`。计划/提交的 PAI job 为 `0/0`；S1 未启动。",
        f"- A 共同支撑：`{a.get('status')}`，G0-1=`{a.get('G0_1')}`，A1 精确复现=`{a.get('A1_exact_reproduction')}`。",
        f"- B 头顶空间：`{b.get('status')}`，G0-2=`{b.get('G0_2')}`；现有 artifact 没有至少两个预设 configured budget levels，因此不能把描述性差异解释为预算因果证据。",
        f"- C 算子阶梯：`{c.get('status')}`，G0-3=`{c.get('G0_3')}`；U4 与既有 atlas exact crosscheck=`{c.get('u4_crosscheck_exact')}`。",
        "- 总体仍保持 `accepted=false`、novelty ceiling=`N2_ORACLE_PROTOCOL_BOUNDARY_ONLY`。本阶段没有 learned/deployable evidence。",
        "",
        "## 不可变的上游事实",
        "",
        "- Stage-1b universal hypothesis=`KILLED_IMMUTABLE`；Stage-2B 的 chunk executability 为 PASS，但 actor-conditioned perturbation 与 actor-history replay 均 BLOCKED，Track A/B 均 INCONCLUSIVE。",
        "- Stage-2C qualification、replay contract 均 BLOCKED；其 shared mutable runtime 污染和 33 个 invalid events 作为历史限制保留。Stage-2D 的事件构造结论也不可改写。",
        "- 本 S0 不重新解释、不覆盖这些事实；只在既有机器 artifact 上做有边界的离线重算。",
        "",
        "## A：共同支撑与复现",
        "",
        "A1 使用 Stage-2C evaluation machine rows 复现冻结表，不把其结果用于 calibration 选择。A2–A5 只使用 calibration rows，并同时保留 96-event `all_contaminated` 与排除 invalid 后的 `replay_valid_subset`。",
        "",
        "| cohort | strongest fixed | restricted Δ(kLR−fixed) | restricted 95% CI | full Δ(kLR−fixed) | full 95% CI | inflation |",
        "|---|---|---:|---|---:|---|---:|",
    ]
    for cohort in ("all_contaminated", "replay_valid_subset"):
        delta = a_cohorts.get(cohort, {}).get("delta", {})
        restricted_ci = delta.get("restricted_bootstrap", {}).get("ci95", [None, None])
        full_ci = delta.get("full_bootstrap", {}).get("ci95", [None, None])
        lines.append(f"| {cohort} | {delta.get('winner')} | {_fmt(delta.get('restricted_delta'))} | [{_fmt(restricted_ci[0], 15)}, {_fmt(restricted_ci[1], 15)}] | {_fmt(delta.get('full_delta'))} | [{_fmt(full_ci[0], 15)}, {_fmt(full_ci[1], 15)}] | {_fmt(delta.get('absolute_inflation'))} |")
    lines += [
        "",
        "A5 的 `any` 与 `fraction` 均明确来自 raw `prefix_cause_violation`，cause type set 来自 raw `cause_violation_type`；每 event 固定聚合 180 行，未把 prefix 行当独立样本。",
        "",
        "## B：头顶空间（仅描述）",
        "",
        f"B1 在 stage 内对齐，未跨 stage 做 event join，也未把相同毫米字符串当作同一 severity。Stage-2C 分别标记 `shared_runtime_known_invalid` 和 `shared_runtime_no_detected_violation`；Stage-2D 标记 `fresh_process_verified`。B2 candidate cells 全部列出，但 configured grid levels 不足，故 G0-2=`{b.get('G0_2')}`，S1 首动作=`BUDGET_SCAN_FIRST`。",
        "",
        "Stage-2C 的旧 aggregation 路径先在 `experiments/r16_p14_stage2c/r16_p14_stage2c/aggregate.py::build_atlas` 中按 `(event_instance_id,prefix_k)` 分组，再由同一函数对每个 operator 的三条 actor rows 计算 `C_family`，并用 `R_U=any(C_family>=2/3)`；B 本次只审计这种结构能否在受控预算下形成可解释 headroom，不能用 actual usage 伪造 preset grid。",
        "",
        "B1 的已发表 Stage-2C baseline 区间是约 10–16%，Stage-2D 冻结 fixed-delay 描述区间是 0.85–3.39%。这两个区间同时随 configured/remaining budget、perturbation severity 以及 runtime purity 变化；两阶段的 event/cohort、几何参数和执行契约也不同，因此不能把任一差值归因给单一机制。S0 唯一量化的是 Stage-2C 从 `all_contaminated` 排除 replay-invalid 事件得到 `replay_valid_subset` 的 invalid-event-exclusion subset sensitivity（见 `axis_audit.json`）；由于两 cohort 的 event 集合不同，这不是 paired event effect，也不是因果分解，budget、severity、runtime 的 causal contribution 均为 `NOT_IDENTIFIABLE`。",
        "",
        "## C：算子相对 recoverability 阶梯",
        "",
    ]
    for cohort in ("all_contaminated", "replay_valid_subset"):
        item = c.get("cohorts", {}).get(cohort, {})
        lines.append(f"- `{cohort}`：events={item.get('events')}，U1/U2/U3/U4 defined={item.get('u1_defined_events')}/{item.get('u2_defined_events')}/{item.get('u3_defined_events')}/{item.get('u4_defined_events')}，both-defined={item.get('both_defined_events')}，Type-7 absolute shift quantiles={item.get('absolute_shift_type7_linear')}，|shift|≥1 fraction={_fmt(item.get('absolute_shift_ge_1_fraction'))}，monotonic violations={item.get('monotonicity_violation_count')}。")
    lines += [
        "",
        f"C 的 U4 与既有 atlas 逐 event crosscheck 为 `{c.get('u4_crosscheck_exact')}`。集合嵌套使单调性在代数上应恒真；本次 violation 被作为实现/缺失/NaN 审计项，而非直接污染证明。valid cohort 的 U1 support 若不足 10，则 G0-3 只能为 `EXPAND_OPERATOR_FAMILY_FIRST`。None 只在单调性比较使用 sentinel d−1=1，不进入安全成功或位移。",
        "",
        "## 机制反解（observation → code trace → mechanism → falsifier）",
        "",
        "1. observation：Stage-2C 不同 operator 的 C_family / boundary 会变化。code trace（仅 Stage-2C）：`experiments/r16_p14_stage2c/r16_p14_stage2c/aggregate.py::build_atlas`（operator actor-mean 与 `R_U`），以及 `experiments/r16_p14_stage2c/r16_p14_stage2c/runtime.py::reconstruct_to_prefix`、`::rollback_action`、`::hold_action`（prefix 重放与 prelude）。mechanism：差异首先来自 hold/rollback 的 action prelude 与 fresh_h4/fresh_h16 的 horizon/调用预算改变，而不是新的学习器。falsifier：下一阶段必须在两个以上预设 configured budget levels、相同事件和相同 tail 下做受控 factorial 对齐；S0 不实现或运行它。",
        "2. observation：A 的 boundary baseline 有定义事件较少，fallback 后 full-support 数字与 restricted 数字可能不同。code trace（仅 Stage-2C）：`experiments/r16_p14_stage2c/r16_p14_stage2c/aggregate.py::build_atlas` 的 boundary loop（`k_last_recoverable`/`prefix_safety_valid`）和本脚本 `::_event_method_rows`（None→d=2/fresh_h16 fallback）。mechanism：restricted/full 差异可由 support selection 与 shared-runtime contamination 驱动，不能直接当成真实 selector 增益。falsifier：在 fresh-process、共同预算且 outcome-blind 的新测量中，预注册同一 cluster bootstrap 后仍需保持方向。",
        "3. observation：Stage-2D 的 matched/cached/fresh 执行有不同 actual calls。code trace（仅 Stage-2D）：`experiments/r16_p14_stage2d/r16_p14_stage2d/runtime.py::arm_plan`、`::execute_branch` 的 mode 分支与 tail loop；该函数在成功分支不 padding tail calls。mechanism：actual call/time 差异是执行路径结果，不等价于 configured budget 或因果 headroom；B2 因缺受控 grid fail-closed。falsifier：至少两个 preset budgets、相同 tail/action/call cap 的 fresh-process matrix。",
        "",
        "## 边界与停止条件",
        "",
        "- C/B/A 均为 diagnostic-only 离线结果；不得写成 VLA、训练、闭环或 accepted=true 证据。",
        "- 本阶段不启动 S1，不做 simulator/model/rollout；任何需要这些资源的问题登记为待测。",
        "- `new_idea_generated=false`；没有生成新 idea，也没有将机制说明包装成部署主张。",
        "- 数据输入、hash、官方 Stage-2D sharded loader、selection receipt、测试和 checksum 见相邻 artifacts。",
    ]
    path = root / "experiments/r16_p14_stage2e/reports/REPORT_S0.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision(root: Path, a: Mapping[str, Any], b: Mapping[str, Any], c: Mapping[str, Any]) -> dict[str, Any]:
    decision = {
        "stage1b_universal_hypothesis": "KILLED_IMMUTABLE",
        "stage2c_status": "BLOCKED_UPSTREAM_IMMUTABLE",
        "stage2d_status": "BLOCKED_UPSTREAM_IMMUTABLE",
        "A_status": a.get("status"), "A_G0_1": a.get("G0_1"),
        "B_status": b.get("status"), "B_G0_2": b.get("G0_2"),
        "C_status": c.get("status"), "C_G0_3": c.get("G0_3"),
        "diagnostic_only": DIAGNOSTIC, "formal_positive_evidence_allowed": FORMAL_POSITIVE,
        "new_idea_generated": NEW_IDEA, "planned_pai_jobs": 0, "submitted_pai_jobs": 0,
        "s1_started": False, "accepted": False, "novelty": "N2_ORACLE_PROTOCOL_BOUNDARY_ONLY",
    }
    safe_dump(root / "experiments/r16_p14_stage2e/decision.json", decision)
    return decision


def write_checksums(root: Path) -> Path:
    """Create stable checksum manifest, excluding the manifest itself."""
    manifest = root / "artifacts/stage2e/SHA256SUMS"
    paths: list[Path] = []
    for base in (root / "experiments/r16_p14_stage2e", root / "artifacts/stage2e"):
        if not base.exists():
            continue
        paths.extend(
            path for path in base.rglob("*")
            if path.is_file() and path != manifest and path.suffix != ".pyc" and "__pycache__" not in path.parts
        )
    # Also cover the two analysis entrypoints, because they are required
    # reproducibility artifacts but live outside the stage directories.
    for path in (root / "scripts/run_r16p14_stage2e_s0.py", root / "scripts/verify_r16p14_stage2e_s0.py"):
        if path.is_file():
            paths.append(path)
    unique = sorted({path.resolve() for path in paths}, key=lambda path: str(path.relative_to(root)))
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in unique]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage-2E S0 CPU-only offline analysis")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.bootstrap_replicates <= 0:
        raise SystemExit("bootstrap replicates must be positive")
    # The prereg file is an immutable prerequisite and is never opened for
    # rewriting. It is read only for existence/hash in the source receipt.
    prereg = root / "experiments/r16_p14_stage2e/PREREG_S0.md"
    if not prereg.is_file():
        raise SystemExit(f"missing frozen preregistration: {prereg}")
    (root / "artifacts/stage2e").mkdir(parents=True, exist_ok=True)
    write_metric_contract(root)
    write_instrumentation_contract(root)
    write_commands(root)
    values, receipt = load_inputs(root)
    write_source_manifest(root)
    write_immutable_predecessor_receipt(root)
    # A/B/C remain independent: an invalid input blocks only the affected
    # path. C uses the recovery rows independently from A/B summaries.
    a = run_a(root, values, receipt, replicates=args.bootstrap_replicates)
    b = run_b(root, values, receipt, replicates=args.bootstrap_replicates)
    c = run_c(root, values, replicates=args.bootstrap_replicates)
    decision = write_decision(root, a, b, c)
    write_report(root, a, b, c)
    # A machine-readable top-level analysis receipt summarizes the actual
    # execution without introducing a second decision source.
    safe_dump(root / "artifacts/stage2e/s0/summary.json", {
        "stage": "R16-P14 Stage-2E/S0", "A": a, "B": b, "C": c, "decision": decision,
        "diagnostic_only": DIAGNOSTIC, "formal_positive_evidence_allowed": FORMAL_POSITIVE,
        "new_idea_generated": NEW_IDEA, "planned_pai_jobs": 0, "submitted_pai_jobs": 0,
    })
    write_checksums(root)
    print(json.dumps({"A": a.get("status"), "B": b.get("status"), "C": c.get("status"), "G0_1": a.get("G0_1"), "G0_2": b.get("G0_2"), "G0_3": c.get("G0_3")}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
