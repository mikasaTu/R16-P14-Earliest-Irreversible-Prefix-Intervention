from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .actor_data import TaskArrays, existing_cache_path
from .actor_eval import load_actor
from .io_utils import atomic_write_csv, atomic_write_json, atomic_write_text
from .settings import (
    ACTION_DIM,
    DEFAULT_CACHE_ROOT,
    TASK_NAMES,
    TASK_TO_INDEX,
)


CONDITIONS = (
    "full",
    "repeat_current_state",
    "zero_action_history",
    "cyclic_task_id",
)


def deployed_predictions(
    model,
    normalization,
    states: np.ndarray,
    histories: np.ndarray,
    task_ids: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    normalized_states = (
        states - normalization.state_mean
    ) / normalization.state_std
    normalized_histories = (
        histories - normalization.action_mean
    ) / normalization.action_std
    with torch.no_grad():
        predicted = model(
            torch.from_numpy(normalized_states).to(device),
            torch.from_numpy(normalized_histories).to(device),
            torch.from_numpy(task_ids).to(device),
        ).cpu().numpy()
    predicted = (
        predicted * normalization.action_std + normalization.action_mean
    )
    predicted = np.clip(predicted, -1.0, 1.0).astype(np.float32)
    predicted[..., -1] = np.where(predicted[..., -1] >= 0.0, 1.0, -1.0)
    return predicted


def condition_inputs(
    states: np.ndarray,
    histories: np.ndarray,
    task_ids: np.ndarray,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = states.copy()
    histories = histories.copy()
    task_ids = task_ids.copy()
    if condition == "full":
        pass
    elif condition == "repeat_current_state":
        states[:] = states[:, -1:, :]
    elif condition == "zero_action_history":
        histories.fill(0.0)
    elif condition == "cyclic_task_id":
        task_ids[:] = (task_ids + 1) % len(TASK_NAMES)
    else:
        raise ValueError(f"unknown condition: {condition}")
    return states, histories, task_ids


def error_metrics(
    predicted: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
) -> dict[str, float]:
    valid = masks[..., None]
    denominator = float(masks.sum())
    position_absolute = np.abs(predicted[..., :6] - targets[..., :6])
    first_position = np.abs(predicted[:, 0, :6] - targets[:, 0, :6])
    gripper_correct = np.sign(predicted[..., 6]) == np.sign(targets[..., 6])
    return {
        "first_action_position_mae": float(first_position.mean()),
        "chunk_position_mae": float(
            (position_absolute * valid).sum() / max(denominator * 6.0, 1.0)
        ),
        "first_action_gripper_accuracy": float(
            (np.sign(predicted[:, 0, 6]) == np.sign(targets[:, 0, 6])).mean()
        ),
        "chunk_gripper_accuracy": float(
            (gripper_correct * masks).sum() / max(denominator, 1.0)
        ),
    }


def selected_rows(arrays: TaskArrays, count: int) -> np.ndarray:
    actual = min(count, len(arrays.features))
    return np.linspace(
        0, len(arrays.features) - 1, num=actual, dtype=np.int64
    )


def audit_checkpoint(
    checkpoint: Path,
    *,
    cache_root: Path,
    samples_per_task: int,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    payload, model, normalization = load_actor(checkpoint, device)
    records: list[dict[str, Any]] = []
    for task in TASK_NAMES:
        path = existing_cache_path(task, cache_root)
        if path is None:
            raise FileNotFoundError(f"missing actor cache for {task}")
        arrays = TaskArrays(task, path)
        rows = selected_rows(arrays, samples_per_task)
        states, histories, task_ids, targets, masks = arrays.batch_raw(rows)
        for condition in CONDITIONS:
            conditioned = condition_inputs(states, histories, task_ids, condition)
            predicted = deployed_predictions(
                model,
                normalization,
                *conditioned,
                device,
            )
            records.append(
                {
                    "policy_seed": int(payload["seed"]),
                    "task": task,
                    "condition": condition,
                    "sample_count": len(rows),
                    "sampling": "deterministic_evenly_spaced_training_cache_rows",
                    **error_metrics(predicted, targets, masks),
                }
            )
    return int(payload["seed"]), records


def mean_record(records: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "first_action_position_mae",
        "chunk_position_mae",
        "first_action_gripper_accuracy",
        "chunk_gripper_accuracy",
    )
    return {key: float(np.mean([record[key] for record in records])) for key in keys}


def phase_bottleneck(item: dict[str, Any]) -> str:
    if item["grasp_or_open_phase_reach_rate"] < 0.50:
        return "early_grasp_or_articulation_entry"
    if item["lift_or_transport_phase_reach_rate"] < 0.50:
        return "lift_or_transport_transition"
    if item["pre_release_or_contact_phase_reach_rate"] < 0.70:
        return "late_approach_or_contact_entry"
    if item["clean_success_rate"] < 0.30:
        return "terminal_precision_after_late_phase"
    return "no_single_phase_bottleneck_at_gate_resolution"


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Post-hoc actor mechanism audit",
        "",
        "This is a code-level reverse-engineering audit, not a new idea, model, hyperparameter search, actor gate, Track-B diagnostic probe, or deployable method. One input factor is masked at a time in frozen checkpoints. Teacher-forced errors use deterministic training-cache rows, so they establish representation dependence but do not by themselves establish rollout causality.",
        "",
        "## Matched clean-rollout comparison",
        "",
        "| Task | ACT success | Weak horizon-1 success | Difference | Seed range | Phase bottleneck |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for task, item in summary["tasks"].items():
        weak = item["weak_reference_success_rate"]
        difference = item["act_minus_weak_success_rate"]
        lines.append(
            f"| {task} | {item['act_clean_success_rate']:.3f} | "
            f"{weak if weak is not None else 'n/a'} | "
            f"{difference if difference is not None else 'n/a'} | "
            f"{item['act_seed_success_range']:.3f} | {item['phase_bottleneck']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen-input ablations",
            "",
            "Positive Δ means the ablation increased deployed first-action position MAE relative to the full actor; negative Δ means it reduced that offline error.",
            "",
            "| Task | Repeat-current Δ | Zero-action-history Δ | Cyclic-task-ID Δ |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for task, item in summary["tasks"].items():
        effects = item["first_action_position_mae_delta_percent"]
        lines.append(
            f"| {task} | {effects['repeat_current_state']:.2f}% | "
            f"{effects['zero_action_history']:.2f}% | "
            f"{effects['cyclic_task_id']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Mechanistic reading",
            "",
        ]
    )
    for task, item in summary["tasks"].items():
        lines.append(f"- `{task}`: {item['mechanistic_reading']}")
    lines.extend(
        [
            "",
            "The matched weak comparison is available only for the two Stage-1 tasks evaluated with the same 10 initial states × 3 seeds and execution horizon 1. Other task improvements or degradations are intentionally left unclaimed. Cross-seed spread and phase reach are rollout evidence; masked-input error deltas are post-hoc localization evidence. Neither identifies a single causal component without retraining or adequately powered rollout ablations.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--qualification-summary", type=Path, required=True)
    parser.add_argument("--weak-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--samples-per-task", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    qualification = json.loads(args.qualification_summary.read_text())
    weak = json.loads(args.weak_summary.read_text())
    raw: list[dict[str, Any]] = []
    observed_seeds = []
    for seed in args.seeds:
        actual_seed, records = audit_checkpoint(
            args.checkpoint_dir / f"seed_{seed}.pt",
            cache_root=args.cache_root,
            samples_per_task=args.samples_per_task,
            device=device,
        )
        observed_seeds.append(actual_seed)
        raw.extend(records)
    if observed_seeds != args.seeds:
        raise ValueError(f"checkpoint seed mismatch: {observed_seeds}")
    tasks: dict[str, Any] = {}
    aggregate_rows: list[dict[str, Any]] = []
    for task in TASK_NAMES:
        by_condition: dict[str, dict[str, float]] = {}
        for condition in CONDITIONS:
            selected = [
                record
                for record in raw
                if record["task"] == task and record["condition"] == condition
            ]
            by_condition[condition] = mean_record(selected)
            aggregate_rows.append(
                {"task": task, "condition": condition, **by_condition[condition]}
            )
        full_error = by_condition["full"]["first_action_position_mae"]
        deltas = {
            condition: 100.0
            * (
                by_condition[condition]["first_action_position_mae"]
                / full_error
                - 1.0
            )
            for condition in CONDITIONS
            if condition != "full"
        }
        actor_item = qualification["tasks"][task]
        weak_rate = (
            weak.get("baseline", {})
            .get(task, {})
            .get("1", {})
            .get("success_rate")
        )
        act_rate = float(actor_item["clean_success_rate"])
        seed_rates = [
            float(actor_item["per_seed"][str(seed)]["success_rate"])
            for seed in args.seeds
        ]
        dominant = max(deltas, key=deltas.get)
        matched_change = None if weak_rate is None else act_rate - float(weak_rate)
        if matched_change is not None and matched_change > 0:
            comparison = (
                f"Matched clean success increased by {matched_change:.3f} as an architecture-and-training bundle; "
            )
        elif matched_change is not None and matched_change < 0:
            comparison = (
                f"Matched clean success decreased by {abs(matched_change):.3f} as an architecture-and-training bundle; "
            )
        elif matched_change is not None:
            comparison = "Matched clean success was unchanged; "
        else:
            comparison = "No exact Stage-1 weak baseline exists for this task; "
        reading = (
            comparison
            + f"the largest offline first-action dependence is `{dominant}` ({deltas[dominant]:+.2f}%). "
            + f"Rollouts localize the first coarse bottleneck to `{phase_bottleneck(actor_item)}` and the seed success range is {max(seed_rates) - min(seed_rates):.3f}. "
            + "This supports a bounded mechanism diagnosis, not a single-component causal claim."
        )
        tasks[task] = {
            "act_clean_success_rate": act_rate,
            "weak_reference_success_rate": weak_rate,
            "act_minus_weak_success_rate": matched_change,
            "act_per_seed_success_rates": seed_rates,
            "act_seed_success_range": max(seed_rates) - min(seed_rates),
            "phase_bottleneck": phase_bottleneck(actor_item),
            "teacher_forced_metrics": by_condition,
            "first_action_position_mae_delta_percent": deltas,
            "dominant_offline_input_dependency": dominant,
            "mechanistic_reading": reading,
        }
    summary = {
        "schema_version": 1,
        "status": "POST_HOC_NON_GATE_MECHANISM_AUDIT",
        "creates_new_idea": False,
        "trains_model_or_probe": False,
        "affects_actor_gate": False,
        "affects_task_or_perturbation_selection": False,
        "track_b_diagnostic_probe": False,
        "conditions": list(CONDITIONS),
        "checkpoint_seeds": list(args.seeds),
        "samples_per_task_seed": args.samples_per_task,
        "sample_scope": "deterministic_evenly_spaced_training_cache_rows",
        "limitations": [
            "Teacher-forced cache error is not rollout success.",
            "The weak comparison identifies the actor/training bundle, not an individual component.",
            "Only two tasks have an exact matched Stage-1 weak reference.",
            "No ablation was used for tuning, task selection, perturbation selection, or checkpoint gating.",
        ],
        "tasks": tasks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "raw_per_seed.json", raw)
    atomic_write_json(args.output_dir / "summary.json", summary)
    atomic_write_csv(
        args.output_dir / "teacher_forced_ablation_metrics.csv",
        aggregate_rows,
        list(aggregate_rows[0]),
    )
    atomic_write_text(args.output_dir / "report.md", render(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
