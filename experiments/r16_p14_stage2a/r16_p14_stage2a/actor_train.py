from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .actor_checkpoint import (
    capture_rng,
    latest_complete,
    restore_rng,
    save_complete,
)
from .actor_data import MultiTaskSampler
from .actor_model import ACTConfig, HistoryConditionedStateACT
from .io_utils import append_jsonl, atomic_write_json
from .settings import (
    ARTIFACT_ROOT,
    DEFAULT_CACHE_ROOT,
    DEFAULT_CKPT_ROOT,
    DEFAULT_LOG_ROOT,
    TASK_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES, default=list(TASK_NAMES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT / "actor")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.10)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_config(args: argparse.Namespace) -> ACTConfig:
    return ACTConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )


def main() -> None:
    args = parse_args()
    if args.save_interval == 1000:
        raise ValueError(
            "save_interval=1000 requires fail-closed milestone retention; "
            "Stage-2A is preregistered at 2000"
        )
    if tuple(args.tasks) != TASK_NAMES and args.steps > 3:
        raise ValueError("formal actor training must use the frozen six-task roster")
    seed_everything(args.seed)
    device = torch.device(args.device)
    sampler_data = MultiTaskSampler(args.tasks, args.cache_root)
    config = model_config(args)
    model = HistoryConditionedStateACT(config).to(device)
    parameter_count = model.parameter_count()
    if parameter_count > 10_000_000:
        raise ValueError(f"parameter budget exceeded: {parameter_count}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    sampling_generator = torch.Generator(device="cpu")
    sampling_generator.manual_seed(args.seed + 100_003)
    lineage = args.checkpoint_root / args.run_id / f"seed_{args.seed}"
    log_path = args.log_root / args.run_id / f"seed_{args.seed}.train.jsonl"
    complete_path = lineage / ".training_complete.json"
    training_contract = {
        "run_id": args.run_id,
        "tasks": list(args.tasks),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "save_interval": args.save_interval,
        "model_config": model.model_config(),
        "parameter_count": parameter_count,
        "task_uniform_sampling": True,
        "demo_budget_per_task": 50 if args.steps > 3 else 2,
    }
    start_step = 0
    latest = latest_complete(lineage)
    if latest is not None:
        payload = torch.load(latest / "checkpoint.pt", map_location=device, weights_only=False)
        if payload["training_contract"] != training_contract:
            raise ValueError("checkpoint training contract mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        restore_rng(payload["rng"], sampling_generator)
        start_step = int(payload["step"])
        print(
            f"AUTO_RESUME=1 RUN_ID={args.run_id} seed={args.seed} "
            f"checkpoint={latest} restored_step={start_step}",
            flush=True,
        )
    else:
        print(
            f"AUTO_RESUME=0 RUN_ID={args.run_id} seed={args.seed} checkpoint=none",
            flush=True,
        )
    model.train()
    started = time.time()
    last_loss = float("nan")
    last_position_loss = float("nan")
    last_gripper_loss = float("nan")
    for step in range(start_step + 1, args.steps + 1):
        states, histories, task_ids, targets, masks = sampler_data.sample(
            args.batch_size, sampling_generator
        )
        states = states.to(device, non_blocking=True)
        histories = histories.to(device, non_blocking=True)
        task_ids = task_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        predicted = model(states, histories, task_ids)
        position = F.smooth_l1_loss(
            predicted[..., :6], targets[..., :6], reduction="none"
        ).mean(dim=-1)
        gripper = F.smooth_l1_loss(
            predicted[..., 6], targets[..., 6], reduction="none"
        )
        position_loss = (position * masks).sum() / masks.sum().clamp_min(1.0)
        gripper_loss = (gripper * masks).sum() / masks.sum().clamp_min(1.0)
        loss = position_loss + gripper_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach().cpu())
        last_position_loss = float(position_loss.detach().cpu())
        last_gripper_loss = float(gripper_loss.detach().cpu())
        if not np.isfinite(last_loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        if step == 1 or step % 20 == 0 or step == args.steps:
            record = {
                "run_id": args.run_id,
                "seed": args.seed,
                "global_step": step,
                "loss": last_loss,
                "position_loss": last_position_loss,
                "gripper_loss": last_gripper_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - started,
                "device": str(device),
                "parameter_count": parameter_count,
            }
            append_jsonl(log_path, record)
            if step == 1:
                atomic_write_json(
                    args.log_root / args.run_id / f"seed_{args.seed}.first_real_work.json",
                    record,
                )
            print(
                f"ACT_TRAIN_STEP seed={args.seed} global_step={step} "
                f"loss={last_loss:.8f}",
                flush=True,
            )
        if step % args.save_interval == 0 or step == args.steps:
            checkpoint = save_complete(
                lineage,
                step,
                {
                    "step": step,
                    "training_contract": training_contract,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": capture_rng(sampling_generator),
                    "normalization": sampler_data.normalization.to_dict(),
                },
            )
            print(f"CHECKPOINT_COMPLETE step={step} path={checkpoint}", flush=True)
    portable = args.artifact_root / "checkpoints" / f"seed_{args.seed}.pt"
    portable.parent.mkdir(parents=True, exist_ok=True)
    temporary = portable.with_name(f".{portable.name}.tmp.{os.getpid()}")
    torch.save(
        {
            "schema_version": 1,
            "run_id": args.run_id,
            "seed": args.seed,
            "step": args.steps,
            "model_config": model.model_config(),
            "parameter_count": parameter_count,
            "model": model.state_dict(),
            "normalization": sampler_data.normalization.to_dict(),
            "task_names": list(TASK_NAMES),
            "training_contract": training_contract,
        },
        temporary,
    )
    os.replace(temporary, portable)
    result = {
        "run_id": args.run_id,
        "seed": args.seed,
        "step": args.steps,
        "loss": last_loss,
        "position_loss": last_position_loss,
        "gripper_loss": last_gripper_loss,
        "parameter_count": parameter_count,
        "portable_checkpoint": str(portable),
        "complete_state_checkpoint": str(latest_complete(lineage)),
    }
    atomic_write_json(complete_path, result)
    atomic_write_json(args.artifact_root / f"seed_{args.seed}.training_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
