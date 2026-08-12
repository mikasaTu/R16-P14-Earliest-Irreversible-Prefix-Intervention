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

from .checkpoint import (
    capture_rng_state,
    find_latest_complete,
    load_checkpoint,
    restore_rng_state,
    save_complete_checkpoint,
)
from .data import ChunkArrays, build_feature_cache, cache_path_for
from .io_utils import append_jsonl, atomic_write_json
from .model import ChunkedBCMLP
from .settings import (
    CHUNK_LENGTH,
    DEFAULT_CACHE_ROOT,
    DEFAULT_CKPT_ROOT,
    DEFAULT_LIBERO_CONFIG,
    DEFAULT_LOG_ROOT,
    DEVELOPMENT_TASKS,
    TRAIN_SEEDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(DEVELOPMENT_TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TRAIN_SEEDS))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", "r16_p14_libero_stage1_dev"))
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--libero-config", type=Path, default=DEFAULT_LIBERO_CONFIG)
    parser.add_argument("--demo-count", type=int, default=50)
    parser.add_argument("--max-states-per-demo", type=int)
    parser.add_argument("--chunk-length", type=int, default=CHUNK_LENGTH)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--exit-after-step", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(args: argparse.Namespace, task_name: str, seed: int) -> dict[str, object]:
    seed_everything(seed)
    device = torch.device(args.device)
    cache_path = cache_path_for(task_name, args.cache_root)
    if not cache_path.is_file():
        print(f"BUILD_CACHE task={task_name} path={cache_path}", flush=True)
        build_feature_cache(
            task_name,
            output_path=cache_path,
            demo_count=args.demo_count,
            max_states_per_demo=args.max_states_per_demo,
            config_dir=args.libero_config,
        )
    arrays = ChunkArrays(cache_path, chunk_length=args.chunk_length)
    features_cpu, targets_cpu, masks_cpu = arrays.normalized_tensors()
    model = ChunkedBCMLP(
        chunk_length=args.chunk_length,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    sampler = torch.Generator(device="cpu")
    sampler.manual_seed(seed + 100_003)

    lineage = args.checkpoint_root / args.run_id / task_name / f"seed_{seed}"
    log_path = args.log_root / args.run_id / task_name / f"seed_{seed}.train.jsonl"
    complete_path = lineage / ".training_complete.json"
    if complete_path.is_file():
        result = json.loads(complete_path.read_text())
        if int(result["step"]) != args.steps:
            raise ValueError(
                f"completed lineage step {result['step']} is incompatible with requested {args.steps}"
            )
        print(f"TRAINING_ALREADY_COMPLETE task={task_name} seed={seed} step={result['step']}", flush=True)
        return result

    latest = find_latest_complete(lineage)
    start_step = 0
    training_contract = {
        "task": task_name,
        "seed": seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "chunk_length": args.chunk_length,
        "hidden_dim": args.hidden_dim,
        "cache_path": str(cache_path),
    }
    if latest is not None:
        payload = load_checkpoint(latest, map_location=device)
        if payload.get("training_contract") != training_contract:
            raise ValueError(
                f"checkpoint contract mismatch at {latest}: "
                f"{payload.get('training_contract')} != {training_contract}"
            )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        restore_rng_state(payload["rng"], sampler)
        start_step = int(payload["step"])
        print(
            f"AUTO_RESUME=1 RUN_ID={args.run_id} task={task_name} seed={seed} "
            f"RESUME_FLAG=true checkpoint={latest} restored_step={start_step}",
            flush=True,
        )
    else:
        print(
            f"AUTO_RESUME=0 RUN_ID={args.run_id} task={task_name} seed={seed} "
            "RESUME_FLAG=false checkpoint=none",
            flush=True,
        )

    wandb_run = None
    if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_ENTITY"):
        import wandb

        wandb_run = wandb.init(
            project=os.environ.get("R16P14_WANDB_PROJECT", "r16-p14-libero-stage1"),
            entity=os.environ["WANDB_ENTITY"],
            id=f"{args.run_id}-{task_name}-seed-{seed}",
            name=f"{task_name}-seed-{seed}",
            group=args.run_id,
            resume="allow",
            config=training_contract,
            dir=os.environ.get("WANDB_DIR"),
        )

    model.train()
    last_loss = float("nan")
    started = time.time()
    for step in range(start_step + 1, args.steps + 1):
        indices = torch.randint(len(features_cpu), (args.batch_size,), generator=sampler)
        features = features_cpu[indices].to(device=device, non_blocking=True)
        targets = targets_cpu[indices].to(device=device, non_blocking=True)
        masks = masks_cpu[indices].to(device=device, non_blocking=True)
        predicted = model(features)
        elementwise = F.smooth_l1_loss(predicted, targets, reduction="none").mean(dim=-1)
        loss = (elementwise * masks).sum() / masks.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach().cpu())
        if not np.isfinite(last_loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {last_loss}")

        if step == 1 or step % 20 == 0 or step == args.steps:
            record = {
                "run_id": args.run_id,
                "task": task_name,
                "seed": seed,
                "global_step": step,
                "loss": last_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(log_path, record)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": last_loss,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/global_step": step,
                    },
                    step=step,
                )
            print(
                f"TRAIN_STEP task={task_name} seed={seed} global_step={step} loss={last_loss:.8f}",
                flush=True,
            )
            if step == 1:
                atomic_write_json(args.log_root / args.run_id / "first_real_work.json", record)

        if step % args.save_interval == 0 or step == args.steps:
            checkpoint_dir = save_complete_checkpoint(
                lineage,
                step=step,
                payload={
                    "run_id": args.run_id,
                    "task": task_name,
                    "seed": seed,
                    "step": step,
                    "model_config": model.config(),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": capture_rng_state(sampler),
                    "normalization": arrays.normalization.to_dict(),
                    "cache_path": str(cache_path),
                    "training_contract": training_contract,
                },
            )
            print(f"CHECKPOINT_COMPLETE step={step} path={checkpoint_dir}", flush=True)
            if args.exit_after_step == step:
                print(f"CONTROLLED_EXIT_AFTER_CHECKPOINT step={step}", flush=True)
                raise SystemExit(75)

    result = {
        "run_id": args.run_id,
        "task": task_name,
        "seed": seed,
        "step": args.steps,
        "loss": last_loss,
        "latest_checkpoint": str(find_latest_complete(lineage)),
    }
    atomic_write_json(complete_path, result)
    if wandb_run is not None:
        wandb_run.summary.update(result)
        wandb_run.finish()
    return result


def main() -> None:
    args = parse_args()
    if args.save_interval == 1000:
        raise ValueError(
            "save_interval=1000 requires the PAI milestone-plus-latest-three retention hook; "
            "this bounded Stage-1 run is preregistered at 2000"
        )
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch size must be positive")
    results = []
    for task_name in args.tasks:
        for seed in args.seeds:
            results.append(train_one(args, task_name, seed))
    summary_path = args.log_root / args.run_id / "training_summary.json"
    atomic_write_json(summary_path, {"run_id": args.run_id, "results": results})
    print(f"TRAINING_SUITE_COMPLETE summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
