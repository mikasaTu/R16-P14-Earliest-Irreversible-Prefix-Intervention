from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r16_p14_stage2b.io_utils import atomic_write_json, load_jsonl
from r16_p14_stage2b.runtime import ActorBundle

from .runtime import reconstruct_anchor, replay_failure_diagnostic


def select_event(path: Path, event_id: str) -> dict[str, Any]:
    matches = [row for row in load_jsonl(path) if row["event_id"] == event_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {event_id!r} in {path}, found {len(matches)}")
    return matches[0]


def run_probe(events_path: Path, event_id: str, device: str, rounds: int) -> dict[str, Any]:
    event = select_event(events_path, event_id)
    bundle = ActorBundle.load(int(event["actor_seed"]), device)
    attempts = []
    for order_slot in range(rounds):
        env, _, audit = reconstruct_anchor(event, bundle, capture_trace=True)
        try:
            attempts.append({
                "order_slot": order_slot,
                "passed": bool(audit["passed"]),
                "diagnostic": replay_failure_diagnostic(event, audit),
            })
        finally:
            env.close()
    return {
        "schema_version": 1,
        "event_id": event_id,
        "event_checkpoint_sha256": event["checkpoint_sha256"],
        "rounds": rounds,
        "all_passed": all(item["passed"] for item in attempts),
        "attempts": attempts,
        "runtime": {
            "uid": os.getuid(),
            "gid": os.getgid(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    payload = run_probe(args.events, args.event_id, args.device, args.rounds)
    atomic_write_json(args.output, payload)
    print(json.dumps({"event_id": args.event_id, "all_passed": payload["all_passed"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
