from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def capture_rng(generator: torch.Generator) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "sampler": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    generator.set_state(state["sampler"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def save_complete(lineage: Path, step: int, payload: dict[str, Any]) -> Path:
    lineage.mkdir(parents=True, exist_ok=True)
    final = lineage / f"step_{step:08d}"
    temporary = lineage / f".step_{step:08d}.tmp.{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    checkpoint = temporary / "checkpoint.pt"
    torch.save(payload, checkpoint)
    with checkpoint.open("rb") as stream:
        os.fsync(stream.fileno())
    marker = {
        "step": step,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    marker_path = temporary / ".complete.json"
    marker_path.write_text(json.dumps(marker, sort_keys=True) + "\n")
    with marker_path.open("rb") as stream:
        os.fsync(stream.fileno())
    if final.exists():
        existing = validate(final)
        if existing["checkpoint_sha256"] != marker["checkpoint_sha256"]:
            raise RuntimeError(f"conflicting checkpoint {final}")
        shutil.rmtree(temporary)
        return final
    os.replace(temporary, final)
    directory_fd = os.open(lineage, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return final


def validate(path: Path) -> dict[str, Any]:
    marker = json.loads((path / ".complete.json").read_text())
    if sha256_file(path / "checkpoint.pt") != marker["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    return marker


def latest_complete(lineage: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    if not lineage.is_dir():
        return None
    for path in lineage.glob("step_*"):
        try:
            marker = validate(path)
        except Exception:
            continue
        candidates.append((int(marker["step"]), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]
