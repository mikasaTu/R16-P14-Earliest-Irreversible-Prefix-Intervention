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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "sampler": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    generator.set_state(state["sampler"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def save_complete_checkpoint(
    lineage_dir: Path,
    *,
    step: int,
    payload: dict[str, Any],
) -> Path:
    lineage_dir.mkdir(parents=True, exist_ok=True)
    final_dir = lineage_dir / f"step_{step:08d}"
    temp_dir = lineage_dir / f".step_{step:08d}.tmp.{os.getpid()}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    checkpoint_path = temp_dir / "checkpoint.pt"
    torch.save(payload, checkpoint_path)
    with checkpoint_path.open("rb") as file:
        os.fsync(file.fileno())
    marker = {
        "step": int(step),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
    }
    marker_path = temp_dir / ".complete.json"
    marker_path.write_text(json.dumps(marker, sort_keys=True) + "\n")
    with marker_path.open("rb") as file:
        os.fsync(file.fileno())
    if final_dir.exists():
        existing = validate_checkpoint(final_dir)
        if existing["checkpoint_sha256"] != marker["checkpoint_sha256"]:
            raise RuntimeError(f"conflicting complete checkpoint: {final_dir}")
        shutil.rmtree(temp_dir)
        return final_dir
    os.replace(temp_dir, final_dir)
    directory_fd = os.open(lineage_dir, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return final_dir


def validate_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    marker_path = checkpoint_dir / ".complete.json"
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    if not marker_path.is_file() or not checkpoint_path.is_file():
        raise ValueError(f"incomplete checkpoint: {checkpoint_dir}")
    marker = json.loads(marker_path.read_text())
    actual = _sha256(checkpoint_path)
    if marker.get("checkpoint_sha256") != actual:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint_dir}")
    return marker


def find_latest_complete(lineage_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    if not lineage_dir.is_dir():
        return None
    for path in lineage_dir.glob("step_*"):
        if not path.is_dir():
            continue
        try:
            marker = validate_checkpoint(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        candidates.append((int(marker["step"]), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def load_checkpoint(checkpoint_dir: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    validate_checkpoint(checkpoint_dir)
    return torch.load(
        checkpoint_dir / "checkpoint.pt", map_location=map_location, weights_only=False
    )
