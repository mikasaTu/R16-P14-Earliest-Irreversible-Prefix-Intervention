from __future__ import annotations

from pathlib import Path

import torch

from .checkpoint import find_latest_complete, load_checkpoint
from .data import Normalization
from .model import ChunkedBCMLP


def load_policy(
    checkpoint_root: Path,
    run_id: str,
    task_name: str,
    seed: int,
    device: torch.device,
) -> tuple[ChunkedBCMLP, Normalization, Path]:
    lineage = Path(checkpoint_root) / run_id / task_name / f"seed_{seed}"
    checkpoint_dir = find_latest_complete(lineage)
    if checkpoint_dir is None:
        raise FileNotFoundError(f"no complete checkpoint below {lineage}")
    payload = load_checkpoint(checkpoint_dir, map_location=device)
    model = ChunkedBCMLP(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    normalization = Normalization.from_dict(payload["normalization"])
    return model, normalization, checkpoint_dir
