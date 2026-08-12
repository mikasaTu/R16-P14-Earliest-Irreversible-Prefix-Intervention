from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import Normalization


class ChunkedBCMLP(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int = 95,
        action_dim: int = 7,
        chunk_length: int = 16,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, chunk_length * action_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.network(features)
        return output.reshape(-1, self.chunk_length, self.action_dim)

    def config(self) -> dict[str, int]:
        return {
            "feature_dim": self.feature_dim,
            "action_dim": self.action_dim,
            "chunk_length": self.chunk_length,
            "hidden_dim": self.hidden_dim,
        }


@torch.no_grad()
def predict_chunk(
    model: ChunkedBCMLP,
    normalization: Normalization,
    feature: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    normalized = (feature - normalization.feature_mean) / normalization.feature_std
    tensor = torch.from_numpy(normalized).to(device=device).unsqueeze(0)
    predicted = model(tensor).squeeze(0).cpu().numpy()
    actions = predicted * normalization.action_std + normalization.action_mean
    actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
    # LIBERO demonstrations use a near-binary gripper command.  MSE near the
    # transition can otherwise create an unsupported half-open command.
    actions[:, -1] = np.where(actions[:, -1] >= 0.0, 1.0, -1.0)
    return actions
