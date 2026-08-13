from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .actor_data import Normalization
from .settings import (
    ACTION_DIM,
    ACTION_HISTORY,
    CHUNK_LENGTH,
    FEATURE_DIM,
    OBS_HISTORY,
    TASK_NAMES,
)


@dataclass(frozen=True)
class ACTConfig:
    state_dim: int = FEATURE_DIM
    action_dim: int = ACTION_DIM
    state_history: int = OBS_HISTORY
    action_history: int = ACTION_HISTORY
    chunk_length: int = CHUNK_LENGTH
    task_count: int = len(TASK_NAMES)
    d_model: int = 192
    nhead: int = 6
    encoder_layers: int = 3
    decoder_layers: int = 3
    dim_feedforward: int = 768
    dropout: float = 0.10


class HistoryConditionedStateACT(nn.Module):
    """Deterministic state-only Action Chunking Transformer.

    The model sees four state-observation tokens, three previously executed
    action tokens, and a task token. Sixteen learned action queries decode one
    action chunk. It has no RGB, world model, or perturbation metadata input.
    """

    def __init__(self, config: ACTConfig = ACTConfig()) -> None:
        super().__init__()
        self.config_value = config
        self.state_projection = nn.Linear(config.state_dim, config.d_model)
        self.action_projection = nn.Linear(config.action_dim, config.d_model)
        self.task_embedding = nn.Embedding(config.task_count, config.d_model)
        token_count = 1 + config.state_history + config.action_history
        self.history_position = nn.Parameter(torch.empty(token_count, config.d_model))
        self.history_type = nn.Embedding(3, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.action_queries = nn.Parameter(
            torch.empty(config.chunk_length, config.d_model)
        )
        self.query_position = nn.Parameter(
            torch.empty(config.chunk_length, config.d_model)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.action_head = nn.Linear(config.d_model, config.action_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.history_position, std=0.02)
        nn.init.normal_(self.action_queries, std=0.02)
        nn.init.normal_(self.query_position, std=0.02)

    def forward(
        self,
        state_history: torch.Tensor,
        action_history: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch = state_history.shape[0]
        task_token = self.task_embedding(task_ids).unsqueeze(1)
        state_tokens = self.state_projection(state_history)
        action_tokens = self.action_projection(action_history)
        tokens = torch.cat([task_token, state_tokens, action_tokens], dim=1)
        type_ids = torch.tensor(
            [0] + [1] * self.config_value.state_history + [2] * self.config_value.action_history,
            device=tokens.device,
        )
        tokens = tokens + self.history_position.unsqueeze(0) + self.history_type(type_ids).unsqueeze(0)
        memory = self.encoder(tokens)
        queries = (
            self.action_queries + self.query_position
        ).unsqueeze(0).expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded)

    def model_config(self) -> dict[str, int | float]:
        return asdict(self.config_value)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@torch.no_grad()
def predict_chunk(
    model: HistoryConditionedStateACT,
    normalization: Normalization,
    state_history: np.ndarray,
    action_history: np.ndarray,
    task_id: int,
    device: torch.device,
) -> np.ndarray:
    states = np.asarray(state_history, dtype=np.float32)
    actions = np.asarray(action_history, dtype=np.float32)
    states = (states - normalization.state_mean) / normalization.state_std
    actions = (actions - normalization.action_mean) / normalization.action_std
    predicted = model(
        torch.from_numpy(states).unsqueeze(0).to(device),
        torch.from_numpy(actions).unsqueeze(0).to(device),
        torch.tensor([task_id], dtype=torch.long, device=device),
    ).squeeze(0).cpu().numpy()
    predicted = predicted * normalization.action_std + normalization.action_mean
    predicted = np.clip(predicted, -1.0, 1.0).astype(np.float32)
    predicted[:, -1] = np.where(predicted[:, -1] >= 0.0, 1.0, -1.0)
    return predicted
