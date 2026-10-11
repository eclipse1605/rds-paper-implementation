"""Criteria, overall, and single-rating ReLU networks from the paper."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from src.log import get_logger

INIT_MEAN = 0.0
INIT_STD = 0.05
LEARNING_RATE = 0.001


def _init_linear(layer: nn.Linear) -> None:
    nn.init.normal_(layer.weight, mean=INIT_MEAN, std=INIT_STD)
    nn.init.normal_(layer.bias, mean=INIT_MEAN, std=INIT_STD)


def build_relu_mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    """Dense ReLU stack with a ReLU output, weights ~ Normal(0, 0.05)."""

    layers: list[nn.Module] = []
    last = in_dim
    for width in hidden:
        linear = nn.Linear(last, width)
        _init_linear(linear)
        layers.extend((linear, nn.ReLU()))
        last = width
    output = nn.Linear(last, out_dim)
    _init_linear(output)
    # Eq. 4 is ReLU(Wh + b). A negative bias has zero gradient, so that
    # output stays 0 for the whole run. Keep the Normal(0, 0.05) draw and
    # shift it up by 1 so every output unit starts active.
    with torch.no_grad():
        output.bias.add_(1.0)
    layers.extend((output, nn.ReLU()))
    return nn.Sequential(*layers)


class EmbeddingDNN(nn.Module):
    """User and item embeddings concatenated into a ReLU MLP."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int,
        hidden: tuple[int, ...],
        out_dim: int,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, mean=INIT_MEAN, std=INIT_STD)
        nn.init.normal_(self.item_emb.weight, mean=INIT_MEAN, std=INIT_STD)
        self.mlp = build_relu_mlp(emb_dim * 2, hidden, out_dim)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        features = torch.cat((self.user_emb(users), self.item_emb(items)), dim=-1)
        return self.mlp(features)


def criteria_dnn(n_users: int, n_items: int) -> EmbeddingDNN:
    """Section 4.3: embeddings 64, hidden [128, 64, 32, 16, 8], 7 outputs."""

    return EmbeddingDNN(n_users, n_items, emb_dim=64, hidden=(128, 64, 32, 16, 8), out_dim=7)


def single_dnn(n_users: int, n_items: int) -> EmbeddingDNN:
    """Section 4.4 ablation: embeddings 256, seven hidden layers, 1 output."""

    return EmbeddingDNN(
        n_users,
        n_items,
        emb_dim=256,
        hidden=(512, 256, 128, 64, 32, 16, 8),
        out_dim=1,
    )


class OverallDNN(nn.Module):
    """Section 4.3: z-scored criteria to one overall rating."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = build_relu_mlp(7, (64, 32, 16, 8), 1)

    def forward(self, criteria: torch.Tensor) -> torch.Tensor:
        return self.mlp(criteria)


def zscore_stats(criteria: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Population mean and standard deviation of the fit split. Zero std becomes 1."""

    mean = criteria.mean(axis=0).astype(np.float32)
    std = criteria.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, np.float32(1.0), std)
    return mean, std


def apply_zscore(criteria: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((criteria - mean) / std).astype(np.float32)


def _batches(tensors: tuple[torch.Tensor, ...], batch_size: int, shuffle: bool, rng: np.random.Generator):
    n = tensors[0].shape[0]
    order = np.arange(n)
    if shuffle:
        rng.shuffle(order)
    for start in range(0, n, batch_size):
        selected = order[start : start + batch_size]
        yield tuple(tensor[selected] for tensor in tensors)


def train_mae(
    model: nn.Module,
    train_tensors: tuple[torch.Tensor, ...],
    val_tensors: tuple[torch.Tensor, ...],
    forward,
    *,
    max_epochs: int = 30,
    patience: int = 5,
    batch_size: int = 512,
    learning_rate: float = LEARNING_RATE,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> float:
    """Adam + MAE. Restore the weights with the best validation MAE.

    `forward(model, batch_tensors)` returns predictions with the same shape as
    the last tensor, which is the target.
    """

    rng = np.random.default_rng(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0
    train_tensors = tuple(tensor.to(device) for tensor in train_tensors)
    val_tensors = tuple(tensor.to(device) for tensor in val_tensors)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in _batches(train_tensors, batch_size, shuffle=True, rng=rng):
            optimizer.zero_grad(set_to_none=True)
            prediction = forward(model, batch)
            loss = loss_fn(prediction, batch[-1])
            loss.backward()
            optimizer.step()
        val_mae = evaluate_mae(model, val_tensors, forward)
        get_logger().info("  epoch %02d  val_mae %.4f", epoch, val_mae)
        if val_mae < best_val:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return best_val


@torch.no_grad()
def evaluate_mae(model: nn.Module, tensors: tuple[torch.Tensor, ...], forward) -> float:
    model.eval()
    prediction = forward(model, tensors)
    return float(torch.mean(torch.abs(prediction - tensors[-1])).item())


@torch.no_grad()
def predict(model: nn.Module, tensors: tuple[torch.Tensor, ...], forward) -> np.ndarray:
    model.eval()
    return forward(model, tensors).detach().cpu().numpy()
