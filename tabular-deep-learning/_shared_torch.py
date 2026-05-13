"""Shared PyTorch training helpers.

Adapted from `ml-library-experiments/02_pytorch_training_loop/run.py`.
Reusing this loop verbatim is intentional — the gradient-norm /
loss-curve instrumentation pattern is the project's existing
established convention.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seeds(seed: int = 42) -> torch.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def make_loader(
    X: torch.Tensor | tuple[torch.Tensor, ...],
    y: torch.Tensor,
    batch_size: int = 1024,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> DataLoader:
    if isinstance(X, tuple):
        ds = TensorDataset(*X, y)
    else:
        ds = TensorDataset(X, y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    metric_fn: Callable | None,
    forward: Callable[[nn.Module, list[torch.Tensor]], torch.Tensor],
) -> tuple[float, float]:
    """Returns (mean loss, metric).

    `metric_fn(y_pred, y_true) -> float`. If None, returns 0.0 for metric.
    `forward(model, batch_inputs)` -> y_pred logits/values. Used so the
    caller controls how multi-input batches are unpacked.
    """
    model.eval()
    losses = []
    preds_all, ys_all = [], []
    for batch in loader:
        *inputs, y = batch
        y_pred = forward(model, inputs)
        losses.append(float(loss_fn(y_pred, y).detach()))
        preds_all.append(y_pred.detach())
        ys_all.append(y.detach())
    y_pred_all = torch.cat(preds_all)
    y_all = torch.cat(ys_all)
    m = float(metric_fn(y_pred_all, y_all)) if metric_fn else 0.0
    return float(np.mean(losses)), m


def _grad_l2(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float((p.grad.detach() ** 2).sum())
    return total ** 0.5


def train_one(
    model: nn.Module,
    loader_tr: DataLoader,
    loader_va: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    epochs: int,
    forward: Callable[[nn.Module, list[torch.Tensor]], torch.Tensor],
    metric_fn: Callable | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    log_grad: bool = True,
    verbose: bool = False,
) -> dict[str, list[float]]:
    """Train ``model`` for ``epochs`` epochs.

    Returns history dict with keys: train_loss, val_loss, val_metric,
    grad_norm, lr. Same shape as ml-library-experiments/02.
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_metric": [],
        "grad_norm": [],
        "lr": [],
    }
    for epoch in range(epochs):
        model.train()
        ep_losses, ep_grad = [], []
        for batch in loader_tr:
            *inputs, y = batch
            optimizer.zero_grad(set_to_none=True)
            y_pred = forward(model, inputs)
            loss = loss_fn(y_pred, y)
            loss.backward()
            if log_grad:
                ep_grad.append(_grad_l2(model))
            optimizer.step()
            ep_losses.append(float(loss.detach()))
        if scheduler is not None:
            scheduler.step()
        val_loss, val_metric = evaluate(model, loader_va, loss_fn, metric_fn, forward)
        history["train_loss"].append(float(np.mean(ep_losses)))
        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_metric)
        history["grad_norm"].append(float(np.mean(ep_grad)) if ep_grad else 0.0)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        if verbose:
            print(
                f"  ep {epoch+1}/{epochs}  "
                f"tr_loss={history['train_loss'][-1]:.4f}  "
                f"va_loss={val_loss:.4f}  va_metric={val_metric:.4f}  "
                f"gnorm={history['grad_norm'][-1]:.3f}"
            )
    return history


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# Convenience forward closures for common layouts


def forward_single(model: nn.Module, inputs: list[torch.Tensor]) -> torch.Tensor:
    (x,) = inputs
    return model(x)


def forward_cont_cat(model: nn.Module, inputs: list[torch.Tensor]) -> torch.Tensor:
    x_cont, x_cat = inputs
    return model(x_cont, x_cat)


def forward_cont_only_ft(model: nn.Module, inputs: list[torch.Tensor]) -> torch.Tensor:
    # FT-Transformer with no cat columns expects x_cat=None
    (x_cont,) = inputs
    return model(x_cont, None)


# Common metric closures


def acc_from_logits(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Binary classification accuracy from raw logits of shape (N, 1) or (N,)."""
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0]
    preds = (y_pred > 0).long()
    return float((preds == y_true).float().mean())


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0]
    return float(((y_pred - y_true) ** 2).mean().sqrt())


if __name__ == "__main__":
    # Smoke test: tiny MLP on random data
    g = set_seeds(42)
    X = torch.randn(256, 8)
    y = (X.sum(1) > 0).long()
    loader_tr = make_loader(X, y, batch_size=32, generator=g)
    loader_va = make_loader(X, y, batch_size=64, shuffle=False)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = lambda p, t: nn.functional.binary_cross_entropy_with_logits(p[:, 0], t.float())
    h = train_one(model, loader_tr, loader_va, opt, loss_fn, epochs=3,
                  forward=forward_single, metric_fn=acc_from_logits, verbose=True)
    print("history keys:", list(h.keys()), "params:", count_params(model))
