"""Experiment 3 — numerical feature embeddings ablation.

Gorishniy et al. 2022 ("On Embeddings for Numerical Features in
Tabular Deep Learning") argued that the *single biggest lever* for
tabular DL is how you encode each scalar numerical feature into a
vector before feeding it to an MLP. Test four encoders on the same MLP
backbone with two tasks:

  * pure-numerical regression (10000 × 20)
  * pure-numerical binary classification (10000 × 20)

Encoders:
  1. Linear — baseline; per-feature affine projection to d_emb.
  2. LinearReLU — same shape with a ReLU activation.
  3. Periodic — sin/cos frequency embeddings (their "PL" recipe).
  4. PiecewiseLinear (PLE) — feature-wise quantile bins, piecewise-linear interpolation.

All four feed a 2-layer rtdl MLP at d_block=128.
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "4")

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from _shared_datasets import make_pure_num_clf, make_pure_num_reg  # noqa: E402
from _shared_torch import (  # noqa: E402
    count_params,
    forward_single,
    make_loader,
    set_seeds,
    train_one,
)

RNG = 42
torch.set_num_threads(4)
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Encoder wrappers — each maps (B, n_features) -> (B, n_features * d_emb).
# ---------------------------------------------------------------------------


class EncoderBackbone(nn.Module):
    def __init__(self, encoder: nn.Module, encoder_flat_dim: int, *, d_out: int,
                 d_block: int = 128, dropout: float = 0.1, n_blocks: int = 2):
        super().__init__()
        import rtdl_revisiting_models as rtdl
        self.encoder = encoder
        self.body = rtdl.MLP(d_in=encoder_flat_dim, d_out=d_out,
                             n_blocks=n_blocks, d_block=d_block, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if z.ndim == 3:
            z = z.flatten(1)
        return self.body(z)


def build_encoder(name: str, X_train: np.ndarray, *, task: str, d_emb: int = 8):
    """Return (encoder_module, flat_dim) for a given name and dataset."""
    import rtdl_num_embeddings as rne
    n_features = X_train.shape[1]
    if name == "Linear":
        enc = rne.LinearEmbeddings(n_features=n_features, d_embedding=d_emb)
        return enc, n_features * d_emb
    if name == "LinearReLU":
        enc = rne.LinearReLUEmbeddings(n_features=n_features, d_embedding=d_emb)
        return enc, n_features * d_emb
    if name == "Periodic":
        # Periodic with `activation=True` (their "PL" recipe). `lite=False`
        # is required as of 0.0.12 because the default lite arg was dropped.
        enc = rne.PeriodicEmbeddings(
            n_features=n_features, d_embedding=d_emb,
            n_frequencies=24, frequency_init_scale=0.01,
            activation=True, lite=False,
        )
        return enc, n_features * d_emb
    if name == "PiecewiseLinear":
        # Pure quantile bins: y=None, regression=None, no tree_kwargs.
        X_t = torch.from_numpy(X_train).float()
        bins = rne.compute_bins(X_t, n_bins=16)
        enc = rne.PiecewiseLinearEmbeddings(
            bins=bins, d_embedding=d_emb, activation=True, version="B",
        )
        return enc, n_features * d_emb
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_encoder(encoder_name: str, task: str, X_tr, y_tr, X_te, y_te,
                  epochs: int = 20):
    set_seeds(RNG)
    enc, flat_dim = build_encoder(encoder_name, X_tr, task=task)
    d_out = 1
    model = EncoderBackbone(enc, flat_dim, d_out=d_out)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    xt_tr = torch.from_numpy(X_tr).float()
    xt_te = torch.from_numpy(X_te).float()
    if task == "clf":
        yt_tr = torch.from_numpy(y_tr).long()
        yt_te = torch.from_numpy(y_te).long()
        loss_fn = lambda p, t: nn.functional.binary_cross_entropy_with_logits(p[:, 0], t.float())
        metric_fn = lambda p, t: float(((p[:, 0] > 0).long() == t).float().mean())
    else:
        yt_tr = torch.from_numpy(y_tr).float()
        yt_te = torch.from_numpy(y_te).float()
        loss_fn = lambda p, t: ((p[:, 0] - t) ** 2).mean()
        metric_fn = lambda p, t: float(((p[:, 0] - t) ** 2).mean().sqrt())

    g = set_seeds(RNG)
    loader_tr = make_loader(xt_tr, yt_tr, batch_size=512, generator=g)
    loader_va = make_loader(xt_te, yt_te, batch_size=2048, shuffle=False)

    t0 = time.perf_counter()
    history = train_one(model, loader_tr, loader_va, opt, loss_fn,
                        epochs=epochs, forward=forward_single,
                        metric_fn=metric_fn, log_grad=False)
    fit_s = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        pred = model(xt_te).detach().numpy().reshape(-1)
    if task == "clf":
        metric = accuracy_score(y_te, (pred > 0).astype(int))
    else:
        metric = float(np.sqrt(mean_squared_error(y_te, pred)))
    return metric, fit_s, count_params(model), history


def run_task(task: str):
    if task == "clf":
        X, y = make_pure_num_clf(n_samples=10_000, n_features=20)
        stratify = y
    else:
        X, y = make_pure_num_reg(n_samples=10_000, n_features=20)
        stratify = None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=RNG, stratify=stratify)
    # Standardize features always; standardize y for regression too.
    sx = StandardScaler().fit(X_tr)
    X_tr = sx.transform(X_tr).astype(np.float32)
    X_te = sx.transform(X_te).astype(np.float32)
    if task == "reg":
        y_mu = float(y_tr.mean()); y_sd = float(y_tr.std())
        y_tr = ((y_tr - y_mu) / y_sd).astype(np.float32)
        y_te = ((y_te - y_mu) / y_sd).astype(np.float32)
    else:
        y_mu, y_sd = 0.0, 1.0

    rows = []
    histories = {}
    for name in ["Linear", "LinearReLU", "Periodic", "PiecewiseLinear"]:
        metric, fit_s, params, hist = train_encoder(name, task, X_tr, y_tr, X_te, y_te)
        if task == "reg":
            metric_unscaled = metric * y_sd
        else:
            metric_unscaled = metric
        rows.append({"encoder": name, "metric": metric_unscaled, "fit_s": fit_s, "params": params})
        histories[name] = hist
        m_name = "acc" if task == "clf" else "RMSE"
        print(f"  {name:<18} {m_name}={metric_unscaled:.4f}  fit={fit_s:5.1f}s  params={params:,}")
    return rows, histories


def save_table(clf_rows, reg_rows, path: Path):
    lines = []
    lines.append(f"{'CLASSIFICATION (10k × 20)':<60}")
    lines.append(f"{'encoder':<18}{'accuracy':>10}{'fit_s':>9}{'params':>14}")
    lines.append("-" * 52)
    for r in clf_rows:
        lines.append(f"{r['encoder']:<18}{r['metric']:>10.4f}{r['fit_s']:>9.1f}{r['params']:>14,d}")
    lines.append("")
    lines.append(f"{'REGRESSION (10k × 20)':<60}")
    lines.append(f"{'encoder':<18}{'RMSE':>10}{'fit_s':>9}{'params':>14}")
    lines.append("-" * 52)
    for r in reg_rows:
        lines.append(f"{r['encoder']:<18}{r['metric']:>10.4f}{r['fit_s']:>9.1f}{r['params']:>14,d}")
    path.write_text("\n".join(lines) + "\n")


def plot_curves(histories: dict, kind: str, path: Path, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, hist in histories.items():
        ax.plot(hist[kind], label=name, marker="o", markersize=3, alpha=0.85)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    t_start = time.perf_counter()
    print("=== pure_num_clf (10k × 20) ===")
    clf_rows, clf_hist = run_task("clf")
    print("=== pure_num_reg (10k × 20) ===")
    reg_rows, reg_hist = run_task("reg")

    save_table(clf_rows, reg_rows, OUT / "encoder_table.txt")
    plot_curves(clf_hist, "val_metric", OUT / "clf_curves.png", "val accuracy")
    plot_curves(reg_hist, "val_metric", OUT / "reg_curves.png", "val RMSE (standardized scale)")

    print(f"\nTotal wall: {time.perf_counter() - t_start:.1f}s")
    print("  outputs: encoder_table.txt, clf_curves.png, reg_curves.png")


if __name__ == "__main__":
    main()
