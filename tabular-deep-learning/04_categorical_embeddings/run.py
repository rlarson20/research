"""Experiment 4 — Categorical-encoding ablation across cardinality.

Direct callback to `ml-library-experiments/01_boosting_categoricals/`,
which found CatBoost's ordered target statistics dominate
LightGBM (integer-coded) and XGBoost (one-hot) on high-cardinality
categoricals. Now ask: among the *DL-side* encodings, which one
matches CatBoost's behavior? Sweep cardinality ∈ {10, 50, 200} and
compare:

  * Learned embedding at d ∈ {1, 4, 16}
  * One-hot
  * Mean target encoding
  * K-fold target encoding (5-fold OOF means)
  * CatBoost native (the reference "ceiling")

DL backbone: the same MLP from experiment 1 with these encodings
applied to cat columns; numericals get a per-feature Linear embedding
(d_emb=8). All datasets are uniform-cardinality variants of the same
mixed-type generator.
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
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from _shared_datasets import make_cardinality_variant  # noqa: E402
from _shared_torch import (  # noqa: E402
    acc_from_logits,
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
CB_DIR = OUT / "_catboost_info"


# ---------------------------------------------------------------------------
# Encoding helpers — each returns (X_tr, X_te) as dense float ndarrays.
# ---------------------------------------------------------------------------


def encode_onehot(Xa_tr: np.ndarray, Xa_te: np.ndarray, card: int) -> tuple[np.ndarray, np.ndarray]:
    n_cat = Xa_tr.shape[1]
    out_tr = np.zeros((Xa_tr.shape[0], n_cat * card), dtype=np.float32)
    out_te = np.zeros((Xa_te.shape[0], n_cat * card), dtype=np.float32)
    for j in range(n_cat):
        out_tr[np.arange(Xa_tr.shape[0]), j * card + Xa_tr[:, j]] = 1.0
        out_te[np.arange(Xa_te.shape[0]), j * card + Xa_te[:, j]] = 1.0
    return out_tr, out_te


def encode_mean_target(Xa_tr: np.ndarray, Xa_te: np.ndarray, y_tr: np.ndarray,
                       card: int, smoothing: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Each cat column replaced by E[y | level] computed on the *train* set,
    with additive smoothing toward the global mean."""
    n_cat = Xa_tr.shape[1]
    global_mean = float(y_tr.mean())
    out_tr = np.zeros_like(Xa_tr, dtype=np.float32)
    out_te = np.zeros_like(Xa_te, dtype=np.float32)
    for j in range(n_cat):
        level_means = np.zeros(card, dtype=np.float32)
        level_counts = np.zeros(card, dtype=np.float32)
        for lvl in range(card):
            mask = Xa_tr[:, j] == lvl
            n = int(mask.sum())
            if n > 0:
                level_means[lvl] = (
                    (y_tr[mask].sum() + smoothing * global_mean) / (n + smoothing)
                )
            else:
                level_means[lvl] = global_mean
            level_counts[lvl] = n
        out_tr[:, j] = level_means[Xa_tr[:, j]]
        out_te[:, j] = level_means[Xa_te[:, j]]
    return out_tr, out_te


def encode_kfold_target(Xa_tr: np.ndarray, Xa_te: np.ndarray, y_tr: np.ndarray,
                        card: int, smoothing: float = 10.0,
                        n_splits: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """OOF k-fold mean-target encoding for the train set; full-train means
    applied to test set."""
    n_cat = Xa_tr.shape[1]
    global_mean = float(y_tr.mean())
    out_tr = np.zeros_like(Xa_tr, dtype=np.float32)
    out_te = np.zeros_like(Xa_te, dtype=np.float32)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RNG)
    for j in range(n_cat):
        for fold_tr, fold_va in kf.split(Xa_tr):
            for lvl in range(card):
                mask = Xa_tr[fold_tr, j] == lvl
                n = int(mask.sum())
                if n > 0:
                    enc = (y_tr[fold_tr][mask].sum() + smoothing * global_mean) / (n + smoothing)
                else:
                    enc = global_mean
                # apply to validation rows where this level appears
                va_mask = Xa_tr[fold_va, j] == lvl
                out_tr[fold_va[va_mask], j] = enc
        # full-train encoding for test rows
        for lvl in range(card):
            mask = Xa_tr[:, j] == lvl
            n = int(mask.sum())
            if n > 0:
                enc = (y_tr[mask].sum() + smoothing * global_mean) / (n + smoothing)
            else:
                enc = global_mean
            out_te[Xa_te[:, j] == lvl, j] = enc
    return out_tr, out_te


# ---------------------------------------------------------------------------
# DL backbone — Linear-encoded numericals + chosen cat encoding -> MLP
# ---------------------------------------------------------------------------


class EmbeddingMLP(nn.Module):
    def __init__(self, n_cont: int, cat_cardinalities: list[int],
                 cat_emb_dim: int, *, hidden: int = 128, depth: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        import rtdl_num_embeddings as rne
        self.num_enc = rne.LinearEmbeddings(n_features=n_cont, d_embedding=8)
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(c, cat_emb_dim) for c in cat_cardinalities]
        )
        d_in = n_cont * 8 + len(cat_cardinalities) * cat_emb_dim
        layers: list[nn.Module] = []
        for i in range(depth):
            din = d_in if i == 0 else hidden
            layers += [
                nn.Linear(din, hidden), nn.BatchNorm1d(hidden),
                nn.ReLU(), nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [self.num_enc(x_cont).flatten(1)]
        for j, emb in enumerate(self.cat_embs):
            parts.append(emb(x_cat[:, j]))
        return self.body(torch.cat(parts, dim=1))


class DenseMLP(nn.Module):
    """Backbone for encodings that produce a pre-flattened dense input
    (one-hot, mean-target, kfold-target). Numericals already encoded into the
    same tensor or fed via a separate Linear embedding for consistency."""

    def __init__(self, n_cont: int, dense_cat_dim: int,
                 *, hidden: int = 128, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        import rtdl_num_embeddings as rne
        self.num_enc = rne.LinearEmbeddings(n_features=n_cont, d_embedding=8)
        d_in = n_cont * 8 + dense_cat_dim
        layers: list[nn.Module] = []
        for i in range(depth):
            din = d_in if i == 0 else hidden
            layers += [
                nn.Linear(din, hidden), nn.BatchNorm1d(hidden),
                nn.ReLU(), nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, x_cont: torch.Tensor, x_dense: torch.Tensor) -> torch.Tensor:
        z = self.num_enc(x_cont).flatten(1)
        return self.body(torch.cat([z, x_dense], dim=1))


# ---------------------------------------------------------------------------
# Per-encoding trainers
# ---------------------------------------------------------------------------


def loss_fn(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits[:, 0], y.float())


def train_emb(card: int, emb_dim: int, *, Xc_tr, Xa_tr, y_tr, Xc_te, Xa_te, y_te):
    g = set_seeds(RNG)
    model = EmbeddingMLP(Xc_tr.shape[1], [card] * Xa_tr.shape[1], cat_emb_dim=emb_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    xc_tr = torch.from_numpy(Xc_tr); xa_tr = torch.from_numpy(Xa_tr).long(); yt_tr = torch.from_numpy(y_tr)
    xc_te = torch.from_numpy(Xc_te); xa_te = torch.from_numpy(Xa_te).long(); yt_te = torch.from_numpy(y_te)
    loader_tr = make_loader((xc_tr, xa_tr), yt_tr, batch_size=1024, generator=g)
    loader_va = make_loader((xc_te, xa_te), yt_te, batch_size=4096, shuffle=False)
    t0 = time.perf_counter()
    train_one(model, loader_tr, loader_va, opt, loss_fn,
              epochs=15, forward=lambda m, ins: m(ins[0], ins[1]),
              metric_fn=acc_from_logits, log_grad=False)
    fit_s = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        pred = model(xc_te, xa_te).numpy().reshape(-1)
    return accuracy_score(y_te, (pred > 0).astype(int)), fit_s, count_params(model)


def train_dense(card: int, *, Xc_tr, dense_tr, y_tr, Xc_te, dense_te, y_te):
    g = set_seeds(RNG)
    model = DenseMLP(Xc_tr.shape[1], dense_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    xc_tr = torch.from_numpy(Xc_tr); xd_tr = torch.from_numpy(dense_tr); yt_tr = torch.from_numpy(y_tr)
    xc_te = torch.from_numpy(Xc_te); xd_te = torch.from_numpy(dense_te); yt_te = torch.from_numpy(y_te)
    loader_tr = make_loader((xc_tr, xd_tr), yt_tr, batch_size=1024, generator=g)
    loader_va = make_loader((xc_te, xd_te), yt_te, batch_size=4096, shuffle=False)
    t0 = time.perf_counter()
    train_one(model, loader_tr, loader_va, opt, loss_fn,
              epochs=15, forward=lambda m, ins: m(ins[0], ins[1]),
              metric_fn=acc_from_logits, log_grad=False)
    fit_s = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        pred = model(xc_te, xd_te).numpy().reshape(-1)
    return accuracy_score(y_te, (pred > 0).astype(int)), fit_s, count_params(model)


def train_catboost(card: int, X_full_tr: pd.DataFrame, y_tr, X_full_te: pd.DataFrame,
                   y_te, cat_names: list[str]):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                           random_seed=RNG, verbose=False, thread_count=4,
                           train_dir=str(CB_DIR), allow_writing_files=False)
    t0 = time.perf_counter()
    m.fit(X_full_tr, y_tr, cat_features=cat_names)
    fit_s = time.perf_counter() - t0
    pred = m.predict(X_full_te).reshape(-1).astype(int)
    return accuracy_score(y_te, pred), fit_s, int(m.tree_count_ * 64)


# ---------------------------------------------------------------------------
# Per-cardinality orchestration
# ---------------------------------------------------------------------------


def run_card(card: int) -> tuple[dict, dict]:
    print(f"=== cardinality={card} ===")
    df, y, num_names, cat_names = make_cardinality_variant(card, n_samples=15_000)
    df_tr, df_te, y_tr, y_te = train_test_split(df, y, test_size=0.2,
                                                random_state=RNG, stratify=y)
    scaler = StandardScaler().fit(df_tr[num_names].values)
    Xc_tr = scaler.transform(df_tr[num_names].values).astype(np.float32)
    Xc_te = scaler.transform(df_te[num_names].values).astype(np.float32)
    Xa_tr = df_tr[cat_names].values.astype(np.int64)
    Xa_te = df_te[cat_names].values.astype(np.int64)
    X_full_tr = df_tr.copy(); X_full_tr[num_names] = Xc_tr
    X_full_te = df_te.copy(); X_full_te[num_names] = Xc_te

    rows = {}
    params = {}

    for d in (1, 4, 16):
        acc, t, p = train_emb(card, d, Xc_tr=Xc_tr, Xa_tr=Xa_tr, y_tr=y_tr,
                              Xc_te=Xc_te, Xa_te=Xa_te, y_te=y_te)
        rows[f"emb-d{d}"] = acc
        params[f"emb-d{d}"] = p
        print(f"  emb d={d:<3} acc={acc:.4f}  fit={t:5.1f}s  params={p:,}")

    onehot_tr, onehot_te = encode_onehot(Xa_tr, Xa_te, card)
    acc, t, p = train_dense(card, Xc_tr=Xc_tr, dense_tr=onehot_tr, y_tr=y_tr,
                            Xc_te=Xc_te, dense_te=onehot_te, y_te=y_te)
    rows["onehot"] = acc; params["onehot"] = p
    print(f"  onehot     acc={acc:.4f}  fit={t:5.1f}s  params={p:,}")

    mt_tr, mt_te = encode_mean_target(Xa_tr, Xa_te, y_tr.astype(np.float32), card)
    acc, t, p = train_dense(card, Xc_tr=Xc_tr, dense_tr=mt_tr, y_tr=y_tr,
                            Xc_te=Xc_te, dense_te=mt_te, y_te=y_te)
    rows["target-mean"] = acc; params["target-mean"] = p
    print(f"  target-mean acc={acc:.4f}  fit={t:5.1f}s  params={p:,}")

    kf_tr, kf_te = encode_kfold_target(Xa_tr, Xa_te, y_tr.astype(np.float32), card)
    acc, t, p = train_dense(card, Xc_tr=Xc_tr, dense_tr=kf_tr, y_tr=y_tr,
                            Xc_te=Xc_te, dense_te=kf_te, y_te=y_te)
    rows["target-kfold"] = acc; params["target-kfold"] = p
    print(f"  target-kfold acc={acc:.4f}  fit={t:5.1f}s  params={p:,}")

    acc, t, p = train_catboost(card, X_full_tr, y_tr, X_full_te, y_te, cat_names)
    rows["catboost"] = acc; params["catboost"] = p
    print(f"  catboost    acc={acc:.4f}  fit={t:5.1f}s  trees×leaves≈{p:,}")
    return rows, params


def save_outputs(by_card: dict[int, dict]):
    encoders = ["emb-d1", "emb-d4", "emb-d16", "onehot", "target-mean",
                "target-kfold", "catboost"]
    # encoding × cardinality matrix
    lines = [f"{'encoder':<14}" + "".join(f"{c:>10d}" for c in by_card)]
    lines.append("-" * (14 + 10 * len(by_card)))
    for enc in encoders:
        row = f"{enc:<14}"
        for c in by_card:
            row += f"{by_card[c]['rows'][enc]:>10.4f}"
        lines.append(row)
    (OUT / "encoding_matrix.txt").write_text("\n".join(lines) + "\n")

    # param-count matrix
    lines = [f"{'encoder':<14}" + "".join(f"{c:>14d}" for c in by_card)]
    lines.append("-" * (14 + 14 * len(by_card)))
    for enc in encoders:
        row = f"{enc:<14}"
        for c in by_card:
            row += f"{by_card[c]['params'][enc]:>14,d}"
        lines.append(row)
    (OUT / "param_count_table.txt").write_text("\n".join(lines) + "\n")

    # plot: cardinality vs accuracy, with CatBoost dashed reference
    fig, ax = plt.subplots(figsize=(9, 5))
    cards = list(by_card)
    style = {
        "emb-d1": ("solid", "#1f77b4"),
        "emb-d4": ("solid", "#2ca02c"),
        "emb-d16": ("solid", "#9467bd"),
        "onehot": ("dotted", "#d62728"),
        "target-mean": ("solid", "#ff7f0e"),
        "target-kfold": ("solid", "#8c564b"),
        "catboost": ("dashed", "black"),
    }
    for enc in encoders:
        ys = [by_card[c]["rows"][enc] for c in cards]
        ls, color = style[enc]
        ax.plot(cards, ys, marker="o", linestyle=ls, color=color, label=enc)
    ax.set_xscale("log")
    ax.set_xticks(cards); ax.set_xticklabels([str(c) for c in cards])
    ax.set_xlabel("cardinality")
    ax.set_ylabel("test accuracy")
    ax.set_title("Categorical encoding × cardinality (CatBoost = reference)")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cardinality_effect.png", dpi=120)
    plt.close(fig)


def main():
    t_start = time.perf_counter()
    by_card = {}
    for card in (10, 50, 200):
        rows, params = run_card(card)
        by_card[card] = {"rows": rows, "params": params}
    save_outputs(by_card)
    print(f"\nTotal wall: {time.perf_counter() - t_start:.1f}s")
    print("  outputs: encoding_matrix.txt, param_count_table.txt, cardinality_effect.png")


if __name__ == "__main__":
    main()
