"""Experiment 2 — rtdl architecture sweep with instrumentation.

The three architectures from "Revisiting Deep Learning Models for
Tabular Data" (Gorishniy et al. 2021) — MLP, ResNet, FT-Transformer —
each trained with depth × dropout ablations on the *same* mixed-type
classification dataset used in experiment 1 (byte-identical via
shared seed=42).

Per-epoch training-loss / val-loss / val-accuracy / gradient L2 are
logged for every config so we can compare what's happening inside the
training loop, not just final accuracy. This is the direct extension
of `ml-library-experiments/02_pytorch_training_loop/` to the rtdl
architecture family.
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from _shared_datasets import make_mixed_clf  # noqa: E402
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


# ---------------------------------------------------------------------------
# Mixed-type adapters: rtdl MLP / ResNet take a single tensor of size d_in,
# but our dataset is cont+cat. We embed cats with a single `nn.Embedding`
# per column and concatenate with the standardized numericals, so MLP and
# ResNet are fed an (n_cont + n_cat × d_emb) flat vector. FTTransformer
# already accepts (x_cont, x_cat) natively.
# ---------------------------------------------------------------------------


class CatEmbedFlat(nn.Module):
    """Project (x_cont, x_cat) into a flat dense vector for MLP/ResNet."""
    def __init__(self, n_cont: int, cat_cardinalities: list[int], d_cat_emb: int = 8):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, d_cat_emb) for c in cat_cardinalities])
        self.d_out = n_cont + len(cat_cardinalities) * d_cat_emb

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [x_cont]
        for j, emb in enumerate(self.embs):
            parts.append(emb(x_cat[:, j]))
        return torch.cat(parts, dim=1)


class MlpFlat(nn.Module):
    def __init__(self, n_cont, cat_cardinalities, *, n_blocks, d_block, dropout):
        super().__init__()
        import rtdl_revisiting_models as rtdl
        self.flat = CatEmbedFlat(n_cont, cat_cardinalities)
        self.body = rtdl.MLP(d_in=self.flat.d_out, d_out=1,
                             n_blocks=n_blocks, d_block=d_block, dropout=dropout)

    def forward(self, x_cont, x_cat):
        return self.body(self.flat(x_cont, x_cat))


class ResNetFlat(nn.Module):
    def __init__(self, n_cont, cat_cardinalities, *, n_blocks, d_block, dropout):
        super().__init__()
        import rtdl_revisiting_models as rtdl
        self.flat = CatEmbedFlat(n_cont, cat_cardinalities)
        self.body = rtdl.ResNet(
            d_in=self.flat.d_out, d_out=1,
            n_blocks=n_blocks, d_block=d_block,
            d_hidden=None, d_hidden_multiplier=2.0,
            dropout1=dropout, dropout2=0.0,
        )

    def forward(self, x_cont, x_cat):
        return self.body(self.flat(x_cont, x_cat))


def make_fttransformer(n_cont, cat_cardinalities, *, n_blocks, dropout):
    import rtdl_revisiting_models as rtdl
    kw = rtdl.FTTransformer.get_default_kwargs(n_blocks=n_blocks)
    # Override the dropout knobs to the sweep value (defaults shape FT's
    # ffn/residual dropouts; setting all three keeps things consistent).
    kw["attention_dropout"] = dropout
    kw["ffn_dropout"] = dropout
    kw["residual_dropout"] = dropout * 0.5
    return rtdl.FTTransformer(
        n_cont_features=n_cont,
        cat_cardinalities=cat_cardinalities,
        d_out=1,
        **kw,
    )


def forward_mixed(model: nn.Module, inputs: list[torch.Tensor]) -> torch.Tensor:
    x_cont, x_cat = inputs
    return model(x_cont, x_cat)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_split():
    df, y, num_names, cat_names = make_mixed_clf(n_samples=20_000, seed=RNG)
    df_tr, df_te, y_tr, y_te = train_test_split(df, y, test_size=0.2,
                                                 random_state=RNG, stratify=y)
    scaler = StandardScaler().fit(df_tr[num_names].values)
    Xc_tr = scaler.transform(df_tr[num_names].values).astype(np.float32)
    Xc_te = scaler.transform(df_te[num_names].values).astype(np.float32)
    Xa_tr = df_tr[cat_names].values.astype(np.int64)
    Xa_te = df_te[cat_names].values.astype(np.int64)
    cat_cards = [int(df[c].nunique()) for c in cat_names]
    return Xc_tr, Xa_tr, y_tr, Xc_te, Xa_te, y_te, len(num_names), cat_cards


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def loss_fn(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits[:, 0], y.float())


def train_config(arch: str, depth: int, dropout: float, *, n_cont, cat_cards,
                 xc_tr, xa_tr, yt_tr, xc_te, xa_te, yt_te):
    g = set_seeds(RNG)
    if arch == "MLP":
        model = MlpFlat(n_cont, cat_cards, n_blocks=depth, d_block=128, dropout=dropout)
        lr = 1e-3
    elif arch == "ResNet":
        model = ResNetFlat(n_cont, cat_cards, n_blocks=depth, d_block=128, dropout=dropout)
        lr = 1e-3
    elif arch == "FT-Transformer":
        model = make_fttransformer(n_cont, cat_cards, n_blocks=depth, dropout=dropout)
        lr = 1e-4
    else:
        raise ValueError(arch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loader_tr = make_loader((xc_tr, xa_tr), yt_tr, batch_size=1024, generator=g)
    loader_va = make_loader((xc_te, xa_te), yt_te, batch_size=4096, shuffle=False)
    epochs = 10 if arch != "FT-Transformer" else 8
    t0 = time.perf_counter()
    history = train_one(
        model, loader_tr, loader_va, opt, loss_fn,
        epochs=epochs, forward=forward_mixed,
        metric_fn=acc_from_logits, log_grad=True,
    )
    fit_s = time.perf_counter() - t0
    return model, history, fit_s


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def plot_curves(records, kind: str, path: Path, ylabel: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=(kind == "val_metric"))
    for ax, arch in zip(axes, ["MLP", "ResNet", "FT-Transformer"]):
        configs = [r for r in records if r["arch"] == arch]
        for r in configs:
            label = f"depth={r['depth']}, p={r['dropout']}"
            ax.plot(r["history"][kind], label=label, alpha=0.85)
        ax.set_title(arch)
        ax.set_xlabel("epoch")
        if kind == "grad_norm":
            ax.set_yscale("log")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_table(records, path: Path):
    lines = [f"{'arch':<16}{'depth':>6}{'dropout':>9}{'val_acc':>10}{'fit_s':>9}{'params':>14}"]
    lines.append("-" * 64)
    for r in sorted(records, key=lambda x: (x["arch"], x["depth"], x["dropout"])):
        lines.append(
            f"{r['arch']:<16}{r['depth']:>6d}{r['dropout']:>9.2f}"
            f"{r['final_acc']:>10.4f}{r['fit_s']:>9.1f}{r['n_params']:>14,d}"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    t_start = time.perf_counter()
    Xc_tr, Xa_tr, y_tr, Xc_te, Xa_te, y_te, n_cont, cat_cards = load_split()

    # to tensors
    xc_tr = torch.from_numpy(Xc_tr)
    xa_tr = torch.from_numpy(Xa_tr)
    yt_tr = torch.from_numpy(y_tr)
    xc_te = torch.from_numpy(Xc_te)
    xa_te = torch.from_numpy(Xa_te)
    yt_te = torch.from_numpy(y_te)

    archs = ["MLP", "ResNet", "FT-Transformer"]
    depths = [2, 4]
    dropouts = [0.0, 0.1, 0.3]

    records = []
    for arch in archs:
        for depth in depths:
            for dropout in dropouts:
                tag = f"{arch} depth={depth} drop={dropout}"
                print(f"  training {tag} ...")
                model, history, fit_s = train_config(
                    arch, depth, dropout,
                    n_cont=n_cont, cat_cards=cat_cards,
                    xc_tr=xc_tr, xa_tr=xa_tr, yt_tr=yt_tr,
                    xc_te=xc_te, xa_te=xa_te, yt_te=yt_te,
                )
                model.eval()
                with torch.no_grad():
                    pred = model(xc_te, xa_te).detach().numpy().reshape(-1)
                acc = accuracy_score(y_te, (pred > 0).astype(int))
                n_params = count_params(model)
                records.append({
                    "arch": arch, "depth": depth, "dropout": dropout,
                    "final_acc": acc, "fit_s": fit_s, "n_params": n_params,
                    "history": history,
                })
                print(f"    -> val_acc={acc:.4f}  fit={fit_s:.1f}s  params={n_params:,}")

    save_table(records, OUT / "arch_table.txt")
    plot_curves(records, "val_metric", OUT / "training_curves.png", "val accuracy")
    plot_curves(records, "grad_norm", OUT / "grad_norms.png", "gradient L2 (log)")

    print(f"\nTotal wall: {time.perf_counter() - t_start:.1f}s")
    print(f"  outputs: arch_table.txt, training_curves.png, grad_norms.png")


if __name__ == "__main__":
    main()
