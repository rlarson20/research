"""Experiment 5 — Sample-size scaling: does DL catch boosting as n grows?

Grinsztajn et al. 2022 ("Why do tree-based models still outperform
deep learning on tabular data?") argued the answer is "no, even at
scale." Test their claim on our synthetic mixed-type clf generator:

  n_train ∈ {500, 2000, 10_000, 40_000}, fixed 5000-row test set

Three contenders: CatBoost (boosting control), FT-Transformer (depth=2,
the cheap-but-near-optimal config from experiment 2), TabNet
(another DL contender to keep the comparison honest).

Outputs the acc-vs-n curve and the wall-clock-vs-n curve. The headline
is whether any DL model crosses CatBoost's line as n grows.
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
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from _shared_datasets import make_mixed_clf  # noqa: E402
from _shared_torch import (  # noqa: E402
    acc_from_logits,
    count_params,
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
# Build pool
# ---------------------------------------------------------------------------


def build_pool(n_total: int = 45_000):
    df, y, num_names, cat_names = make_mixed_clf(n_samples=n_total, seed=RNG)
    cat_cards = [int(df[c].nunique()) for c in cat_names]
    # Deterministic shuffle: rng.permutation
    rng = np.random.default_rng(RNG)
    order = rng.permutation(len(df))
    df = df.iloc[order].reset_index(drop=True)
    y = y[order]

    n_test = 5_000
    df_te = df.iloc[:n_test]
    y_te = y[:n_test]
    df_pool = df.iloc[n_test:].reset_index(drop=True)
    y_pool = y[n_test:]
    return df_pool, y_pool, df_te, y_te, num_names, cat_names, cat_cards


def prep(df_tr, df_te, num_names, cat_names):
    scaler = StandardScaler().fit(df_tr[num_names].values)
    Xc_tr = scaler.transform(df_tr[num_names].values).astype(np.float32)
    Xc_te = scaler.transform(df_te[num_names].values).astype(np.float32)
    Xa_tr = df_tr[cat_names].values.astype(np.int64)
    Xa_te = df_te[cat_names].values.astype(np.int64)
    X_full_tr = df_tr.copy(); X_full_tr[num_names] = Xc_tr
    X_full_te = df_te.copy(); X_full_te[num_names] = Xc_te
    return Xc_tr, Xc_te, Xa_tr, Xa_te, X_full_tr, X_full_te


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def run_catboost(X_full_tr, y_tr, X_full_te, y_te, cat_names):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                           random_seed=RNG, verbose=False, thread_count=4,
                           train_dir=str(CB_DIR), allow_writing_files=False)
    t0 = time.perf_counter()
    m.fit(X_full_tr, y_tr, cat_features=cat_names)
    fit_s = time.perf_counter() - t0
    acc = accuracy_score(y_te, m.predict(X_full_te).reshape(-1).astype(int))
    return acc, fit_s, int(m.tree_count_ * 64)


def fttransformer_epochs(n_train: int) -> int:
    # Epoch budget that scales inversely with n_train (per plan).
    if n_train <= 1_000:
        return 30
    if n_train <= 3_000:
        return 20
    if n_train <= 15_000:
        return 12
    return 6  # n=40k


def run_fttransformer(Xc_tr, Xa_tr, y_tr, Xc_te, Xa_te, y_te, cat_cards):
    import rtdl_revisiting_models as rtdl
    g = set_seeds(RNG)
    kw = rtdl.FTTransformer.get_default_kwargs(n_blocks=2)  # exp-2's winner
    model = rtdl.FTTransformer(
        n_cont_features=Xc_tr.shape[1],
        cat_cardinalities=cat_cards, d_out=1, **kw,
    )
    opt = rtdl.FTTransformer.make_default_optimizer(model)
    xc_tr = torch.from_numpy(Xc_tr); xa_tr = torch.from_numpy(Xa_tr)
    yt_tr = torch.from_numpy(y_tr)
    xc_te = torch.from_numpy(Xc_te); xa_te = torch.from_numpy(Xa_te)
    yt_te = torch.from_numpy(y_te)
    loader_tr = make_loader((xc_tr, xa_tr), yt_tr, batch_size=512, generator=g)
    loader_va = make_loader((xc_te, xa_te), yt_te, batch_size=4096, shuffle=False)
    loss_fn = lambda p, t: nn.functional.binary_cross_entropy_with_logits(p[:, 0], t.float())

    epochs = fttransformer_epochs(len(y_tr))
    t0 = time.perf_counter()
    train_one(model, loader_tr, loader_va, opt, loss_fn,
              epochs=epochs,
              forward=lambda m, ins: m(ins[0], ins[1]),
              metric_fn=acc_from_logits, log_grad=False)
    fit_s = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        pred = model(xc_te, xa_te).numpy().reshape(-1)
    acc = accuracy_score(y_te, (pred > 0).astype(int))
    return acc, fit_s, count_params(model)


def run_tabnet(X_full_tr, y_tr, X_full_te, y_te, cat_idxs, cat_dims):
    from pytorch_tabnet.tab_model import TabNetClassifier
    set_seeds(RNG)
    m = TabNetClassifier(
        n_d=8, n_a=8, n_steps=3,
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=4,
        seed=RNG, device_name="cpu", verbose=0,
    )
    X_tr = X_full_tr.values.astype(np.float32)
    X_te = X_full_te.values.astype(np.float32)
    epochs = 30
    if len(y_tr) <= 1_000:
        bsz = 256; vbsz = 64
    elif len(y_tr) <= 3_000:
        bsz = 512; vbsz = 128
    else:
        bsz = 2048; vbsz = 256
    t0 = time.perf_counter()
    m.fit(X_tr, y_tr.astype(np.int64),
          max_epochs=epochs, patience=10, batch_size=bsz, virtual_batch_size=vbsz)
    fit_s = time.perf_counter() - t0
    acc = accuracy_score(y_te, m.predict(X_te).reshape(-1))
    return acc, fit_s, sum(p.numel() for p in m.network.parameters())


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main():
    t_start = time.perf_counter()
    df_pool, y_pool, df_te, y_te, num_names, cat_names, cat_cards = build_pool()

    results: dict[str, list[dict]] = {"CatBoost": [], "FT-Transformer": [], "TabNet": []}

    for n in (500, 2_000, 10_000, 40_000):
        print(f"=== n_train = {n} ===")
        df_tr = df_pool.iloc[:n].reset_index(drop=True)
        y_tr = y_pool[:n]
        Xc_tr, Xc_te, Xa_tr, Xa_te, X_full_tr, X_full_te = prep(
            df_tr, df_te, num_names, cat_names
        )

        acc, t, params = run_catboost(X_full_tr, y_tr, X_full_te, y_te, cat_names)
        results["CatBoost"].append({"n": n, "acc": acc, "fit_s": t, "params": params})
        print(f"  CatBoost       acc={acc:.4f}  fit={t:5.1f}s  params={params:,}")

        acc, t, params = run_fttransformer(Xc_tr, Xa_tr, y_tr, Xc_te, Xa_te, y_te, cat_cards)
        results["FT-Transformer"].append({"n": n, "acc": acc, "fit_s": t, "params": params})
        print(f"  FT-Transformer acc={acc:.4f}  fit={t:5.1f}s  params={params:,}")

        cat_idxs = [X_full_tr.columns.get_loc(c) for c in cat_names]
        acc, t, params = run_tabnet(X_full_tr, y_tr, X_full_te, y_te, cat_idxs, cat_cards)
        results["TabNet"].append({"n": n, "acc": acc, "fit_s": t, "params": params})
        print(f"  TabNet         acc={acc:.4f}  fit={t:5.1f}s  params={params:,}")

    # save outputs
    save_outputs(results)
    print(f"\nTotal wall: {time.perf_counter() - t_start:.1f}s")
    print("  outputs: acc_vs_n.png, wallclock_vs_n.png, crossover_table.txt")


def save_outputs(results: dict[str, list[dict]]):
    # crossover table
    ns = [r["n"] for r in results["CatBoost"]]
    cb_acc = np.array([r["acc"] for r in results["CatBoost"]])
    ft_acc = np.array([r["acc"] for r in results["FT-Transformer"]])
    tn_acc = np.array([r["acc"] for r in results["TabNet"]])
    lines = [f"{'n':>8}{'CatBoost':>12}{'FT-T':>12}{'TabNet':>12}"
             f"{'Δ FT-CB':>12}{'Δ TN-CB':>12}"]
    lines.append("-" * 68)
    for i, n in enumerate(ns):
        lines.append(
            f"{n:>8d}{cb_acc[i]:>12.4f}{ft_acc[i]:>12.4f}{tn_acc[i]:>12.4f}"
            f"{ft_acc[i]-cb_acc[i]:>+12.4f}{tn_acc[i]-cb_acc[i]:>+12.4f}"
        )
    (OUT / "crossover_table.txt").write_text("\n".join(lines) + "\n")

    # acc-vs-n plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, color in [("CatBoost", "#ff7f0e"),
                         ("FT-Transformer", "#9467bd"),
                         ("TabNet", "#2ca02c")]:
        accs = [r["acc"] for r in results[model]]
        ax.plot(ns, accs, marker="o", label=model, color=color)
    ax.set_xscale("log")
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("n_train")
    ax.set_ylabel("test accuracy")
    ax.set_title("Does DL catch boosting as n grows?")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "acc_vs_n.png", dpi=120)
    plt.close(fig)

    # wallclock-vs-n plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, color in [("CatBoost", "#ff7f0e"),
                         ("FT-Transformer", "#9467bd"),
                         ("TabNet", "#2ca02c")]:
        ts = [r["fit_s"] for r in results[model]]
        ax.plot(ns, ts, marker="o", label=model, color=color)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("n_train")
    ax.set_ylabel("fit time, seconds (log)")
    ax.set_title("Wall-clock vs n_train")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "wallclock_vs_n.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
