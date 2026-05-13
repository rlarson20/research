"""Experiment 1 — Headline shootout: boosting vs deep learning.

Five models head-to-head on three datasets:

  * synthetic mixed-type binary classification (~20k rows)
  * synthetic mixed-type regression (~20k rows)
  * load_digits (1797 × 64) — the small-data control

Models: LightGBM, CatBoost (boosting controls); TabNet, hand-rolled
MLP, FT-Transformer (DL contenders). Tracks accuracy / RMSE, wall-clock,
and parameter count. Establishes the headline finding for the project.
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
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from _shared_datasets import make_mixed_clf, make_mixed_reg  # noqa: E402
from _shared_torch import (  # noqa: E402
    acc_from_logits,
    count_params,
    forward_cont_cat,
    forward_single,
    make_loader,
    rmse,
    set_seeds,
    train_one,
)

RNG = 42
torch.set_num_threads(4)
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
CB_DIR = OUT / "_catboost_info"


# ---------------------------------------------------------------------------
# Hand-rolled MLP for mixed-type input (cont + cat embeddings).
# ---------------------------------------------------------------------------


class MixedMLP(nn.Module):
    def __init__(self, n_cont: int, cat_cardinalities: list[int], d_out: int = 1,
                 hidden: int = 128, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.ModuleList(
            [nn.Embedding(c, min(16, max(2, c // 2))) for c in cat_cardinalities]
        )
        emb_total = sum(min(16, max(2, c // 2)) for c in cat_cardinalities)
        d_in = n_cont + emb_total
        layers: list[nn.Module] = []
        for i in range(depth):
            din = d_in if i == 0 else hidden
            layers += [
                nn.Linear(din, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor | None) -> torch.Tensor:
        parts = [x_cont]
        if x_cat is not None and len(self.emb) > 0:
            for j, emb in enumerate(self.emb):
                parts.append(emb(x_cat[:, j]))
        x = torch.cat(parts, dim=1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Runners — each takes (X_tr, y_tr, X_te, y_te, *aux) and returns
# (metric, fit_seconds, n_params).
# ---------------------------------------------------------------------------


def run_lgbm(task: str, X_tr, y_tr, X_te, y_te, cat_idxs):
    import lightgbm as lgb

    cls = lgb.LGBMClassifier if task == "clf" else lgb.LGBMRegressor
    m = cls(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        deterministic=True,
        force_row_wise=True,
        n_jobs=4,
        verbose=-1,
        random_state=RNG,
    )
    t0 = time.perf_counter()
    if cat_idxs:
        m.fit(X_tr, y_tr, categorical_feature=cat_idxs)
    else:
        m.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    pred = m.predict(X_te)
    metric = accuracy_score(y_te, pred) if task == "clf" else float(np.sqrt(mean_squared_error(y_te, pred)))
    n_params = int(m.booster_.num_trees() * m.booster_.params.get("num_leaves", 31))
    return metric, fit_s, n_params


def run_catboost(task: str, X_tr, y_tr, X_te, y_te, cat_features):
    """X_tr/X_te must be DataFrames if cat_features is non-empty
    (CatBoost rejects float ndarrays with cat_features set)."""
    from catboost import CatBoostClassifier, CatBoostRegressor

    cls = CatBoostClassifier if task == "clf" else CatBoostRegressor
    m = cls(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        random_seed=RNG,
        verbose=False,
        thread_count=4,
        train_dir=str(CB_DIR),
        allow_writing_files=False,
    )
    t0 = time.perf_counter()
    if cat_features:
        m.fit(X_tr, y_tr, cat_features=cat_features)
    else:
        m.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    pred = m.predict(X_te)
    if task == "clf":
        metric = accuracy_score(y_te, pred.reshape(-1).astype(int))
    else:
        metric = float(np.sqrt(mean_squared_error(y_te, pred.reshape(-1))))
    n_params = int(m.tree_count_ * 2 ** m.get_param("depth"))
    return metric, fit_s, n_params


def run_tabnet(task: str, X_tr, y_tr, X_te, y_te, cat_idxs, cat_dims):
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

    set_seeds(RNG)
    cls = TabNetClassifier if task == "clf" else TabNetRegressor
    m = cls(
        n_d=8, n_a=8, n_steps=3, gamma=1.3,
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=4,
        seed=RNG, device_name="cpu", verbose=0,
    )
    t0 = time.perf_counter()
    if task == "clf":
        m.fit(X_tr.astype(np.float32), y_tr.astype(np.int64),
              max_epochs=30, patience=8, batch_size=2048, virtual_batch_size=256)
    else:
        m.fit(X_tr.astype(np.float32), y_tr.astype(np.float32).reshape(-1, 1),
              max_epochs=30, patience=8, batch_size=2048, virtual_batch_size=256)
    fit_s = time.perf_counter() - t0
    pred = m.predict(X_te.astype(np.float32))
    if task == "clf":
        metric = accuracy_score(y_te, pred.reshape(-1))
    else:
        metric = float(np.sqrt(mean_squared_error(y_te, pred.reshape(-1))))
    n_params = sum(p.numel() for p in m.network.parameters())
    return metric, fit_s, n_params


def _mixed_loss(task: str):
    if task == "clf":
        return lambda p, t: nn.functional.binary_cross_entropy_with_logits(p[:, 0], t.float())
    return lambda p, t: ((p[:, 0] - t) ** 2).mean()


def run_mlp_mixed(task, X_cont_tr, X_cat_tr, y_tr, X_cont_te, X_cat_te, y_te, cat_cards):
    g = set_seeds(RNG)
    model = MixedMLP(X_cont_tr.shape[1], cat_cards, d_out=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    def to_t(x_cont, x_cat, y):
        xc = torch.from_numpy(x_cont).float()
        xa = torch.from_numpy(x_cat).long() if x_cat is not None else None
        yt = torch.from_numpy(y).float() if task == "reg" else torch.from_numpy(y).long()
        return xc, xa, yt

    xc_tr, xa_tr, yt_tr = to_t(X_cont_tr, X_cat_tr, y_tr)
    xc_te, xa_te, yt_te = to_t(X_cont_te, X_cat_te, y_te)
    loader_tr = make_loader((xc_tr, xa_tr), yt_tr, batch_size=1024, generator=g)
    loader_va = make_loader((xc_te, xa_te), yt_te, batch_size=4096, shuffle=False)

    t0 = time.perf_counter()
    train_one(
        model, loader_tr, loader_va, opt, _mixed_loss(task),
        epochs=15, forward=forward_cont_cat,
        metric_fn=acc_from_logits if task == "clf" else rmse,
        log_grad=False,
    )
    fit_s = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        pred = model(xc_te, xa_te).detach().cpu().numpy().reshape(-1)
    if task == "clf":
        metric = accuracy_score(y_te, (pred > 0).astype(int))
    else:
        metric = float(np.sqrt(mean_squared_error(y_te, pred)))
    return metric, fit_s, count_params(model)


def run_fttransformer(task, X_cont_tr, X_cat_tr, y_tr, X_cont_te, X_cat_te, y_te, cat_cards):
    import rtdl_revisiting_models as rtdl

    g = set_seeds(RNG)
    kw = rtdl.FTTransformer.get_default_kwargs(n_blocks=3)
    model = rtdl.FTTransformer(
        n_cont_features=X_cont_tr.shape[1],
        cat_cardinalities=cat_cards if cat_cards else [],
        d_out=1,
        **kw,
    )
    opt = rtdl.FTTransformer.make_default_optimizer(model)

    def to_t(x_cont, x_cat, y):
        xc = torch.from_numpy(x_cont).float()
        xa = torch.from_numpy(x_cat).long() if (x_cat is not None and len(cat_cards) > 0) else None
        yt = torch.from_numpy(y).float() if task == "reg" else torch.from_numpy(y).long()
        return xc, xa, yt

    xc_tr, xa_tr, yt_tr = to_t(X_cont_tr, X_cat_tr, y_tr)
    xc_te, xa_te, yt_te = to_t(X_cont_te, X_cat_te, y_te)
    if xa_tr is None:
        loader_tr = make_loader(xc_tr, yt_tr, batch_size=512, generator=g)
        loader_va = make_loader(xc_te, yt_te, batch_size=2048, shuffle=False)
        forward = lambda m, ins: m(ins[0], None)
    else:
        loader_tr = make_loader((xc_tr, xa_tr), yt_tr, batch_size=512, generator=g)
        loader_va = make_loader((xc_te, xa_te), yt_te, batch_size=2048, shuffle=False)
        forward = forward_cont_cat

    t0 = time.perf_counter()
    train_one(
        model, loader_tr, loader_va, opt, _mixed_loss(task),
        epochs=12, forward=forward,
        metric_fn=acc_from_logits if task == "clf" else rmse,
        log_grad=False,
    )
    fit_s = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        pred = model(xc_te, xa_te).detach().cpu().numpy().reshape(-1)
    if task == "clf":
        metric = accuracy_score(y_te, (pred > 0).astype(int))
    else:
        metric = float(np.sqrt(mean_squared_error(y_te, pred)))
    return metric, fit_s, count_params(model)


# ---------------------------------------------------------------------------
# Per-dataset orchestration
# ---------------------------------------------------------------------------


def split_mixed(df: pd.DataFrame, y: np.ndarray, num_names: list[str], cat_names: list[str]):
    df_tr, df_te, y_tr, y_te = train_test_split(
        df, y, test_size=0.2, random_state=RNG,
        stratify=y if y.dtype != np.float64 else None,
    )
    scaler = StandardScaler().fit(df_tr[num_names].values)
    Xc_tr = scaler.transform(df_tr[num_names].values).astype(np.float32)
    Xc_te = scaler.transform(df_te[num_names].values).astype(np.float32)
    Xa_tr = df_tr[cat_names].values.astype(np.int64) if cat_names else None
    Xa_te = df_te[cat_names].values.astype(np.int64) if cat_names else None
    # For LGBM/CB which want a single frame: concatenated frame retaining cat cols
    X_full_tr = df_tr.copy()
    X_full_te = df_te.copy()
    X_full_tr[num_names] = Xc_tr
    X_full_te[num_names] = Xc_te
    return X_full_tr, X_full_te, Xc_tr, Xc_te, Xa_tr, Xa_te, y_tr, y_te


def run_dataset_mixed(task: str, name: str):
    print(f"=== {name} ({task}) ===")
    if task == "clf":
        df, y, num_names, cat_names = make_mixed_clf(n_samples=20_000, seed=RNG)
    else:
        df, y, num_names, cat_names = make_mixed_reg(n_samples=20_000, seed=RNG)
    cat_cards = [int(df[c].nunique()) for c in cat_names]

    X_full_tr, X_full_te, Xc_tr, Xc_te, Xa_tr, Xa_te, y_tr, y_te = split_mixed(
        df, y, num_names, cat_names
    )

    # For regression: standardize y for the DL models (they share an MSE
    # loss whose gradient scales with |y|; without this their lr is wrong
    # by orders of magnitude). Boosters are unaffected, so they see raw y.
    if task == "reg":
        y_mu, y_sd = float(y_tr.mean()), float(y_tr.std())
        y_tr_dl = (y_tr - y_mu) / y_sd
        y_te_dl = (y_te - y_mu) / y_sd
    else:
        y_tr_dl, y_te_dl = y_tr, y_te
        y_mu, y_sd = 0.0, 1.0

    rows = []

    # LightGBM — pass DataFrame so categorical_feature names work
    metric, t, params = run_lgbm(task, X_full_tr, y_tr, X_full_te, y_te, cat_names)
    rows.append(("LightGBM", metric, t, params))
    print(f"  LightGBM        metric={metric:.4f}  fit={t:5.1f}s  params={params}")

    # CatBoost — pass DataFrame so dtypes survive
    metric, t, params = run_catboost(task, X_full_tr, y_tr, X_full_te, y_te, cat_names)
    rows.append(("CatBoost", metric, t, params))
    print(f"  CatBoost        metric={metric:.4f}  fit={t:5.1f}s  params={params}")

    # DL models train on standardized y; we de-scale predictions before
    # computing RMSE so the metric is comparable to the boosters.
    def descale(metric, _task=task):
        return metric * y_sd if _task == "reg" else metric

    # TabNet — needs ndarray with int columns left as integers
    cat_idx_in_full = [X_full_tr.columns.get_loc(c) for c in cat_names]
    X_full_arr_tr = X_full_tr.values.astype(np.float64)
    X_full_arr_te = X_full_te.values.astype(np.float64)
    metric, t, params = run_tabnet(task, X_full_arr_tr, y_tr_dl, X_full_arr_te, y_te_dl, cat_idx_in_full, cat_cards)
    metric = descale(metric)
    rows.append(("TabNet", metric, t, params))
    print(f"  TabNet          metric={metric:.4f}  fit={t:5.1f}s  params={params}")

    # MLP (mixed)
    metric, t, params = run_mlp_mixed(task, Xc_tr, Xa_tr, y_tr_dl, Xc_te, Xa_te, y_te_dl, cat_cards)
    metric = descale(metric)
    rows.append(("MLP", metric, t, params))
    print(f"  MLP             metric={metric:.4f}  fit={t:5.1f}s  params={params}")

    # FT-Transformer
    metric, t, params = run_fttransformer(task, Xc_tr, Xa_tr, y_tr_dl, Xc_te, Xa_te, y_te_dl, cat_cards)
    metric = descale(metric)
    rows.append(("FT-Transformer", metric, t, params))
    print(f"  FT-Transformer  metric={metric:.4f}  fit={t:5.1f}s  params={params}")

    return rows


def run_dataset_digits():
    print("=== digits (clf) ===")
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RNG, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    # On digits (10-class clf), use LGBM/CB multi-class natively; for the
    # neural models we use the multi-class head.
    rows = []

    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                            deterministic=True, force_row_wise=True,
                            n_jobs=4, verbose=-1, random_state=RNG)
    t0 = time.perf_counter()
    m.fit(X_tr, y_tr)
    t = time.perf_counter() - t0
    metric = accuracy_score(y_te, m.predict(X_te))
    params = int(m.booster_.num_trees() * 31)
    rows.append(("LightGBM", metric, t, params))
    print(f"  LightGBM        acc={metric:.4f}  fit={t:5.1f}s")

    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                           random_seed=RNG, verbose=False, thread_count=4,
                           train_dir=str(CB_DIR), allow_writing_files=False)
    t0 = time.perf_counter()
    m.fit(X_tr, y_tr)
    t = time.perf_counter() - t0
    metric = accuracy_score(y_te, m.predict(X_te).reshape(-1).astype(int))
    params = int(m.tree_count_ * 64)
    rows.append(("CatBoost", metric, t, params))
    print(f"  CatBoost        acc={metric:.4f}  fit={t:5.1f}s")

    from pytorch_tabnet.tab_model import TabNetClassifier
    set_seeds(RNG)
    m = TabNetClassifier(n_d=8, n_a=8, n_steps=3, seed=RNG, device_name="cpu", verbose=0)
    t0 = time.perf_counter()
    m.fit(X_tr, y_tr.astype(np.int64), max_epochs=30, patience=8, batch_size=256, virtual_batch_size=64)
    t = time.perf_counter() - t0
    metric = accuracy_score(y_te, m.predict(X_te).reshape(-1))
    params = sum(p.numel() for p in m.network.parameters())
    rows.append(("TabNet", metric, t, params))
    print(f"  TabNet          acc={metric:.4f}  fit={t:5.1f}s")

    # Hand-rolled MLP for 10-class digits
    g = set_seeds(RNG)
    n_in = X_tr.shape[1]
    mlp = nn.Sequential(
        nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 10),
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    xt_tr = torch.from_numpy(X_tr); yt_tr = torch.from_numpy(y_tr).long()
    xt_te = torch.from_numpy(X_te); yt_te = torch.from_numpy(y_te).long()
    loader_tr = make_loader(xt_tr, yt_tr, batch_size=64, generator=g)
    loader_va = make_loader(xt_te, yt_te, batch_size=256, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = lambda p, t: float((p.argmax(1) == t).float().mean())
    t0 = time.perf_counter()
    train_one(mlp, loader_tr, loader_va, opt, loss_fn,
              epochs=15, forward=forward_single, metric_fn=metric_fn, log_grad=False)
    t = time.perf_counter() - t0
    mlp.eval()
    with torch.no_grad():
        pred = mlp(xt_te).argmax(1).numpy()
    metric = accuracy_score(y_te, pred)
    params = count_params(mlp)
    rows.append(("MLP", metric, t, params))
    print(f"  MLP             acc={metric:.4f}  fit={t:5.1f}s")

    # FT-Transformer on pure-numerical digits (cat_cardinalities=[])
    import rtdl_revisiting_models as rtdl
    set_seeds(RNG)
    kw = rtdl.FTTransformer.get_default_kwargs(n_blocks=3)
    ftt = rtdl.FTTransformer(n_cont_features=n_in, cat_cardinalities=[], d_out=10, **kw)
    opt = rtdl.FTTransformer.make_default_optimizer(ftt)
    loader_tr2 = make_loader(xt_tr, yt_tr, batch_size=64, generator=g)
    loader_va2 = make_loader(xt_te, yt_te, batch_size=256, shuffle=False)
    forward = lambda m, ins: m(ins[0], None)
    t0 = time.perf_counter()
    train_one(ftt, loader_tr2, loader_va2, opt, loss_fn,
              epochs=12, forward=forward, metric_fn=metric_fn, log_grad=False)
    t = time.perf_counter() - t0
    ftt.eval()
    with torch.no_grad():
        pred = ftt(xt_te, None).argmax(1).numpy()
    metric = accuracy_score(y_te, pred)
    params = count_params(ftt)
    rows.append(("FT-Transformer", metric, t, params))
    print(f"  FT-Transformer  acc={metric:.4f}  fit={t:5.1f}s")

    return rows


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def save_table(rows, path: Path, metric_name: str):
    lines = [f"{'model':<18}{metric_name:>10}{'fit_s':>10}{'params':>14}"]
    lines.append("-" * 52)
    for model, metric, fit_s, params in rows:
        lines.append(f"{model:<18}{metric:>10.4f}{fit_s:>10.2f}{params:>14,d}")
    path.write_text("\n".join(lines) + "\n")
    print(f"  -> {path.name}")


def main():
    t_start = time.perf_counter()
    clf_rows = run_dataset_mixed("clf", "mixed_clf_20k")
    save_table(clf_rows, OUT / "results_clf.txt", "accuracy")

    reg_rows = run_dataset_mixed("reg", "mixed_reg_20k")
    save_table(reg_rows, OUT / "results_reg.txt", "RMSE")

    digits_rows = run_dataset_digits()
    save_table(digits_rows, OUT / "results_digits.txt", "accuracy")

    # Summary plot: normalized "winner score" per dataset×model
    models = [r[0] for r in clf_rows]
    clf_metric = np.array([r[1] for r in clf_rows])
    reg_metric = np.array([r[1] for r in reg_rows])
    digits_metric = np.array([r[1] for r in digits_rows])
    # For clf/digits higher is better; for reg lower is better -> invert via 1 - rmse/max
    reg_score = 1.0 - reg_metric / reg_metric.max()
    digits_score = digits_metric
    clf_score = clf_metric

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(models))
    w = 0.27
    ax.bar(x - w, clf_score, w, label="Mixed-clf 20k (acc)")
    ax.bar(x, digits_score, w, label="Digits (acc)")
    ax.bar(x + w, reg_score, w, label="Mixed-reg 20k (1 - RMSE/RMSE_max)")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("score (higher = better)")
    ax.set_title("Headline shootout — boosting vs deep learning")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "shootout_summary.png", dpi=120)
    plt.close(fig)

    # Wall-clock vs parameter count scatter
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"LightGBM": "#1f77b4", "CatBoost": "#ff7f0e", "TabNet": "#2ca02c",
              "MLP": "#d62728", "FT-Transformer": "#9467bd"}
    all_rows = [("clf", r) for r in clf_rows] + [("reg", r) for r in reg_rows] + [("digits", r) for r in digits_rows]
    markers = {"clf": "o", "reg": "s", "digits": "^"}
    for ds, (model, _metric, fit_s, params) in all_rows:
        ax.scatter(max(params, 1), fit_s, c=colors[model], marker=markers[ds],
                   s=80, edgecolor="black", linewidth=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("parameter count (log)")
    ax.set_ylabel("fit time, seconds (log)")
    ax.set_title("Wall-clock vs parameter count")
    handles_model = [plt.Line2D([0],[0], marker="o", linestyle="", color=c, label=m,
                                markeredgecolor="black", markersize=8) for m, c in colors.items()]
    handles_ds = [plt.Line2D([0],[0], marker=mk, linestyle="", color="gray", label=ds, markersize=8)
                  for ds, mk in markers.items()]
    ax.legend(handles=handles_model + handles_ds, loc="best", fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "wallclock_vs_params.png", dpi=120)
    plt.close(fig)
    print(f"  -> shootout_summary.png, wallclock_vs_params.png")

    print(f"\nTotal wall: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
