"""Experiment 6 — Robustness probe.

Experiment 5 showed DL beats CatBoost on clean synthetic data. But
the Grinsztajn 2022 thesis cites real-world quirks (outliers,
missingness) as the reason DL loses on real data. Test that
hypothesis directly: train on a clean 40k training set, then evaluate
on three flavors of corrupted test set.

  (i)   Gaussian noise on numerical features, σ ∈ {0.1, 0.5, 1.0} (std-units).
  (ii)  MCAR missingness, rate ∈ {0.05, 0.20, 0.50}.
        - CatBoost and LightGBM get NaN directly (native handling).
        - FT-Transformer must zero-fill (no native missing support).
  (iii) Mean shift on the 3 most important numerical features,
        μ ∈ {0.5, 1.0, 2.0} std-units (importance ranked by LightGBM).

Three models — CatBoost, LightGBM, FT-Transformer — fitted once on the
clean train set. Twenty-seven evaluation passes total.
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
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from _shared_datasets import corrupt, make_mixed_clf  # noqa: E402
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
# Data: same shuffle / split as experiment 5 so the comparison stays fair.
# ---------------------------------------------------------------------------


def build_data():
    df, y, num_names, cat_names = make_mixed_clf(n_samples=45_000, seed=RNG)
    cat_cards = [int(df[c].nunique()) for c in cat_names]
    rng = np.random.default_rng(RNG)
    order = rng.permutation(len(df))
    df = df.iloc[order].reset_index(drop=True)
    y = y[order]
    df_te = df.iloc[:5_000].reset_index(drop=True)
    y_te = y[:5_000]
    df_tr = df.iloc[5_000:45_000].reset_index(drop=True)
    y_tr = y[5_000:45_000]

    scaler = StandardScaler().fit(df_tr[num_names].values)
    Xc_tr = scaler.transform(df_tr[num_names].values).astype(np.float32)
    Xc_te_clean = scaler.transform(df_te[num_names].values).astype(np.float32)
    Xa_tr = df_tr[cat_names].values.astype(np.int64)
    Xa_te = df_te[cat_names].values.astype(np.int64)

    X_full_tr = df_tr.copy(); X_full_tr[num_names] = Xc_tr
    X_full_te = df_te.copy(); X_full_te[num_names] = Xc_te_clean

    return (X_full_tr, y_tr, X_full_te, y_te,
            Xc_tr, Xa_tr, Xc_te_clean, Xa_te,
            num_names, cat_names, cat_cards, scaler)


# ---------------------------------------------------------------------------
# Train (once each on clean data)
# ---------------------------------------------------------------------------


def fit_catboost(X_full_tr, y_tr, cat_names):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                           random_seed=RNG, verbose=False, thread_count=4,
                           train_dir=str(CB_DIR), allow_writing_files=False,
                           nan_mode="Min")  # explicit NaN policy
    t0 = time.perf_counter()
    m.fit(X_full_tr, y_tr, cat_features=cat_names)
    return m, time.perf_counter() - t0


def fit_lightgbm(X_full_tr, y_tr, cat_names):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                            deterministic=True, force_row_wise=True, n_jobs=4,
                            verbose=-1, random_state=RNG)
    t0 = time.perf_counter()
    m.fit(X_full_tr, y_tr, categorical_feature=cat_names)
    return m, time.perf_counter() - t0


def fit_fttransformer(Xc_tr, Xa_tr, y_tr, cat_cards):
    import rtdl_revisiting_models as rtdl
    g = set_seeds(RNG)
    kw = rtdl.FTTransformer.get_default_kwargs(n_blocks=2)
    model = rtdl.FTTransformer(
        n_cont_features=Xc_tr.shape[1],
        cat_cardinalities=cat_cards, d_out=1, **kw,
    )
    opt = rtdl.FTTransformer.make_default_optimizer(model)
    xc_tr = torch.from_numpy(Xc_tr); xa_tr = torch.from_numpy(Xa_tr)
    yt_tr = torch.from_numpy(y_tr)
    # Need a val loader for train_one — reuse a tiny slice of train
    n_val = 2_000
    loader_tr = make_loader((xc_tr[n_val:], xa_tr[n_val:]), yt_tr[n_val:],
                            batch_size=512, generator=g)
    loader_va = make_loader((xc_tr[:n_val], xa_tr[:n_val]), yt_tr[:n_val],
                            batch_size=2048, shuffle=False)
    loss_fn = lambda p, t: nn.functional.binary_cross_entropy_with_logits(p[:, 0], t.float())
    t0 = time.perf_counter()
    train_one(model, loader_tr, loader_va, opt, loss_fn,
              epochs=6,
              forward=lambda m, ins: m(ins[0], ins[1]),
              metric_fn=acc_from_logits, log_grad=False)
    return model, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Evaluation under corruption
# ---------------------------------------------------------------------------


def eval_catboost(m, X_full_te, y_te):
    return accuracy_score(y_te, m.predict(X_full_te).reshape(-1).astype(int))


def eval_lightgbm(m, X_full_te, y_te):
    return accuracy_score(y_te, m.predict(X_full_te))


def eval_ftt(model, Xc_te, Xa_te, y_te):
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xc_te), torch.from_numpy(Xa_te)).numpy().reshape(-1)
    return accuracy_score(y_te, (pred > 0).astype(int))


# ---------------------------------------------------------------------------
# Top-3 feature importance from LightGBM, for the shift corruption.
# ---------------------------------------------------------------------------


def lgbm_top_num_indices(m_lgbm, num_names: list[str], k: int = 3) -> list[int]:
    fi = m_lgbm.feature_importances_
    cols = m_lgbm.feature_name_
    pairs = [(cols[i], fi[i]) for i in range(len(cols)) if cols[i] in num_names]
    pairs.sort(key=lambda p: -p[1])
    top_names = [p[0] for p in pairs[:k]]
    return [num_names.index(n) for n in top_names], top_names


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main():
    t_start = time.perf_counter()
    (X_full_tr, y_tr, X_full_te, y_te,
     Xc_tr, Xa_tr, Xc_te_clean, Xa_te,
     num_names, cat_names, cat_cards, scaler) = build_data()

    print("Fitting models on clean train set ...")
    m_cb, t_cb = fit_catboost(X_full_tr, y_tr, cat_names)
    print(f"  CatBoost fit={t_cb:.1f}s")
    m_lgb, t_lgb = fit_lightgbm(X_full_tr, y_tr, cat_names)
    print(f"  LightGBM fit={t_lgb:.1f}s")
    m_ft, t_ft = fit_fttransformer(Xc_tr, Xa_tr, y_tr, cat_cards)
    print(f"  FT-T fit={t_ft:.1f}s")

    # Clean test accuracy as the baseline (corruption magnitude 0)
    clean = {
        "CatBoost": eval_catboost(m_cb, X_full_te, y_te),
        "LightGBM": eval_lightgbm(m_lgb, X_full_te, y_te),
        "FT-T": eval_ftt(m_ft, Xc_te_clean, Xa_te, y_te),
    }
    print(f"\nClean baseline: {clean}")

    top_num_idx, top_num_names = lgbm_top_num_indices(m_lgb, num_names, k=3)
    print(f"\nTop-3 numerical features (LightGBM importance): {top_num_names}")

    rows: list[dict] = []

    # (i) Gaussian noise on numericals
    for sigma in (0.1, 0.5, 1.0):
        Xc_te_n = corrupt(Xc_te_clean, "gauss", sigma, seed=RNG).astype(np.float32)
        # rebuild full df for boosters
        X_full_te_n = X_full_te.copy(); X_full_te_n[num_names] = Xc_te_n
        rows.append({"kind": "gauss", "magnitude": sigma,
                     "CatBoost": eval_catboost(m_cb, X_full_te_n, y_te),
                     "LightGBM": eval_lightgbm(m_lgb, X_full_te_n, y_te),
                     "FT-T": eval_ftt(m_ft, Xc_te_n, Xa_te, y_te)})
        print(f"  gauss σ={sigma}: {rows[-1]}")

    # (ii) MCAR missingness
    for rate in (0.05, 0.20, 0.50):
        # numericals get NaN; for FT-T, zero-fill the NaNs after the mask
        Xc_te_m = corrupt(Xc_te_clean, "mcar", rate, seed=RNG).astype(np.float32)
        # for boosters, the full DF with NaN
        X_full_te_m = X_full_te.copy(); X_full_te_m[num_names] = Xc_te_m
        # for FT-T, zero-fill
        Xc_te_m_zf = np.where(np.isnan(Xc_te_m), 0.0, Xc_te_m).astype(np.float32)
        rows.append({"kind": "mcar", "magnitude": rate,
                     "CatBoost": eval_catboost(m_cb, X_full_te_m, y_te),
                     "LightGBM": eval_lightgbm(m_lgb, X_full_te_m, y_te),
                     "FT-T": eval_ftt(m_ft, Xc_te_m_zf, Xa_te, y_te)})
        print(f"  mcar rate={rate}: {rows[-1]}")

    # (iii) Mean shift on top-3 numericals
    for mu in (0.5, 1.0, 2.0):
        Xc_te_s = corrupt(Xc_te_clean, "shift", mu, seed=RNG,
                          shift_cols=tuple(top_num_idx)).astype(np.float32)
        X_full_te_s = X_full_te.copy(); X_full_te_s[num_names] = Xc_te_s
        rows.append({"kind": "shift", "magnitude": mu,
                     "CatBoost": eval_catboost(m_cb, X_full_te_s, y_te),
                     "LightGBM": eval_lightgbm(m_lgb, X_full_te_s, y_te),
                     "FT-T": eval_ftt(m_ft, Xc_te_s, Xa_te, y_te)})
        print(f"  shift μ={mu}: {rows[-1]}")

    save_outputs(clean, rows)
    print(f"\nTotal wall: {time.perf_counter() - t_start:.1f}s")
    print("  outputs: degradation_table.txt, degradation_{noise,missing,shift}.png")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def save_outputs(clean: dict, rows: list[dict]):
    # table
    lines = [f"{'kind':<8}{'mag':>8}{'CatBoost':>12}{'LightGBM':>12}{'FT-T':>10}"
             f"{'Δ CB':>10}{'Δ LGB':>10}{'Δ FT-T':>10}"]
    lines.append("-" * 80)
    # clean
    lines.append(f"{'clean':<8}{'0.000':>8}{clean['CatBoost']:>12.4f}"
                 f"{clean['LightGBM']:>12.4f}{clean['FT-T']:>10.4f}"
                 f"{'0.000':>10}{'0.000':>10}{'0.000':>10}")
    for r in rows:
        lines.append(
            f"{r['kind']:<8}{r['magnitude']:>8.3f}"
            f"{r['CatBoost']:>12.4f}{r['LightGBM']:>12.4f}{r['FT-T']:>10.4f}"
            f"{r['CatBoost']-clean['CatBoost']:>+10.4f}"
            f"{r['LightGBM']-clean['LightGBM']:>+10.4f}"
            f"{r['FT-T']-clean['FT-T']:>+10.4f}"
        )
    (OUT / "degradation_table.txt").write_text("\n".join(lines) + "\n")

    # three panels
    for kind, fname in [("gauss", "degradation_noise.png"),
                        ("mcar", "degradation_missing.png"),
                        ("shift", "degradation_shift.png")]:
        subset = [r for r in rows if r["kind"] == kind]
        mags = [0.0] + [r["magnitude"] for r in subset]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model, color in [("CatBoost", "#ff7f0e"),
                             ("LightGBM", "#1f77b4"),
                             ("FT-T", "#9467bd")]:
            ys = [clean[model]] + [r[model] for r in subset]
            ax.plot(mags, ys, marker="o", label=model, color=color)
        kind_title = {"gauss": "Gaussian noise σ (std-units)",
                      "mcar": "MCAR missing rate",
                      "shift": "Mean shift μ on top-3 num features"}[kind]
        ax.set_xlabel(kind_title)
        ax.set_ylabel("test accuracy")
        ax.set_title(f"Robustness — {kind}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
