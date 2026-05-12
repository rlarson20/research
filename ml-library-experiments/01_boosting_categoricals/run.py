"""Experiment 1 — Boosting and categoricals: XGBoost vs LightGBM vs CatBoost.

The three libraries genuinely diverge on one thing: high-cardinality
categorical handling. Build a controlled dataset where categoricals are
*informative* by construction, then compare native cat (CatBoost), integer-
coded native cat (LightGBM), and one-hot (XGBoost) at varying cardinalities.
Tune the LightGBM pipeline with Optuna. Explain the winner with SHAP.
"""
from __future__ import annotations

import os
# Cap BLAS threads — boosters oversubscribe on small data otherwise.
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "4")

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import optuna
import shap

RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_DEPTH = 4

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def make_dataset(cardinality: int, n_extra_cats: int = 3):
    """Diabetes regression binarized on the median, plus synthetic categoricals
    that carry information about the target (an additive shift per category).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    data = load_diabetes()
    X_num = data.data
    y_cont = data.target
    n = X_num.shape[0]

    cats = []
    for i in range(n_extra_cats):
        codes = rng.integers(0, cardinality, size=n)
        # informative: each category gets a level effect; only the first cat
        # column has a strong effect to make signal-vs-noise interesting.
        level_effect = rng.normal(0, 30 if i == 0 else 8, size=cardinality)
        y_cont = y_cont + level_effect[codes]
        cats.append(codes)
    cats = np.stack(cats, axis=1)

    y = (y_cont > np.median(y_cont)).astype(np.int64)

    num_names = [f"num_{n}" for n in data.feature_names]
    cat_names = [f"cat_{i}" for i in range(n_extra_cats)]
    df = pd.DataFrame(np.concatenate([X_num, cats], axis=1),
                       columns=num_names + cat_names)
    for c in cat_names:
        df[c] = df[c].astype("int64")
    return df, y, num_names, cat_names


def fit_xgb(X_tr, y_tr, X_te, y_te, num_names, cat_names):
    # XGBoost requires one-hot for true categoricals (its `enable_categorical`
    # is experimental; one-hot is what most users actually do).
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(X_tr[cat_names])
    X_tr_oh = np.concatenate([X_tr[num_names].to_numpy(), ohe.transform(X_tr[cat_names])], axis=1)
    X_te_oh = np.concatenate([X_te[num_names].to_numpy(), ohe.transform(X_te[cat_names])], axis=1)
    t0 = time.perf_counter()
    clf = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        tree_method="hist", random_state=RANDOM_STATE,
        eval_metric="logloss", n_jobs=4, verbosity=0,
    )
    clf.fit(X_tr_oh, y_tr)
    fit_s = time.perf_counter() - t0
    p = clf.predict_proba(X_te_oh)[:, 1]
    return clf, p, fit_s, X_tr_oh, X_te_oh


def fit_lgb(X_tr, y_tr, X_te, y_te, num_names, cat_names):
    # LightGBM: pass cat column names directly; integer-coded.
    t0 = time.perf_counter()
    clf = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE, deterministic=True, force_row_wise=True,
        verbose=-1, n_jobs=4,
    )
    clf.fit(X_tr, y_tr, categorical_feature=cat_names)
    fit_s = time.perf_counter() - t0
    p = clf.predict_proba(X_te)[:, 1]
    return clf, p, fit_s


def fit_cat(X_tr, y_tr, X_te, y_te, num_names, cat_names):
    pool_tr = Pool(X_tr, y_tr, cat_features=cat_names)
    pool_te = Pool(X_te, cat_features=cat_names)
    t0 = time.perf_counter()
    clf = CatBoostClassifier(
        iterations=N_ESTIMATORS, depth=MAX_DEPTH,
        random_seed=RANDOM_STATE, verbose=False, thread_count=4,
    )
    clf.fit(pool_tr)
    fit_s = time.perf_counter() - t0
    p = clf.predict_proba(pool_te)[:, 1]
    return clf, p, fit_s


def bench(cardinality):
    df, y, num_names, cat_names = make_dataset(cardinality)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    out = []
    _, p, t = fit_lgb(X_tr, y_tr, X_te, y_te, num_names, cat_names)
    out.append(("LightGBM", log_loss(y_te, p), roc_auc_score(y_te, p), t))
    _, p, t = fit_cat(X_tr, y_tr, X_te, y_te, num_names, cat_names)
    out.append(("CatBoost", log_loss(y_te, p), roc_auc_score(y_te, p), t))
    _, p, t, _, _ = fit_xgb(X_tr, y_tr, X_te, y_te, num_names, cat_names)
    out.append(("XGBoost-OH", log_loss(y_te, p), roc_auc_score(y_te, p), t))
    return out


def main():
    print("[1] Sweep cardinality 5 → 80 across three libraries …")
    cards = [5, 20, 50, 80]
    sweep = {c: bench(c) for c in cards}
    for c, rows in sweep.items():
        print(f"  cardinality={c}")
        for name, ll, auc, t in rows:
            print(f"    {name:10s}  logloss={ll:.4f}  auc={auc:.4f}  fit={t:.2f}s")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    libs = ["LightGBM", "CatBoost", "XGBoost-OH"]
    for lib_i, lib in enumerate(libs):
        ll = [next(r[1] for r in sweep[c] if r[0] == lib) for c in cards]
        auc = [next(r[2] for r in sweep[c] if r[0] == lib) for c in cards]
        ax1.plot(cards, ll, "o-", label=lib)
        ax2.plot(cards, auc, "o-", label=lib)
    ax1.set_xlabel("categorical cardinality"); ax1.set_ylabel("test log-loss")
    ax1.set_title("Log-loss vs cardinality"); ax1.legend()
    ax2.set_xlabel("categorical cardinality"); ax2.set_ylabel("test AUC")
    ax2.set_title("AUC vs cardinality"); ax2.legend()
    fig.tight_layout(); fig.savefig(OUT / "cardinality_sweep.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for lib in libs:
        ts = [next(r[3] for r in sweep[c] if r[0] == lib) for c in cards]
        ax.plot(cards, ts, "o-", label=lib)
    ax.set_xlabel("categorical cardinality"); ax.set_ylabel("fit time (s)")
    ax.set_title("Training time vs cardinality"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "time_vs_cardinality.png", dpi=120); plt.close(fig)

    lines = ["cardinality library     logloss   auc      fit_s",
             "----------- -------     -------   ---      -----"]
    for c, rows in sweep.items():
        for name, ll, auc, t in rows:
            lines.append(f"{c:11d} {name:10s}  {ll:.4f}    {auc:.4f}   {t:.2f}")
    (OUT / "cardinality_table.txt").write_text("\n".join(lines) + "\n")

    print("\n[2] Optuna TPE on LightGBM (20 trials, MedianPruner) @ cardinality=50 …")
    df, y, num_names, cat_names = make_dataset(50)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=RANDOM_STATE)

    def objective(trial):
        params = dict(
            n_estimators=300,
            learning_rate=trial.suggest_float("lr", 1e-3, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 64),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 40),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            random_state=RANDOM_STATE, deterministic=True, force_row_wise=True,
            verbose=-1, n_jobs=4,
        )
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr2, y_tr2, categorical_feature=cat_names,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20, verbose=False)])
        p = clf.predict_proba(X_val)[:, 1]
        return log_loss(y_val, p)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=20, show_progress_bar=False)
    print(f"  best logloss={study.best_value:.4f}  params={study.best_params}")

    try:
        from optuna.visualization.matplotlib import plot_parallel_coordinate
        ax = plot_parallel_coordinate(study)
        fig = ax.get_figure(); fig.set_size_inches(9, 4)
        fig.tight_layout(); fig.savefig(OUT / "optuna_parallel.png", dpi=120); plt.close(fig)
    except Exception as e:
        print(f"  (parallel-coords plot skipped: {e})")

    # Refit best on full training, evaluate on held-out test.
    best_clf = lgb.LGBMClassifier(
        n_estimators=300, **study.best_params,
        random_state=RANDOM_STATE, deterministic=True, force_row_wise=True,
        verbose=-1, n_jobs=4)
    best_clf.fit(X_tr, y_tr, categorical_feature=cat_names)
    p_test = best_clf.predict_proba(X_te)[:, 1]
    tuned_ll = log_loss(y_te, p_test); tuned_auc = roc_auc_score(y_te, p_test)
    print(f"  tuned test  logloss={tuned_ll:.4f}  auc={tuned_auc:.4f}")

    print("\n[3] SHAP TreeExplainer on 100 test rows …")
    explainer = shap.TreeExplainer(best_clf)
    X_te_sample = X_te.sample(100, random_state=RANDOM_STATE).reset_index(drop=True)
    sv = explainer.shap_values(X_te_sample)
    if isinstance(sv, list):  # older shap returned list for binary
        sv = sv[1]

    plt.figure(figsize=(8, 5))
    shap.summary_plot(sv, X_te_sample, show=False, max_display=12)
    plt.tight_layout(); plt.savefig(OUT / "shap_beeswarm.png", dpi=120); plt.close()

    # Local waterfall for one row.
    plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap.Explanation(
        values=sv[0], base_values=explainer.expected_value,
        data=X_te_sample.iloc[0].values, feature_names=X_te_sample.columns.tolist()),
        max_display=10, show=False)
    plt.tight_layout(); plt.savefig(OUT / "shap_waterfall_example.png", dpi=120); plt.close()

    summary = "\n".join([
        f"sweep_cardinalities: {cards}",
        f"best_optuna_logloss_val: {study.best_value:.4f}",
        f"best_optuna_params: {study.best_params}",
        f"tuned_test_logloss: {tuned_ll:.4f}",
        f"tuned_test_auc: {tuned_auc:.4f}",
    ])
    (OUT / "winner.txt").write_text(summary + "\n")
    print("\n" + summary)


if __name__ == "__main__":
    main()
