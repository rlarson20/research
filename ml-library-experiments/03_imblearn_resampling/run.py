"""Experiment 3 — imbalanced-learn: when does explicit resampling beat class weighting?

Sklearn-experiment 4 found that on `make_classification(weights=[0.95, 0.05])`,
`class_weight="balanced"` is good enough for LogReg and threshold-tuning
is the biggest lever. But sklearn-Exp-4 never compared against
explicit resampling (SMOTE, ADASYN, RandomUnderSampler, …). This
experiment fills that gap.

Setup: severe imbalance (99/1) — harder than the 95/5 in Exp-4, where
class_weight has been shown to be sufficient. The hypothesis under test:
**at 99/1 severity, explicit minority-class generation (SMOTE/ADASYN)
catches up with or beats `class_weight="balanced"` because the loss
reweighting has too few minority examples to learn the boundary shape
from**.
"""
from __future__ import annotations

import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "4")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN

RANDOM_STATE = 42
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def make_data(n=10000, weights=(0.99, 0.01), class_sep=0.7):
    X, y = make_classification(
        n_samples=n, n_features=20, n_informative=10, n_redundant=5,
        n_clusters_per_class=2, weights=list(weights), class_sep=class_sep,
        flip_y=0.01, random_state=RANDOM_STATE,
    )
    return X, y


SAMPLERS = {
    "none": None,
    "class_weight": "class_weight",  # special-cased: no sampler, weight in clf
    "RandomOver": RandomOverSampler(random_state=RANDOM_STATE),
    "SMOTE": SMOTE(random_state=RANDOM_STATE, k_neighbors=5),
    "ADASYN": ADASYN(random_state=RANDOM_STATE, n_neighbors=5),
    "RandomUnder": RandomUnderSampler(random_state=RANDOM_STATE),
    "NearMiss-1": NearMiss(version=1),
    "SMOTE+Tomek": SMOTETomek(random_state=RANDOM_STATE),
    "SMOTE+ENN": SMOTEENN(random_state=RANDOM_STATE),
}


def fit_eval(name, sampler, X_tr, y_tr, X_te, y_te):
    clf_kwargs = dict(max_iter=2000, random_state=RANDOM_STATE)
    if sampler == "class_weight":
        clf = LogisticRegression(class_weight="balanced", **clf_kwargs)
        pipe = ImbPipeline([("scaler", StandardScaler()), ("clf", clf)])
    elif sampler is None:
        clf = LogisticRegression(**clf_kwargs)
        pipe = ImbPipeline([("scaler", StandardScaler()), ("clf", clf)])
    else:
        clf = LogisticRegression(**clf_kwargs)
        pipe = ImbPipeline([
            ("scaler", StandardScaler()), ("sampler", sampler), ("clf", clf),
        ])
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    yhat = (p >= 0.5).astype(int)

    # threshold-tuned F1 for the minority class
    pr, rc, th = precision_recall_curve(y_te, p)
    f1s = 2 * pr * rc / np.clip(pr + rc, 1e-9, None)
    best_idx = int(np.argmax(f1s))
    best_th = float(th[max(best_idx - 1, 0)]) if best_idx < len(th) else 0.5
    yhat_tuned = (p >= best_th).astype(int)

    return {
        "name": name,
        "auc": roc_auc_score(y_te, p),
        "ap": average_precision_score(y_te, p),
        "f1_at_0.5": f1_score(y_te, yhat),
        "prec_at_0.5": precision_score(y_te, yhat, zero_division=0),
        "rec_at_0.5": recall_score(y_te, yhat, zero_division=0),
        "f1_tuned": f1_score(y_te, yhat_tuned),
        "best_th": best_th,
        "p_proba": p,
        "yhat_0.5": yhat,
    }


def main():
    X, y = make_data()
    pos_rate = y.mean()
    print(f"Dataset: 10000×20  pos_rate={pos_rate:.3f}  "
          f"(n_minority={int(y.sum())})")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    print("\n[1] Resampler sweep (LogReg head, threshold=0.5 + tuned) …")
    results = []
    for name, samp in SAMPLERS.items():
        r = fit_eval(name, samp, X_tr, y_tr, X_te, y_te)
        results.append(r)
        print(f"  {name:14s} AUC={r['auc']:.4f}  AP={r['ap']:.4f}  "
              f"F1@0.5={r['f1_at_0.5']:.3f}  F1_tuned={r['f1_tuned']:.3f} "
              f"(th={r['best_th']:.3f})")

    # Compact table
    lines = ["sampler        AUC      AP       F1@0.5  prec@0.5 rec@0.5  F1_tuned  best_th",
             "-------        ---      --       ------  -------- -------  --------  -------"]
    for r in results:
        lines.append(
            f"{r['name']:14s} {r['auc']:.4f}   {r['ap']:.4f}   "
            f"{r['f1_at_0.5']:.3f}   {r['prec_at_0.5']:.3f}    "
            f"{r['rec_at_0.5']:.3f}    {r['f1_tuned']:.3f}     {r['best_th']:.3f}"
        )
    (OUT / "resampler_table.txt").write_text("\n".join(lines) + "\n")

    # --- bar chart: F1 @ 0.5 vs F1 tuned ---
    names = [r["name"] for r in results]
    f1_default = [r["f1_at_0.5"] for r in results]
    f1_tuned = [r["f1_tuned"] for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, f1_default, width=0.4, label="F1 @ threshold=0.5")
    ax.bar(x + 0.2, f1_tuned, width=0.4, label="F1 @ tuned threshold")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("F1 (minority class)")
    ax.set_title("Resampling vs class-weight: F1 default vs threshold-tuned")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "f1_default_vs_tuned.png", dpi=120); plt.close(fig)

    # --- precision-recall curves overlay (subset for clarity) ---
    showlist = ["none", "class_weight", "SMOTE", "ADASYN", "RandomUnder", "SMOTE+ENN"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results:
        if r["name"] not in showlist:
            continue
        pr, rc, _ = precision_recall_curve(y_te, r["p_proba"])
        ax.plot(rc, pr, label=f"{r['name']} (AP={r['ap']:.3f})")
    ax.axhline(pos_rate, color="gray", ls=":", label=f"chance={pos_rate:.3f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title(f"PR curves under different resampling (99/1 imbalance)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "pr_curves.png", dpi=120); plt.close(fig)

    # --- CV-stability check on top-3 (5 folds) ---
    print("\n[2] 5-fold CV stability for top 3 by AP …")
    top3 = sorted(results, key=lambda r: r["ap"], reverse=True)[:3]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for r in top3:
        samp = SAMPLERS[r["name"]]
        if samp == "class_weight":
            pipe = ImbPipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced",
                                            max_iter=2000, random_state=RANDOM_STATE)),
            ])
        elif samp is None:
            pipe = ImbPipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ])
        else:
            pipe = ImbPipeline([
                ("scaler", StandardScaler()), ("sampler", samp),
                ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ])
        ap = cross_val_score(pipe, X_tr, y_tr, cv=skf,
                              scoring="average_precision", n_jobs=-1)
        cv_rows.append((r["name"], ap.mean(), ap.std()))
        print(f"  {r['name']:14s} CV AP = {ap.mean():.4f} ± {ap.std():.4f}")

    # --- confusion matrix for winner @ tuned threshold ---
    winner = max(results, key=lambda r: r["f1_tuned"])
    yhat_winner = (winner["p_proba"] >= winner["best_th"]).astype(int)
    cm = confusion_matrix(y_te, yhat_winner)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.matshow(cm, cmap="Blues", alpha=0.3)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    fontsize=14, color="black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticklabels(["true 0", "true 1"])
    ax.set_title(f"Winner: {winner['name']}  th={winner['best_th']:.3f}\n"
                  f"F1={winner['f1_tuned']:.3f}")
    fig.tight_layout(); fig.savefig(OUT / "confusion_winner.png", dpi=120); plt.close(fig)

    summary = [
        f"dataset: make_classification 99/1, 10000 x 20",
        f"minority count: train={int(y_tr.sum())}  test={int(y_te.sum())}",
        f"winner by F1_tuned: {winner['name']}  th={winner['best_th']:.3f}  "
        f"F1={winner['f1_tuned']:.3f}",
        "",
        "Top 3 by AP, 5-fold CV:",
        *[f"  {n:14s} {m:.4f} ± {s:.4f}" for n, m, s in cv_rows],
    ]
    (OUT / "winner.txt").write_text("\n".join(summary) + "\n")
    print("\n" + "\n".join(summary))


if __name__ == "__main__":
    main()
