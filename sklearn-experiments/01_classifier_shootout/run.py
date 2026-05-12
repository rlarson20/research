"""Experiment 1 — Classifier shootout + interpretability.

Six classifiers compared via 5-fold stratified CV on the breast-cancer
dataset, top-2 tuned with GridSearchCV, winner explained with permutation
importance, partial dependence, and a learning curve.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def make_pipe(estimator):
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def cv_table(X, y, skf):
    models = {
        "LogReg": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "GradBoost": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVC-rbf": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=RANDOM_STATE),
    }
    rows = []
    for name, est in models.items():
        pipe = make_pipe(est)
        acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        f1 = cross_val_score(pipe, X, y, cv=skf, scoring="f1", n_jobs=-1)
        rows.append((name, acc.mean(), acc.std(), f1.mean(), f1.std()))
        print(f"  {name:13s} acc={acc.mean():.4f}±{acc.std():.4f}  f1={f1.mean():.4f}±{f1.std():.4f}")
    return rows, models


def write_cv_table(rows):
    lines = [
        "model            acc_mean  acc_std   f1_mean   f1_std",
        "-----            --------  -------   -------   ------",
    ]
    for name, a_m, a_s, f_m, f_s in rows:
        lines.append(f"{name:16s} {a_m:.4f}    {a_s:.4f}    {f_m:.4f}    {f_s:.4f}")
    (OUT / "cv_table.txt").write_text("\n".join(lines) + "\n")


GRIDS = {
    "LogReg": {"clf__C": [0.1, 1.0, 10.0]},
    "RandomForest": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 6, 12],
    },
    "GradBoost": {
        "clf__n_estimators": [100, 300],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [2, 3],
    },
    "SVC-rbf": {"clf__C": [0.5, 1.0, 5.0], "clf__gamma": ["scale", 0.05]},
    "KNN": {"clf__n_neighbors": [3, 5, 7, 11]},
    "MLP": {"clf__alpha": [1e-4, 1e-3], "clf__learning_rate_init": [1e-3, 5e-3]},
}


def tune_and_eval(name, pipe, X_train, y_train, X_test, y_test, skf):
    grid = GridSearchCV(pipe, GRIDS[name], cv=skf, scoring="f1", n_jobs=-1, refit=True)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1] if hasattr(grid, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")
    print(f"    tuned {name:13s} best={grid.best_params_}")
    print(f"      test acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}")
    return grid, acc, f1, auc


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    print(f"Dataset: breast_cancer  X={X.shape}  pos-rate={(y == 1).mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n[1] 5-fold CV (mean ± std):")
    rows, models = cv_table(X_train, y_train, skf)
    write_cv_table(rows)

    rows_sorted = sorted(rows, key=lambda r: r[3], reverse=True)
    top2 = [r[0] for r in rows_sorted[:2]]
    print(f"\n[2] Top-2 by f1: {top2}")

    tuned = {}
    for name in top2:
        pipe = make_pipe(models[name])
        grid, acc, f1, auc = tune_and_eval(name, pipe, X_train, y_train, X_test, y_test, skf)
        tuned[name] = (grid, acc, f1, auc)

    winner = max(tuned, key=lambda n: tuned[n][2])
    winner_grid = tuned[winner][0]
    print(f"\n[3] Winner: {winner}  best_params={winner_grid.best_params_}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        winner_grid, X_test, y_test, display_labels=data.target_names, ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix — {winner}")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    print("\n[4] Permutation importance on the test set …")
    perm = permutation_importance(
        winner_grid, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, scoring="f1", n_jobs=-1
    )
    order = np.argsort(perm.importances_mean)[::-1][:10]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        perm.importances_mean[order][::-1],
        xerr=perm.importances_std[order][::-1],
        color="steelblue",
    )
    ax.set_xlabel("mean F1 drop when permuted")
    ax.set_title(f"Permutation importance (top 10) — {winner}")
    fig.tight_layout()
    fig.savefig(OUT / "perm_importance.png", dpi=120)
    plt.close(fig)

    top_feats = order[:2].tolist()
    print(f"\n[5] Partial-dependence for features: {[feature_names[i] for i in top_feats]}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    PartialDependenceDisplay.from_estimator(
        winner_grid, X_train, features=top_feats, feature_names=feature_names, ax=axes, kind="average"
    )
    fig.suptitle(f"Partial dependence — {winner}")
    fig.tight_layout()
    fig.savefig(OUT / "pdp.png", dpi=120)
    plt.close(fig)

    print("\n[6] Learning curve …")
    train_sizes, train_scores, val_scores = learning_curve(
        winner_grid.best_estimator_, X_train, y_train,
        cv=skf, scoring="f1", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8), random_state=RANDOM_STATE,
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="train")
    ax.fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.15,
    )
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="cv")
    ax.fill_between(
        train_sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.15,
    )
    ax.set_xlabel("training set size")
    ax.set_ylabel("F1")
    ax.set_title(f"Learning curve — {winner}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.png", dpi=120)
    plt.close(fig)

    summary = "\n".join(
        [
            f"winner: {winner}",
            f"best_params: {winner_grid.best_params_}",
            f"test_accuracy: {tuned[winner][1]:.4f}",
            f"test_f1: {tuned[winner][2]:.4f}",
            f"test_auc: {tuned[winner][3]:.4f}",
            "top_features (perm_importance):",
            *[f"  {feature_names[i]}: {perm.importances_mean[i]:.4f}" for i in order],
        ]
    )
    (OUT / "winner.txt").write_text(summary + "\n")
    print("\n" + summary)


if __name__ == "__main__":
    main()
