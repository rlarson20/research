"""Experiment 4 — Imbalanced classification: class weights, threshold tuning, calibration."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def minority_metrics(y_true, y_pred):
    return (
        precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    )


def build_data():
    X, y = make_classification(
        n_samples=10_000, n_features=20, n_informative=8, n_redundant=4,
        weights=[0.95, 0.05], class_sep=0.8, random_state=RANDOM_STATE,
    )
    print(f"Dataset: imbalanced synthetic  X={X.shape}  minority={int(y.sum())} ({y.mean():.3f})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def class_weight_section(X_train, X_test, y_train, y_test):
    print("\n[1] Class weight study (minority class metrics at threshold 0.5):")
    rows = []
    specs = [
        ("LogReg", lambda cw: LogisticRegression(max_iter=5000, class_weight=cw, random_state=RANDOM_STATE), True),
        ("RandomForest", lambda cw: RandomForestClassifier(n_estimators=300, class_weight=cw, random_state=RANDOM_STATE, n_jobs=-1), True),
        ("GradBoost", lambda cw: GradientBoostingClassifier(random_state=RANDOM_STATE), False),
    ]
    for name, ctor, supports_cw in specs:
        weights = (None, "balanced") if supports_cw else (None,)
        for cw in weights:
            model = ctor(cw).fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            prec, rec, f1 = minority_metrics(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            cw_label = str(cw) if supports_cw else "n/a"
            rows.append((name, cw_label, prec, rec, f1, auc))
            print(f"  {name:13s} cw={cw_label:8s}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

    lines = ["model         class_weight  precision  recall   f1      auc",
             "-----         ------------  ---------  ------   --      ---"]
    for name, cw, p, r, f, a in rows:
        lines.append(f"{name:13s} {cw:12s} {p:.3f}      {r:.3f}    {f:.3f}   {a:.3f}")
    (OUT / "class_weight_table.txt").write_text("\n".join(lines) + "\n")
    return rows


def threshold_section(X_train, X_test, y_train, y_test, best_model_name="GradBoost"):
    print("\n[2] Threshold tuning on the strongest model …")
    model = GradientBoostingClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best_idx = int(np.argmax(f1[:-1]))
    best_t = float(thresholds[best_idx])

    y_pred_05 = (y_score >= 0.5).astype(int)
    y_pred_t = (y_score >= best_t).astype(int)
    p05, r05, f05 = minority_metrics(y_test, y_pred_05)
    pt, rt, ft = minority_metrics(y_test, y_pred_t)
    print(f"  default threshold 0.5  -> P={p05:.3f}  R={r05:.3f}  F1={f05:.3f}")
    print(f"  optimal threshold {best_t:.3f}  -> P={pt:.3f}  R={rt:.3f}  F1={ft:.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label="PR curve")
    ax.scatter(r05, p05, color="orange", zorder=3, label="threshold = 0.5")
    ax.scatter(rt, pt, color="red", zorder=3, label=f"F1-optimal t = {best_t:.3f}")
    no_skill = (y_test == 1).mean()
    ax.axhline(no_skill, color="gray", linestyle="--", linewidth=1, label=f"no-skill (P={no_skill:.3f})")
    ax.set_xlabel("recall (minority)")
    ax.set_ylabel("precision (minority)")
    ax.set_title("Precision-Recall — GradientBoostingClassifier")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "pr_curve.png", dpi=120)
    plt.close(fig)

    lines = [
        f"model: GradientBoostingClassifier (default class_weight)",
        f"default_threshold_0.5     precision={p05:.3f}  recall={r05:.3f}  f1={f05:.3f}",
        f"f1_optimal_threshold={best_t:.4f}  precision={pt:.3f}  recall={rt:.3f}  f1={ft:.3f}",
        f"f1_uplift_vs_default = {ft - f05:+.3f}",
    ]
    (OUT / "threshold_table.txt").write_text("\n".join(lines) + "\n")
    return best_t, model


def calibration_section(X_train, X_test, y_train, y_test):
    print("\n[3] Calibration: uncalibrated RF vs isotonic vs sigmoid …")
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
    rf_iso = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        method="isotonic", cv=5,
    ).fit(X_train, y_train)
    rf_sig = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        method="sigmoid", cv=5,
    ).fit(X_train, y_train)

    models = {
        "RF uncalibrated": rf,
        "RF + isotonic":   rf_iso,
        "RF + sigmoid":    rf_sig,
    }
    briers = {}
    for name, m in models.items():
        proba = m.predict_proba(X_test)[:, 1]
        briers[name] = brier_score_loss(y_test, proba)
        print(f"  {name:18s} Brier={briers[name]:.5f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, m in models.items():
        CalibrationDisplay.from_estimator(m, X_test, y_test, n_bins=10, name=name, ax=ax)
    ax.set_title("Reliability diagram (perfect = diagonal)")
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=120)
    plt.close(fig)

    lines = ["model                Brier_score",
             "-----                -----------"]
    for name, b in briers.items():
        lines.append(f"{name:20s} {b:.5f}")
    (OUT / "brier.txt").write_text("\n".join(lines) + "\n")
    return briers


def main():
    X_train, X_test, y_train, y_test = build_data()
    cw_rows = class_weight_section(X_train, X_test, y_train, y_test)
    best_t, _ = threshold_section(X_train, X_test, y_train, y_test)
    briers = calibration_section(X_train, X_test, y_train, y_test)

    summary = [
        "minority-class metrics at threshold 0.5:",
        *[f"  {n} cw={cw}: P={p:.3f} R={r:.3f} F1={f:.3f} AUC={a:.3f}" for n, cw, p, r, f, a in cw_rows],
        f"f1-optimal threshold for GBM: {best_t:.4f}",
        "calibration brier scores:",
        *[f"  {n}: {b:.5f}" for n, b in briers.items()],
    ]
    (OUT / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n" + "\n".join(summary))


if __name__ == "__main__":
    main()
