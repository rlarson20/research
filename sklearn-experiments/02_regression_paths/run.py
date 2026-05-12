"""Experiment 2 — Regression + regularization paths + feature selection."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    mutual_info_regression,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LassoCV,
    LinearRegression,
    Ridge,
    lasso_path,
)
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

RANDOM_STATE = 42
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def build_augmented_matrix(X, feature_names, n_noise=10, rng=None):
    rng = np.random.default_rng(RANDOM_STATE) if rng is None else rng
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    poly_names = poly.get_feature_names_out(feature_names).tolist()
    noise = rng.standard_normal((X.shape[0], n_noise))
    noise_names = [f"noise_{i}" for i in range(n_noise)]
    X_aug = np.hstack([X_poly, noise])
    names = poly_names + noise_names
    return X_aug, names


def cv_table(X, y, kf):
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.5, random_state=RANDOM_STATE, max_iter=20000),
        "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=20000),
        "GBR": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "RF": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "SVR-rbf": SVR(kernel="rbf"),
    }
    rows = []
    for name, est in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", est)])
        if name == "SVR-rbf":
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(X.shape[0], size=min(300, X.shape[0]), replace=False)
            Xs, ys = X[idx], y[idx]
        else:
            Xs, ys = X, y
        rmse = -cross_val_score(pipe, Xs, ys, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)
        r2 = cross_val_score(pipe, Xs, ys, cv=kf, scoring="r2", n_jobs=-1)
        rows.append((name, rmse.mean(), rmse.std(), r2.mean(), r2.std()))
        print(f"  {name:11s} rmse={rmse.mean():8.3f}±{rmse.std():.3f}  r2={r2.mean():.3f}±{r2.std():.3f}")
    return rows, models


def write_cv_table(rows):
    lines = [
        "model       rmse_mean  rmse_std  r2_mean   r2_std",
        "-----       ---------  --------  -------   ------",
    ]
    for name, r_m, r_s, q_m, q_s in rows:
        lines.append(f"{name:11s} {r_m:8.3f}   {r_s:.3f}     {q_m:.3f}     {q_s:.3f}")
    (OUT / "cv_table.txt").write_text("\n".join(lines) + "\n")


def lasso_path_plot(X, y, feature_names):
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    alphas = np.logspace(-3, 1, 80)
    alphas_path, coefs, _ = lasso_path(Xs, y, alphas=alphas, max_iter=20000)
    cv = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=20000, n_jobs=-1).fit(Xs, y)
    is_noise = np.array([name.startswith("noise_") for name in feature_names])

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(feature_names):
        color = "lightgray" if is_noise[i] else None
        lw = 0.7 if is_noise[i] else 1.0
        ax.plot(alphas_path, coefs[i], color=color, linewidth=lw)
    ax.set_xscale("log")
    ax.axvline(cv.alpha_, color="red", linestyle="--", label=f"LassoCV α = {cv.alpha_:.4f}")
    ax.set_xlabel("alpha (log)")
    ax.set_ylabel("coefficient")
    ax.set_title("LASSO regularization path (gray = injected noise features)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "lasso_path.png", dpi=120)
    plt.close(fig)

    survived = [n for n, c in zip(feature_names, cv.coef_) if abs(c) > 1e-8]
    n_noise_surv = sum(1 for n in survived if n.startswith("noise_"))
    return cv, survived, n_noise_surv


def evaluate_selector(name, selector, X_train, y_train, X_test, y_test, feature_names):
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("select", selector),
            ("reg", Ridge(alpha=1.0)),
        ]
    )
    pipe.fit(X_train, y_train)
    rmse = root_mean_squared_error(y_test, pipe.predict(X_test))
    support = pipe.named_steps["select"].get_support()
    chosen = [feature_names[i] for i, s in enumerate(support) if s]
    n_chosen = len(chosen)
    n_noise = sum(1 for c in chosen if c.startswith("noise_"))
    print(f"  {name:18s} rmse={rmse:7.3f}  chose {n_chosen} feats, {n_noise} are noise")
    return name, rmse, n_chosen, n_noise, chosen


def feature_selection_plot(results):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [r[0] for r in results]
    rmses = [r[1] for r in results]
    bars = ax.bar(names, rmses, color="steelblue")
    for r, b in zip(results, bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.5,
            f"{r[2]} feats\n{r[3]} noise",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("downstream Ridge test-RMSE")
    ax.set_title("Feature selection methods (lower is better)")
    fig.tight_layout()
    fig.savefig(OUT / "feature_selection.png", dpi=120)
    plt.close(fig)


def residual_plot(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    resid = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_test, y_pred, alpha=0.6, color="steelblue")
    lo, hi = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0].set_xlabel("actual"); axes[0].set_ylabel("predicted")
    axes[0].set_title(f"Predicted vs. actual — {name}\nR²={r2_score(y_test, y_pred):.3f}")
    axes[1].scatter(y_pred, resid, alpha=0.6, color="steelblue")
    axes[1].axhline(0, color="k", linestyle="--", linewidth=1)
    axes[1].set_xlabel("predicted"); axes[1].set_ylabel("residual")
    axes[1].set_title("Residuals vs. fitted")
    fig.tight_layout()
    fig.savefig(OUT / "residuals.png", dpi=120)
    plt.close(fig)


def main():
    data = load_diabetes()
    X0, y, names0 = data.data, data.target, list(data.feature_names)
    X, feature_names = build_augmented_matrix(X0, names0)
    print(f"Dataset: diabetes augmented  X={X.shape}  (10 originals + 45 interactions + 10 noise)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n[1] 5-fold CV on the full augmented matrix:")
    rows, models = cv_table(X_train, y_train, kf)
    write_cv_table(rows)
    best_name = min(rows, key=lambda r: r[1])[0]
    print(f"  -> best by CV RMSE: {best_name}")

    print("\n[2] LASSO regularization path …")
    cv_lasso, survived, n_noise_surv = lasso_path_plot(X_train, y_train, feature_names)
    print(f"  LassoCV α={cv_lasso.alpha_:.4f}; kept {len(survived)} features, {n_noise_surv} of them are noise")

    print("\n[3] Feature selectors (downstream Ridge test-RMSE):")
    results = []
    results.append(
        evaluate_selector(
            "SelectKBest(MI)",
            SelectKBest(score_func=mutual_info_regression, k=10),
            X_train, y_train, X_test, y_test, feature_names,
        )
    )
    results.append(
        evaluate_selector(
            "RFE(Ridge)",
            RFE(Ridge(alpha=1.0), n_features_to_select=10),
            X_train, y_train, X_test, y_test, feature_names,
        )
    )
    results.append(
        evaluate_selector(
            "SelectFromModel(LassoCV)",
            SelectFromModel(LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=20000)),
            X_train, y_train, X_test, y_test, feature_names,
        )
    )
    results.append(
        evaluate_selector(
            "SelectFromModel(RF)",
            SelectFromModel(RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
            X_train, y_train, X_test, y_test, feature_names,
        )
    )
    feature_selection_plot(results)

    baseline_pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))]).fit(X_train, y_train)
    baseline_rmse = root_mean_squared_error(y_test, baseline_pipe.predict(X_test))
    print(f"  (no selection baseline: Ridge test-RMSE = {baseline_rmse:.3f})")

    print(f"\n[4] Residual plot for best CV model ({best_name}) …")
    best_pipe = Pipeline([("scaler", StandardScaler()), ("reg", models[best_name])]).fit(X_train, y_train)
    residual_plot(best_name, best_pipe, X_test, y_test)

    lines = [
        f"best_cv_model: {best_name}",
        f"baseline_no_selection_rmse: {baseline_rmse:.3f}",
        f"lassocv_alpha: {cv_lasso.alpha_:.4f}",
        f"lassocv_features_kept: {len(survived)} ({n_noise_surv} noise)",
        "feature_selection_results:",
        *[f"  {r[0]}: rmse={r[1]:.3f}  feats={r[2]}  noise={r[3]}" for r in results],
    ]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
