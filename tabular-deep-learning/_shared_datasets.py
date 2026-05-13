"""Deterministic synthetic dataset generators reused across experiments.

The project's central question — *does tabular DL beat boosting?* — is
sensitive to dataset shape (mixed-type vs pure-numerical, cardinality,
n_samples, noise). The sandbox blocks every real-data host
(OpenML / HF / figshare), so we control the data ourselves. Every
generator is seeded and produces byte-identical output for the same
arguments.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler


def make_mixed_clf(
    n_samples: int = 20_000,
    n_num: int = 12,
    n_cat: int = 6,
    cat_cardinalities: tuple[int, ...] = (10, 10, 50, 50, 200, 200),
    cat_signal_strength: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Synthetic mixed-type binary classification.

    Numerical block from ``make_classification`` (informative + redundant
    structure), categoricals appended with sampled per-level effects added
    to the continuous score before re-binarizing on the median, so cats
    carry real signal.
    """
    if len(cat_cardinalities) != n_cat:
        raise ValueError("cat_cardinalities length must equal n_cat")

    rng = np.random.default_rng(seed)
    X_num, _ = make_classification(
        n_samples=n_samples,
        n_features=n_num,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=2,
        class_sep=1.0,
        flip_y=0.02,
        random_state=seed,
    )
    score = X_num[:, :8].sum(axis=1)

    cat_cols = {}
    for j, card in enumerate(cat_cardinalities):
        levels = rng.integers(0, card, size=n_samples)
        effects = rng.normal(0.0, cat_signal_strength, size=card)
        score = score + effects[levels]
        cat_cols[f"cat_{j}"] = levels.astype(np.int64)

    score = score + rng.normal(0.0, 0.5, size=n_samples)
    y = (score > np.median(score)).astype(np.int64)

    num_names = [f"num_{i}" for i in range(n_num)]
    df = pd.DataFrame(X_num, columns=num_names)
    for name, col in cat_cols.items():
        df[name] = col
    return df, y, num_names, list(cat_cols.keys())


def make_mixed_reg(
    n_samples: int = 20_000,
    n_num: int = 12,
    n_cat: int = 6,
    cat_cardinalities: tuple[int, ...] = (10, 10, 50, 50, 200, 200),
    cat_signal_strength: float = 1.0,
    noise: float = 10.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Synthetic mixed-type regression. Same recipe as ``make_mixed_clf``
    but y stays continuous.
    """
    if len(cat_cardinalities) != n_cat:
        raise ValueError("cat_cardinalities length must equal n_cat")

    rng = np.random.default_rng(seed)
    X_num, y_num = make_regression(
        n_samples=n_samples,
        n_features=n_num,
        n_informative=8,
        noise=noise,
        random_state=seed,
    )
    y = y_num.copy().astype(np.float64)

    cat_cols = {}
    for j, card in enumerate(cat_cardinalities):
        levels = rng.integers(0, card, size=n_samples)
        # scale cat effects to the y stddev so they're comparable to numericals
        effects = rng.normal(0.0, cat_signal_strength * y.std(), size=card)
        y = y + effects[levels]
        cat_cols[f"cat_{j}"] = levels.astype(np.int64)

    num_names = [f"num_{i}" for i in range(n_num)]
    df = pd.DataFrame(X_num, columns=num_names)
    for name, col in cat_cols.items():
        df[name] = col
    return df, y, num_names, list(cat_cols.keys())


def make_pure_num_clf(
    n_samples: int = 10_000,
    n_features: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.6),
        n_redundant=int(n_features * 0.2),
        n_clusters_per_class=2,
        class_sep=1.0,
        flip_y=0.02,
        random_state=seed,
    )
    return X.astype(np.float32), y.astype(np.int64)


def make_pure_num_reg(
    n_samples: int = 10_000,
    n_features: int = 20,
    noise: float = 10.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.6),
        noise=noise,
        random_state=seed,
    )
    return X.astype(np.float32), y.astype(np.float32)


def make_cardinality_variant(
    cardinality: int,
    n_samples: int = 15_000,
    n_num: int = 10,
    n_cat: int = 4,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Uniform-cardinality variant for the categorical-encoding ablation.

    All ``n_cat`` cat columns get the same cardinality. Used to sweep
    cardinality ∈ {10, 50, 200}.
    """
    return make_mixed_clf(
        n_samples=n_samples,
        n_num=n_num,
        n_cat=n_cat,
        cat_cardinalities=(cardinality,) * n_cat,
        seed=seed,
    )


def corrupt(
    X: np.ndarray,
    kind: Literal["gauss", "mcar", "shift"],
    magnitude: float,
    seed: int = 0,
    shift_cols: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Apply a deterministic corruption to a feature matrix.

    - ``gauss``: add N(0, magnitude) noise to every column. Magnitude is
      in standard-deviation units of each column.
    - ``mcar``: replace ``magnitude`` fraction of cells with NaN.
    - ``shift``: add ``magnitude`` * column-stddev to ``shift_cols`` only.
    """
    rng = np.random.default_rng(seed)
    out = X.astype(np.float64, copy=True)
    if kind == "gauss":
        col_std = out.std(axis=0)
        out = out + rng.normal(0.0, magnitude, size=out.shape) * col_std
    elif kind == "mcar":
        mask = rng.random(out.shape) < magnitude
        out[mask] = np.nan
    elif kind == "shift":
        if shift_cols is None:
            shift_cols = tuple(range(min(3, out.shape[1])))
        col_std = out[:, list(shift_cols)].std(axis=0)
        out[:, list(shift_cols)] = out[:, list(shift_cols)] + magnitude * col_std
    else:
        raise ValueError(f"unknown corruption: {kind}")
    return out


def standardize_numericals(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    num_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Z-score the numerical columns; return new frames + the fitted
    scaler so downstream code can apply it to corrupted test variants.
    """
    scaler = StandardScaler().fit(df_train[num_names].values)
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[num_names] = scaler.transform(df_train[num_names].values)
    df_test[num_names] = scaler.transform(df_test[num_names].values)
    return df_train, df_test, scaler


if __name__ == "__main__":
    # Smoke tests — invoked via `uv run python _shared_datasets.py`.
    print("=== make_mixed_clf ===")
    df, y, num, cat = make_mixed_clf(n_samples=1000)
    print("shape:", df.shape, "y dist:", np.bincount(y))
    print("cats:", {c: df[c].nunique() for c in cat})

    print("=== make_mixed_reg ===")
    df, y, num, cat = make_mixed_reg(n_samples=1000)
    print("shape:", df.shape, "y mean/std:", float(y.mean()), float(y.std()))

    print("=== make_pure_num_clf ===")
    X, y = make_pure_num_clf(n_samples=500)
    print("shape:", X.shape, "y dist:", np.bincount(y))

    print("=== make_pure_num_reg ===")
    X, y = make_pure_num_reg(n_samples=500)
    print("shape:", X.shape, "y std:", float(y.std()))

    print("=== make_cardinality_variant ===")
    df, y, num, cat = make_cardinality_variant(50, n_samples=500)
    print("shape:", df.shape, "cats:", {c: df[c].nunique() for c in cat})

    print("=== corrupt ===")
    Xc = corrupt(X, "gauss", 0.5, seed=0)
    print("gauss diff std:", float((Xc - X).std()))
    Xc = corrupt(X, "mcar", 0.2, seed=0)
    print("mcar nan fraction:", float(np.isnan(Xc).mean()))
    Xc = corrupt(X, "shift", 1.0, seed=0, shift_cols=(0, 1, 2))
    print("shift col0 mean diff:", float((Xc[:, 0] - X[:, 0]).mean()))

    # determinism check
    df1, *_ = make_mixed_clf(n_samples=200, seed=42)
    df2, *_ = make_mixed_clf(n_samples=200, seed=42)
    assert (df1.values == df2.values).all(), "non-deterministic!"
    print("OK: determinism confirmed.")
