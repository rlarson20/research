# Experiment 6 — Robustness probe

## Goal

Experiment 5 showed DL beats CatBoost on *clean* synthetic data. The
Grinsztajn 2022 thesis cites real-world quirks — outliers, missingness,
distribution drift — as the reason DL loses on real tabular data. This
experiment tests that hypothesis directly: take the same clean
training set as experiment 5, train once, then evaluate on three
families of corrupted test sets.

## Setup

- **Data:** same `make_mixed_clf(n_samples=45_000, seed=42)` pool /
  test split as experiment 5 (5000 test, 40000 train). FT-Transformer
  uses 38000 train + 2000 internal val (cheap held-out fold for the
  shared `train_one` loop).
- **Models (fitted once each on the clean train set):**
  - CatBoost — `iterations=400, depth=6, nan_mode="Min"` (explicit, the
    default for binary clf).
  - LightGBM — `n_estimators=400, max_depth=6, deterministic=True`.
  - FT-Transformer — depth=2 default, 6 epochs (matches experiment 5).
- **Corruptions (via `_shared_datasets.corrupt`):**
  - `gauss(σ)` — N(0, σ × col_std) noise added to every numerical
    column. σ ∈ {0.1, 0.5, 1.0}.
  - `mcar(rate)` — Random fraction `rate` of cells in the test
    feature matrix replaced with NaN. rate ∈ {0.05, 0.20, 0.50}.
    Boosters get NaN directly (native handling); FT-Transformer
    zero-fills.
  - `shift(μ)` — Add `μ × col_std` to the **top-3 most important
    numerical features** (importance ranked by LightGBM's
    `feature_importances_`). μ ∈ {0.5, 1.0, 2.0}.
- **Cat columns are not corrupted** in any setting. The corruptions
  only touch the numerical block — keeping the comparison about
  numerical-feature robustness.

## APIs

- `catboost.CatBoostClassifier(nan_mode="Min")` — explicit policy for
  the missingness probe. CatBoost's defaults for binary clf are
  `nan_mode="Min"`, which encodes NaN as "less than the smallest
  observed value" at split time.
- `lightgbm.LGBMClassifier(...)` — handles NaN natively without an
  explicit flag.
- FT-Transformer — torch model evaluated under `torch.no_grad()`;
  NaN test cells replaced with `0.0` via `np.where(np.isnan, 0, x)`.

## Outputs

- `degradation_table.txt` — 10 rows × 8 columns: 1 clean baseline +
  3 corruptions × 3 magnitudes, with absolute accuracy and Δ-from-
  baseline for each model.
- `degradation_noise.png`, `degradation_missing.png`, `degradation_shift.png` —
  one panel per corruption family, accuracy vs magnitude.
