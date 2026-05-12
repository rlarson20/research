# Experiment 1 — Boosting and categoricals

## Goal

Find the experimental setup where XGBoost, LightGBM, and CatBoost actually
diverge. They tie on small clean numeric tabular data; the real difference
is **how they ingest categoricals**. Build a controlled binary classification
problem where high-cardinality categoricals carry real signal, sweep
cardinality, and compare native handling (CatBoost), integer-coded native
(LightGBM `categorical_feature=`), and forced one-hot (XGBoost).

Pair with Optuna for TPE hyperparameter tuning and SHAP TreeExplainer
for explanations on the winner.

## Dataset

`sklearn.datasets.load_diabetes()` (442 × 10) binarized on the median, with
**3 synthetic categorical columns** appended. Each category gets an additive
per-level effect on the continuous target before binarization, so by
construction the categoricals are informative. The first cat has a strong
effect (σ=30) and the others noisier (σ=8). Cardinality sweeps over
{5, 20, 50, 80}.

## APIs used and why

### Boosters
- **`xgboost.XGBClassifier(tree_method="hist")`** — histogram method is
  the modern default; `enable_categorical=True` exists but is still
  experimental, so we do the realistic-user thing: `OneHotEncoder` the
  cat columns and feed them as numerics.
- **`lightgbm.LGBMClassifier(deterministic=True, force_row_wise=True)`** —
  determinism flags matter; LightGBM is non-reproducible by default with
  `n_jobs>1`. `categorical_feature=` accepts integer-coded columns
  directly and uses LightGBM's native partitioning algorithm.
- **`catboost.CatBoostClassifier`** with `Pool(cat_features=…)` — CatBoost
  consumes raw categoricals via ordered target statistics and ordered
  boosting, no encoding required.

### Tooling
- **`optuna.create_study(sampler=TPESampler(seed=42))`** — 20 trials of
  Tree-structured Parzen Estimator over lr / num_leaves / depth /
  min_child_samples / reg_alpha / reg_lambda. `MedianPruner` was originally
  planned but `optuna.integration.LightGBMPruningCallback` now requires
  the separate `optuna-integration[lightgbm]` package; we use LightGBM's
  built-in `early_stopping(20)` instead, which gets most of the benefit.
- **`shap.TreeExplainer`** — exact tree-path SHAP values, fast on small
  models. Beeswarm gives global importance + direction, waterfall gives
  one local explanation.

### Determinism gotchas worth documenting
- **XGBoost**: non-deterministic with `n_jobs>1` unless
  `tree_method="hist"`. We use it.
- **LightGBM**: must set `deterministic=True, force_row_wise=True`.
- **CatBoost**: deterministic by default; just pass `random_seed=`.

## How to read the outputs

- `cardinality_table.txt` — log-loss, AUC, fit time per (library × cardinality).
- `cardinality_sweep.png` — log-loss and AUC curves as cardinality grows.
  The story: CatBoost stays flat-good, XGBoost/LightGBM degrade.
- `time_vs_cardinality.png` — training time per library as one-hot blows
  up XGBoost's feature count.
- `optuna_parallel.png` — parallel-coordinates over the 20 TPE trials,
  colored by objective; shows which regions of hyperparameter space were
  productive.
- `shap_beeswarm.png` — global feature importance + direction for the
  tuned LightGBM on 100 test rows.
- `shap_waterfall_example.png` — single-prediction explanation for one
  test row.
- `winner.txt` — best Optuna params + tuned test metrics.
