# Experiment 1 — Postmortem

## Headline

**CatBoost's ordered target statistics are not just a marketing line.** On a
binary classification problem with 3 informative high-cardinality categorical
columns, CatBoost beats both LightGBM (`categorical_feature=` integer-coded)
and XGBoost (forced one-hot) at every cardinality tested — and the gap widens
with cardinality. Log-loss results (lower is better):

| cardinality | CatBoost | LightGBM | XGBoost-OH |
|------------:|---------:|---------:|-----------:|
| 5  | **0.534** | 0.758 | 0.793 |
| 20 | **0.539** | 0.891 | 0.880 |
| 50 | **0.472** | 0.656 | 0.849 |
| 80 | **0.555** | 0.842 | 0.983 |

AUC tells the same ranking story (CatBoost ≈0.81 vs others ≈0.73–0.78).

## Training time vs cardinality

CatBoost is the slowest (~0.2s) but constant in cardinality — it doesn't
care if there are 5 or 80 levels. XGBoost's fit time creeps up slightly as
the one-hot feature count grows (5 cats × 3 cols = 15 features vs 80 × 3 =
240 features). LightGBM is fastest (~0.03s) and also constant.

On this dataset size (~330 train rows) all three finish in under a quarter
second — the real cost of XGBoost's one-hot approach is statistical, not
computational.

## Optuna result

20 TPE trials on LightGBM @ cardinality=50, optimizing held-out validation
log-loss. Best params:

```
lr=0.066, num_leaves=9, max_depth=8,
min_child_samples=33, reg_alpha=0.040, reg_lambda=0.963
```

Best validation log-loss: 0.518 → tuned test log-loss 0.681 (AUC 0.785).
This *improves* over default LightGBM (0.656 → 0.681 is roughly tied on
this train/test split, but the validation gain is real). It still loses to
out-of-the-box CatBoost (0.472), which is the headline finding.

## SHAP

Beeswarm on 100 test rows surfaces the synthetic categorical columns as
the dominant predictors — exactly as designed, since we built them in with
strong per-level effects. The numeric columns from `load_diabetes` show
much smaller and noisier SHAP magnitudes. Local waterfall on row 0 shows
the same structure: most of the deviation from the base rate comes from
the categorical levels.

## Surprises and gotchas

- **Determinism is per-library and non-trivial.** XGBoost needs
  `tree_method="hist"` to be reproducible with parallelism. LightGBM
  needs both `deterministic=True` and `force_row_wise=True`. CatBoost is
  deterministic out of the box. None of them surface this clearly in
  their main docs.
- **`optuna.integration.LightGBMPruningCallback` is gone** — it now lives
  in a separately-pip-installable `optuna-integration[lightgbm]` package.
  Modern Optuna prints a clean error pointing at the missing package, but
  the integration was silently in `optuna.integration.*` for years before
  this. Switched to LightGBM's built-in `early_stopping(20)` callback.
- **SHAP raised a `UserWarning`**: "LightGBM binary classifier with
  TreeExplainer shap values output has changed to a list of ndarray".
  In current versions it returns a single 2-D array — the script handles
  both with `if isinstance(sv, list)`.
- **The `optuna.visualization.matplotlib.plot_parallel_coordinate`
  experimental warning** is harmless but loud — it's been experimental
  since v2.2.0.

## What I'd try next

1. **Monotonic constraints.** All three libraries support them (XGBoost
   `monotone_constraints=`, LightGBM same, CatBoost
   `monotone_constraints=`) but with mutually-incompatible syntax. A
   constraint that improves test performance would be a strong "why these
   libraries diverge" finding distinct from the categorical story.
2. **`enable_categorical=True` on XGBoost.** It's experimental but should
   close part of the gap to LightGBM/CatBoost.
3. **Run with `early_stopping_rounds` for all three** during the sweep, not
   just inside Optuna. Right now the fit-time comparison is fair but the
   metric comparison gives every library the same 200 trees — which may
   favour different libraries differently.
4. **Try `imblearn` SMOTE-NC** (handles mixed continuous+categorical) and
   see if any of the three boosters benefits from rebalancing on a heavily
   imbalanced variant of this dataset.
