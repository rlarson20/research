# ML-library experiments (non-sklearn)

Four small, self-contained experiments that exercise popular Python ML
libraries *outside* scikit-learn. Sibling project to
[`../sklearn-experiments/`](../sklearn-experiments/) and follows the same
convention: each experiment is in its own subfolder with

- `run.py` — the experiment script (`uv run python NN_*/run.py`)
- `summary.md` — *what* APIs are used and *why*
- `postmortem.md` — *what actually happened* when the script ran
- `outputs/` — PNG plots and tabular `.txt` summaries

## Reproducing

```
cd ml-library-experiments
uv sync
uv run python 01_boosting_categoricals/run.py    # ~5s
uv run python 02_pytorch_training_loop/run.py    # ~9s
uv run python 03_imblearn_resampling/run.py      # ~7s
uv run python 04_numpyro_bayesian/run.py         # ~23s
```

Every script uses `random_state=42`, runs in under ~30s on a single CPU,
and writes outputs to its `outputs/` folder. Tested on Python 3.11 with
`xgboost==3.2.0`, `lightgbm==4.6.0`, `catboost==1.2.10`, `optuna==4.8.0`,
`shap==0.51.0`, `torch==2.11.0+cu130` (CPU, no GPU used),
`imbalanced-learn==0.14.1`, `numpyro==0.21.0`, `jax==0.10.0`,
`statsmodels==0.14.6`.

## Experiments at a glance

| # | Folder | Topic | Dataset | Headline finding |
|---|---|---|---|---|
| 1 | [`01_boosting_categoricals/`](01_boosting_categoricals/) | XGBoost vs LightGBM vs CatBoost on high-cardinality categoricals; Optuna TPE + SHAP TreeExplainer | `load_diabetes` binarized + 3 synthetic informative categoricals, cardinality {5, 20, 50, 80} | **CatBoost dominates** at every cardinality (log-loss ≈ 0.50 vs LightGBM/XGBoost ≈ 0.80) — its ordered target statistics genuinely beat both integer-coded native (LightGBM) and forced one-hot (XGBoost). |
| 2 | [`02_pytorch_training_loop/`](02_pytorch_training_loop/) | Hand-rolled PyTorch training loop with instrumentation: weight-init × BatchNorm × LR-schedule ablations, plus a tiny CNN | `load_digits` (1797 × 64) | The hand-rolled loop's value isn't accuracy (98.0% vs sklearn's 97.3%) — it's the **gradient-norm trajectory**: BN keeps gradient L2 ≈ 2-4 throughout training, no-BN runs decay **~10×**. Also: aggressive LR schedules (Step, Cosine) *hurt* at 12 epochs because they anneal before the model converges. |
| 3 | [`03_imblearn_resampling/`](03_imblearn_resampling/) | imbalanced-learn: SMOTE, ADASYN, RandomUnder, NearMiss, SMOTE+Tomek, SMOTE+ENN, RandomOver vs `class_weight="balanced"` vs nothing | `make_classification(weights=[0.99, 0.01])`, 10000 × 20 | **At 99/1 imbalance, do nothing — just lower the threshold.** Plain LogReg + tuned threshold (F1=0.327) beats every imblearn resampler (best: SMOTE-family at 0.246) and beats `class_weight="balanced"` (0.246). The tutorial-popular SMOTE move is actively worse here. |
| 4 | [`04_numpyro_bayesian/`](04_numpyro_bayesian/) | NumPyro Bayesian linear regression with Normal prior and regularized horseshoe prior, vs sklearn `Lasso`/`Ridge` and statsmodels OLS | `load_diabetes` raw + noise-augmented (10 N(0,1) cols) | Three different winners by metric: **Lasso wins noise rejection** (mean \|β_noise\|=0.92), **Ridge wins signal recovery** (4× better), and the **horseshoe is the only sparsity method that ships calibrated uncertainty intervals** (CrI width 13.2 on signal cols). On raw data Bayesian CrIs and OLS CIs agree closely. |

## Note on Experiment 3

The Phase 4 plan called for a Hugging Face transformers experiment here
(zero-shot vs frozen-encoder linear probe). The sandbox blocks
`huggingface.co` (HTTP 403 `host_not_allowed`) and there's no cached
model on disk, so we pivoted to imbalanced-learn — also a popular
non-sklearn library, also pairs cleanly with sklearn-experiment-4 on
calibration, and fully offline. Details in
[`03_imblearn_resampling/summary.md`](03_imblearn_resampling/summary.md).

## Cross-experiment observations

### 1. "Modern" libraries don't beat sklearn on small clean data

The sklearn-experiments project found that `LogisticRegression` beat the
ensembles on `load_breast_cancer`. This project found the same shape of
result several more times:

- **Experiment 2**: a hand-rolled PyTorch MLP improves the sklearn
  `MLPClassifier` baseline by only 0.7 percentage points (97.3% → 98.0%),
  and a small CNN doesn't improve over an MLP on 8×8 inputs.
- **Experiment 3**: vanilla `LogisticRegression` + threshold tuning beats
  every imbalanced-learn resampler and `class_weight="balanced"`.
- **Experiment 4**: `Ridge(α=1.0)` recovers signal coefficients 4×
  better than full Bayesian inference under either prior.

Sklearn's defaults are remarkably hard to beat. **The wins from
"upgrading" come from features sklearn doesn't have**, not from
accuracy: CatBoost gets you categorical handling, PyTorch gets you the
training loop you can instrument, imblearn gets you the *option* to
resample (sometimes useful, sometimes harmful), Bayesian methods get
you calibrated uncertainty. None of those are "sklearn but more
accurate".

### 2. AUC and threshold-dependent metrics keep disagreeing

sklearn-experiment-4 had `RandomForest(class_weight="balanced")`
improve AUC 0.913 → 0.928 while *losing* F1 at threshold 0.5
(0.212 → 0.107). This project's Experiment 3 made the same lesson
sharper: at 99/1 imbalance, **AUC for the resampled fits is *higher*
than for the do-nothing baseline (0.67 vs 0.64), but AP (the
imbalance-aware metric) is *lower* (0.13 vs 0.21).** AUC measures
average ranking quality across thresholds; the part of the ranking
that matters when you have 35 minority examples can degrade even
when the average improves.

### 3. Library-determinism flags are non-obvious and per-library

- XGBoost is non-deterministic with `n_jobs>1` unless
  `tree_method="hist"`.
- LightGBM needs both `deterministic=True` and `force_row_wise=True`.
- CatBoost is deterministic by default.
- PyTorch needs `manual_seed(42)` *and* `DataLoader(generator=…)`
  manual-seeded; `use_deterministic_algorithms(True)` crashes on some
  ops.
- NumPyro: `JAX_PLATFORMS=cpu` env var + `jax.config.update("jax_platform_name", "cpu")`
  before any other JAX import or the GPU probe wastes seconds.

None of these are mentioned in the libraries' getting-started docs.

### 4. The pipeline API class matters

`imblearn.pipeline.Pipeline` is *not* a drop-in replacement for
`sklearn.pipeline.Pipeline`. The imblearn version correctly skips the
sampler on test folds during cross-validation; the sklearn version
would silently leak resampled rows into the test set. Always use
imblearn's pipeline when a sampler is in the chain.

The same pattern: NumPyro needs you to think about CPU thread layout
(`set_host_device_count(1)`, `chain_method="sequential"`); JAX needs
the `JAX_PLATFORMS=cpu` env var to skip GPU probing. Defaults that
match the *expected* deployment environment leak through.

### 5. The instrumentation case for hand-written training loops

PyTorch's value over sklearn's `MLPClassifier` isn't a final-accuracy
delta — it's all the *intermediate* quantities that the loop exposes.
Per-epoch gradient norm makes BatchNorm's effect visible. Per-epoch
LR (under a scheduler) lets you debug "the schedule decayed too early".
Per-batch loss lets you compare optimizers honestly. None of those are
accessible inside `MLPClassifier.fit()`.

## Library surface area covered

A non-exhaustive list:

- **Boosting**: `xgboost.XGBClassifier(tree_method="hist")`,
  `lightgbm.LGBMClassifier(categorical_feature=, deterministic=,
  force_row_wise=)`, `catboost.CatBoostClassifier` + `Pool(cat_features=)`.
- **HPO**: `optuna.create_study(sampler=TPESampler)` with
  `lightgbm.early_stopping` callback, `plot_parallel_coordinate`.
- **Explainability**: `shap.TreeExplainer` + `summary_plot` (beeswarm)
  + `plots.waterfall`.
- **Deep learning**: `torch.nn.{Module, Sequential, Linear, Conv2d,
  MaxPool2d, BatchNorm1d, ReLU, Flatten, CrossEntropyLoss}`,
  `torch.optim.{Adam, lr_scheduler.{StepLR, CosineAnnealingLR}}`,
  `torch.nn.init.{xavier_uniform_, kaiming_uniform_}`,
  `torch.utils.data.{DataLoader, TensorDataset}`,
  `torch.set_num_threads`, `torch.manual_seed`.
- **Imbalanced data**: `imblearn.over_sampling.{SMOTE, ADASYN,
  RandomOverSampler}`, `imblearn.under_sampling.{RandomUnderSampler,
  NearMiss}`, `imblearn.combine.{SMOTETomek, SMOTEENN}`,
  `imblearn.pipeline.Pipeline`.
- **Bayesian / probabilistic**: `numpyro.{sample, distributions.{Normal,
  HalfNormal, HalfCauchy, InverseGamma}}`,
  `numpyro.infer.{MCMC, NUTS, Predictive}`,
  `jax.{random.PRNGKey, config.update}`.
- **Frequentist baselines**: `statsmodels.OLS.fit(...).conf_int`,
  sklearn `LinearRegression`, `Lasso`, `Ridge`, `LogisticRegression`,
  `MLPClassifier`, `StandardScaler`,
  `precision_recall_curve`, `average_precision_score`,
  `roc_auc_score`, `f1_score`.

For per-API explanations see each experiment's `summary.md`; for what
each one *actually did* on the data see the `postmortem.md`.
