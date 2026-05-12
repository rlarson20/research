# Experiment 2 — Postmortem

## Headline

On `load_diabetes()` augmented with 45 pairwise interactions + 10 noise
columns (X.shape = 442 × 65):

- **ElasticNet** beat every other regressor in 5-fold CV. The L1+L2 mix
  handles the noise and the redundant interaction terms better than plain
  Ridge or pure LASSO at default alphas.
- **LASSO's regularization path** confirms it: at the CV-chosen
  `α=3.82`, 53 of the 65 coefficients are zero. Only one of the 10 injected
  noise columns survives.
- **`SelectFromModel(LassoCV)` was the best feature selector**, beating
  even no-selection baseline Ridge (test-RMSE 49.1 vs 52.4).

## Cross-validated performance (full 65-feature matrix)

| Model       | rmse_mean | rmse_std | r2_mean | r2_std |
|---          |---        |---       |---      |---     |
| ElasticNet  | 57.229    | 2.538    | 0.445   | 0.045  |
| Lasso       | 58.028    | 3.408    | 0.430   | 0.050  |
| RandomForest| 59.579    | 2.396    | 0.398   | 0.052  |
| GradBoost   | 60.009    | 1.825    | 0.389   | 0.062  |
| Ridge       | 61.323    | 4.860    | 0.365   | 0.062  |
| Linear      | 64.110    | 5.572    | 0.306   | 0.076  |
| SVR-rbf     | 78.401    | 6.397    | -0.025  | 0.065  |

(SVR-rbf was trained on a 300-row subsample; even at defaults it's the
worst — the RBF kernel + default `C/gamma` is a poor fit for this scale
and would need a tuning grid to be competitive.)

## LASSO path

`LassoCV(cv=5)` chose `α = 3.8238` (out of a `np.logspace(-3, 1, 80)` grid).
Of the 65 features:

- 12 features retain non-zero coefficients at the chosen alpha.
- Of those 12, **1 is a noise column** — false-positive rate 1/10 = 10%.
- True positives: 11 of the 55 signal features survived.

The `lasso_path.png` plot shows the path with noise features rendered in
light gray; they cluster around zero across the whole alpha range, exactly
as desired.

## Feature selection results

Downstream `Ridge(α=1.0)` test-RMSE after each selector:

| Selector                   | test-RMSE | # selected | # noise selected |
|---                         |---        |---         |---               |
| no-selection baseline      | 52.386    | 65         | 10               |
| `SelectFromModel(LassoCV)` | **49.119**| 12         | 1                |
| `SelectKBest(MI, k=10)`    | 52.595    | 10         | 0                |
| `RFE(Ridge)` (k=10)        | 53.053    | 10         | 0                |
| `SelectFromModel(RF)`      | 53.151    | 6          | 1                |

**Both `SelectKBest(MI)` and `RFE(Ridge)` perfectly excluded all noise
columns.** They scored marginally worse than the LASSO selector because
their fixed `k=10` cap is just below the LASSO's preferred 12-feature
solution — they drop two informative interactions that LASSO keeps.

The Random-Forest selector kept only 6 features (mean-importance threshold
is conservative for high-dim, low-signal data) and slightly underperformed.

## Surprises and gotchas

- **ElasticNet > Lasso at default alphas.** ElasticNet's L2 component
  stabilizes the highly-correlated interaction columns LASSO would otherwise
  arbitrarily pick one of.
- **LASSO selector at the embedded alpha beat plain Ridge with no
  selection.** The 3-point RMSE drop (52.4 → 49.1) is a clean reminder that
  noise features actively hurt linear models even with L2 — they steal
  coefficient mass.
- **`mean_squared_error(..., squared=False)` is deprecated** in sklearn 1.8
  in favour of the dedicated `root_mean_squared_error` function. Used the
  latter from the start.
- **SVR with defaults is awful.** `kernel="rbf"` + default `C=1.0,
  gamma="scale"` produces R² ≈ 0. With this small a dataset, a real SVR
  exercise would demand a `GridSearchCV` over `(C, gamma, epsilon)`. Left
  it on defaults to demonstrate why ungated SVR is a bad first guess.

## What I'd try next

1. Tune ElasticNet via `ElasticNetCV` to confirm the picked `l1_ratio` and
   compare to the LASSO-selected Ridge pipeline.
2. Replace `SelectKBest(MI)` k=10 with `SelectKBest(MI, k=12)` to disentangle
   "fewer features" from "MI vs LASSO scoring".
3. Try `HistGradientBoostingRegressor` — its sparse-aware splits should be
   robust to noise columns natively, often making selection unnecessary.
4. Use `PartialDependenceDisplay` on the ElasticNet to inspect the strongest
   interaction terms it kept.
