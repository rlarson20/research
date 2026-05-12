# Experiment 2 — Regression + regularization paths + feature selection

## Goal

(1) Compare seven regressors on a moderately-sized tabular regression task.
(2) Plot the LASSO regularization path and contrast it with `LassoCV`'s
chosen alpha. (3) Evaluate four feature-selection strategies on the same
augmented feature matrix and report which ones actually drop the
deliberately-injected noise columns.

## Dataset

`sklearn.datasets.load_diabetes()` — 442 samples, 10 features, regression
target (disease progression one year later). `fetch_california_housing` was
the plan-A dataset but the sandbox blocks network downloads (HTTP 403); the
plan's documented fallback was diabetes + engineered features.

Augmentation pipeline (built inline by `build_augmented_matrix`):
- Start with the 10 original features (already scaled by sklearn).
- Append degree-2 interaction terms via `PolynomialFeatures(degree=2,
  interaction_only=True, include_bias=False)` — yields `C(10,2) = 45`
  pairwise products.
- Append 10 i.i.d. standard-normal "noise" columns labelled `noise_0…9`.
- Final matrix: 442 × 65, with **10 columns of pure noise** that any decent
  feature selector should drop.

## sklearn APIs used and why

### Regressors compared (defaults except where noted)
- `LinearRegression` — no regularization; baseline.
- `Ridge(alpha=1.0)` — L2; shrinks coefficients toward zero, never to zero.
- `Lasso(alpha=…)` — L1; produces sparse solutions, can zero coefficients.
- `ElasticNet(alpha=…, l1_ratio=0.5)` — convex combination of L1+L2; mixes
  selection and shrinkage.
- `GradientBoostingRegressor` — sequential additive regression trees.
- `RandomForestRegressor(n_estimators=300)` — bagged regression trees.
- `SVR(kernel="rbf")` — kernel regression; trained on a 300-row subsample
  for speed because RBF SVR is O(n²) in training.

### Regularization paths
- `lasso_path(X, y, alphas=...)` — computes the full sequence of LASSO
  coefficient vectors along a logarithmic alpha grid. Returns
  `(alphas, coefs, _)` where `coefs.shape == (n_features, n_alphas)`. Used
  to plot all 65 coefficient trajectories vs `log(alpha)` and visualize
  the order in which features enter the model as alpha decreases.
- `LassoCV(cv=5)` — fits LASSO at many alphas and picks the one with the
  lowest 5-fold MSE. Its `.alpha_` attribute is overlaid on the path plot.

### Feature selection methods
All four selectors are evaluated by the **downstream test-RMSE of a
`Ridge(alpha=1.0)`** that's fit on the selected feature subset (≤10 cols)
and scored on a held-out 20% test set. This makes the metric end-to-end:
how much performance does each selection rule retain?

- `SelectKBest(score_func=mutual_info_regression, k=10)` — picks the 10
  features with the highest mutual information against the target. Captures
  nonlinear monotonic associations the linear models can't see directly.
- `RFE(estimator=Ridge(), n_features_to_select=10)` — recursive feature
  elimination: fits Ridge, drops the smallest-coefficient feature, refits,
  repeats. Greedy backward selection driven by linear-model coefficients.
- `SelectFromModel(LassoCV(...))` — selects the features whose LASSO
  coefficient is non-zero at the CV-chosen alpha. Embeds selection in
  fitting.
- `SelectFromModel(RandomForestRegressor(...))` — selects features whose
  tree-importance exceeds the mean. Captures non-linear/interaction signal,
  but importance can be biased toward high-cardinality / continuous features.

The "noise-survival rate" of each method (fraction of selected features that
are noise) is reported alongside RMSE.

### Reporting / scoring
- `cross_val_score(..., scoring="neg_root_mean_squared_error")` — sklearn
  returns negative for "higher is better" convention; we negate back to
  report RMSE.
- `r2_score`, residual scatter — sanity-check the best regressor.

## How to read the outputs

- `cv_table.txt` — per-model mean ± std RMSE & R² across 5-fold CV.
- `lasso_path.png` — 65 coefficient trajectories vs `log(alpha)` with the
  `LassoCV`-chosen alpha marked. Features whose curve stays at zero across
  the whole range are the ones LASSO never bothers with.
- `feature_selection.png` — bar chart: test-RMSE of downstream Ridge after
  each of the four selectors, with the noise-survival fraction annotated.
- `residuals.png` — predicted vs. actual + residuals-vs-fitted for the
  best-CV regressor.
