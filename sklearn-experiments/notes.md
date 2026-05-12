# Working notes

Diary appended as work proceeds. Each experiment lives in its own subfolder
with a `summary.md` (APIs used + why) and a `postmortem.md` (what happened).

## Setup

- `uv init --no-readme --no-workspace --bare`, then
  `uv add scikit-learn numpy pandas matplotlib`.
- Versions installed: scikit-learn 1.8.0, numpy 2.4.4, pandas 3.0.3, matplotlib 3.10.9.
- Confirmed `fetch_california_housing` is blocked by the sandbox network policy
  (HTTP 403). Switched Experiment 2 to `load_diabetes()` + engineered features
  (the fallback already noted in the plan).

## Experiment 1 — classifier shootout

Ran six classifiers under a shared `StandardScaler` pipeline, 5-fold
stratified CV, on `load_breast_cancer()`. `LogisticRegression(C=0.1)` won
both CV and the tuned head-to-head on the held-out test set
(acc=0.9737, F1=0.9793, AUC=0.996). Permutation importance on the held-out
test set was clean: the "worst …" feature variants dominate, matching the
clinical intuition that the most-malignant region of a tumor is more
diagnostic than the mean. Learning curve converges tight above 0.97 by
n≈200 — capacity-saturated.

Gotchas: sklearn 1.8 deprecated `penalty=` on `LogisticRegression`; dropped
it from the grid. `SVC(probability=True)` is slow because of the internal
Platt-scaling CV — kept it because we need probabilities for AUC.

## Experiment 2 — regression paths + feature selection

Augmented `load_diabetes()` to 65 columns (10 originals + 45 pairwise
interactions + 10 i.i.d. noise). `ElasticNet` wins CV (RMSE 57.2);
`LassoCV(α=3.82)` retains 12 features with only one noise survivor.
`SelectFromModel(LassoCV)` is the only selector to beat the no-selection
Ridge baseline (49.1 vs 52.4). `SelectKBest(MI)` and `RFE(Ridge)` cleanly
exclude *all* 10 noise columns but lose to LASSO on RMSE because the fixed
`k=10` cap is below the 12-feature optimum. `SVR(kernel="rbf")` on defaults
delivers R²≈0 — needs tuning.

`sklearn.metrics.mean_squared_error(..., squared=False)` is deprecated;
used the new `root_mean_squared_error` from the start.

## Experiment 3 — unsupervised + anomaly detection

`load_digits()` (1797 × 64). Agglomerative+Ward wins clustering by ARI
(0.664) vs KMeans (0.531); both have near-identical silhouette (~0.13),
a clean example of internal-vs-external metric disagreement. DBSCAN is
the wrong tool here — even with a silhouette-grid-chosen `eps=3.24`,
82% of points are labeled noise (curse of dimensionality on raw pixel
space). t-SNE separates the digits into ten clear islands; PCA's 2D
projection leaves everything as one diffuse blob — classic.

Anomaly detection on digit-1 + 5% injected anomalies: `EllipticEnvelope`
0.988, `IsolationForest` 0.981, `LOF` 0.966, `OneClassSVM` 0.861. MCD
covariance beats everything because digit-1 pixels are roughly
elliptical-Gaussian. OneClassSVM at defaults trails — needs a small
grid like SVR did.

Gotchas: the canonical `argmax(diff(k-distance))` DBSCAN eps heuristic
picks `eps=30.5` here (one huge cluster) because the largest single
jump is at the tail caused by a handful of true outliers. Replaced
with a silhouette grid over the 10-90th percentile of 5-NN distances.
`LocalOutlierFactor` has no `predict` by default — its scores live on
`negative_outlier_factor_`. Don't include `-1` (noise) when computing
silhouette on DBSCAN output.

## Experiment 4 — imbalanced + calibration

Synthetic 95/5 binary problem (10k × 20 via `make_classification`).
Headline: threshold tuning matters more than anything else. Moving
the GBM threshold from 0.5 → 0.147 jumped minority-F1 from 0.286 to
0.445 — biggest single improvement, no retraining.

`class_weight="balanced"` is not universally good: LogReg gets a 7×
F1 lift (recall: balancing rescues a model that was always predicting
0); RF *loses* F1 at threshold 0.5 (its precision goes up but recall
crashes); GBM doesn't accept the kwarg. AUC improves under balancing
for both even when threshold-0.5 F1 doesn't — that's the textbook
"AUC measures ranking, F1 measures the operating point" lesson.

Calibration: RF Brier 0.0356 → 0.033 with both isotonic and sigmoid
(~7% improvement). RF was only mildly over-confident on this dataset.
Reliability diagram shows uncalibrated RF systematically *under*-
predicting on the high end and over-predicting on the low end — the
classic "bootstrap smearing" pattern.

Gotcha: `GradientBoostingClassifier(class_weight=...)` is silently
ignored — the constructor doesn't accept the kwarg and there's no
warning. My first run gave duplicate identical rows; fixed with an
explicit `supports_cw` flag.
