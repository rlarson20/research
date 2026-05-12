# Scikit-learn experiments

Four small, self-contained experiments that exercise a broad slice of
`scikit-learn`: supervised tabular classification with interpretability,
regularized regression with feature selection, unsupervised clustering with
dimensionality-reduced visualization plus anomaly detection, and imbalanced
classification with calibration and threshold tuning.

Each experiment lives in its own subfolder containing:

- `run.py` — the experiment script (`uv run python NN_*/run.py`)
- `summary.md` — *what* sklearn APIs are used and *why*
- `postmortem.md` — *what actually happened* when the script ran
- `outputs/` — PNG plots and tabular `.txt` summaries

## Reproducing

```
cd sklearn-experiments
uv sync
uv run python 01_classifier_shootout/run.py
uv run python 02_regression_paths/run.py
uv run python 03_unsupervised/run.py
uv run python 04_imbalanced_calibration/run.py
```

Tested with `scikit-learn==1.8.0`, `numpy==2.4.4`, `pandas==3.0.3`,
`matplotlib==3.10.9` (see `pyproject.toml`). Every script uses
`random_state=42` and runs in under ~30s on a single CPU.

## Experiments at a glance

| # | Folder | Topic | Dataset | Headline result |
|---|---|---|---|---|
| 1 | [`01_classifier_shootout/`](01_classifier_shootout/) | Classifier comparison + tuning + interpretability | `load_breast_cancer` (569 × 30) | `LogisticRegression(C=0.1)` wins both CV and test (F1=0.979, AUC=0.996), beating RF/GBM/SVC/KNN/MLP. |
| 2 | [`02_regression_paths/`](02_regression_paths/) | Regression + regularization paths + feature selection | `load_diabetes` augmented to 65 cols (45 interactions + 10 noise) | `ElasticNet` wins CV; `SelectFromModel(LassoCV)` is the only selector that *beats* the no-selection Ridge baseline (RMSE 49.1 vs 52.4). |
| 3 | [`03_unsupervised/`](03_unsupervised/) | Clustering + dim-reduction + anomaly detection | `load_digits` (1797 × 64) | Agglomerative+Ward dominates by ARI (0.66 vs 0.53 for KMeans); DBSCAN fails on 64-D pixels (82% noise). t-SNE separates ten clean islands where PCA shows one blob. `EllipticEnvelope` tops anomaly AUC at 0.988. |
| 4 | [`04_imbalanced_calibration/`](04_imbalanced_calibration/) | Imbalanced classification: class weights, threshold tuning, calibration | `make_classification(weights=[0.95, 0.05])` | Threshold tuning is the single biggest lever (F1 0.286 → 0.445 by dropping threshold from 0.5 → 0.147). `class_weight="balanced"` rescues LogReg but hurts RF at threshold 0.5. Isotonic/sigmoid calibration shave ~7% off RF Brier score. |

## Cross-experiment observations

### 1. Linear models are not dead

Two of the four experiments produced winners that were either fully linear
(LogReg on breast cancer) or essentially-linear (ElasticNet on the
augmented diabetes matrix). On small, well-conditioned tabular data,
`StandardScaler` + a regularized linear model with a tiny CV-tuned penalty
is hard to beat — even by gradient boosting.

### 2. AUC and F1 don't tell the same story

Experiment 4 made this concrete: `RandomForest(class_weight="balanced")`
improved AUC from 0.913 → 0.928 *while losing F1 at threshold 0.5* (0.212
→ 0.107). AUC is invariant to threshold; threshold-dependent metrics
(precision/recall/F1) are not. **If you care about a particular operating
point, tune the threshold separately from the model.**

### 3. Default sklearn parameters are a mixed bag

- `RBF SVR/SVC` at defaults is reliably bad on anything non-trivial (R² ≈
  0 in Exp 2; trailed everything else in Exp 3 anomaly AUC). It always
  wants a `(C, gamma)` grid.
- `DBSCAN` has no sensible default `eps`; the popular k-distance-knee
  heuristic itself fails on high-D data (Exp 3). Plan for a silhouette
  grid.
- `GradientBoostingClassifier` silently ignores `class_weight=` — caused
  duplicate-row bug in Exp 4 first run.

### 4. The sklearn 1.8 deprecation surface is real

Hit two during this work:

- `LogisticRegression(penalty=...)` is deprecated in favor of the
  `l1_ratio` / `C` parameterization.
- `mean_squared_error(..., squared=False)` is deprecated; use the
  dedicated `root_mean_squared_error`.

Both throw `FutureWarning`s on import, not at use-site — if you're
running CV at scale the warning blizzard is loud.

### 5. The inspection APIs are the most underrated tools in sklearn

`permutation_importance`, `PartialDependenceDisplay`, and `learning_curve`
together took the breast-cancer model from "97% F1 black box" to "I can
see what features drive the prediction, in what direction, and whether
more data would help". These are model-agnostic and live one import away.

## sklearn surface area covered

A non-exhaustive list of what these four scripts touch:

- **Datasets**: `load_breast_cancer`, `load_diabetes`, `load_digits`,
  `make_classification`.
- **Preprocessing**: `StandardScaler`, `PolynomialFeatures`.
- **Pipelines**: `Pipeline`.
- **Model selection**: `train_test_split`, `KFold`, `StratifiedKFold`,
  `cross_val_score`, `GridSearchCV`, `learning_curve`.
- **Linear models**: `LogisticRegression`, `LinearRegression`, `Ridge`,
  `Lasso`, `ElasticNet`, `LassoCV`, `lasso_path`.
- **Trees / ensembles**: `RandomForestClassifier`,
  `RandomForestRegressor`, `GradientBoostingClassifier`,
  `GradientBoostingRegressor`, `IsolationForest`.
- **SVM / neighbors / NN**: `SVC`, `SVR`, `OneClassSVM`,
  `KNeighborsClassifier`, `NearestNeighbors`, `LocalOutlierFactor`,
  `MLPClassifier`.
- **Clustering**: `KMeans`, `AgglomerativeClustering`, `DBSCAN`.
- **Decomposition / manifold**: `PCA`, `TSNE`.
- **Covariance**: `EllipticEnvelope`.
- **Feature selection**: `SelectKBest`, `mutual_info_regression`, `RFE`,
  `SelectFromModel`.
- **Inspection**: `permutation_importance`,
  `PartialDependenceDisplay`.
- **Calibration**: `CalibratedClassifierCV`, `CalibrationDisplay`.
- **Metrics**: `accuracy_score`, `f1_score`, `precision_score`,
  `recall_score`, `roc_auc_score`, `precision_recall_curve`,
  `ConfusionMatrixDisplay`, `silhouette_score`, `adjusted_rand_score`,
  `root_mean_squared_error`, `r2_score`, `brier_score_loss`.

For per-API explanations see each experiment's `summary.md`; for what
each one *actually did* on the data see the `postmortem.md`.
