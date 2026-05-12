# Experiment 1 — Classifier shootout + interpretability

## Goal

Compare six off-the-shelf classifiers on a small, fully-numeric binary dataset
using a shared preprocessing pipeline and cross-validation; hyperparameter-tune
the top two; then explain the winning model with the inspection APIs.

## Dataset

`sklearn.datasets.load_breast_cancer()` — 569 samples, 30 numeric features,
binary target (malignant vs. benign). Ships with sklearn, no download.

## sklearn APIs used and why

### Plumbing
- `train_test_split(..., stratify=y)` — produces a held-out test set with the
  same class-balance as the full dataset. Stratification matters even for
  near-balanced data because it lowers variance of the test estimate.
- `Pipeline([("scaler", StandardScaler()), ("clf", est)])` — bundles
  preprocessing with the estimator so scaling parameters are fit only on the
  training fold during CV. This prevents the classic leak where a scaler fit
  on the entire dataset gives the test set information about training-fold
  statistics.
- `StandardScaler` — zero-mean / unit-variance per feature. Required by
  distance-based and gradient-based learners (KNN, SVM, MLP, LogReg).
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` — keeps the
  class ratio constant across CV folds, again reducing estimator variance.
- `cross_val_score(pipe, X, y, cv=skf, scoring=...)` — runs the full
  pipeline on each fold and returns a vector of scores per fold.

### Models compared (defaults except where noted)
- `LogisticRegression(max_iter=5000)` — linear baseline; well-calibrated
  out of the box on this dataset.
- `RandomForestClassifier(n_estimators=300, random_state=42)` — bagged
  decision trees; nonlinear, robust, no scaling needed but kept inside the
  pipeline for parity.
- `GradientBoostingClassifier(random_state=42)` — sequential additive trees;
  often the strongest tabular model in this size regime.
- `SVC(kernel="rbf", probability=True, random_state=42)` — kernel margin
  classifier; `probability=True` enables ROC-AUC at the cost of an internal
  Platt-scaling CV (slow).
- `KNeighborsClassifier()` — instance-based; included as a sanity baseline.
- `MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)` —
  small neural net; benefits strongly from scaling.

### Tuning
- `GridSearchCV(pipe, param_grid, cv=skf, scoring="f1", n_jobs=-1)` — applied
  to the top-2 CV finishers. The grids are intentionally small (10-20 fits per
  model) so this stays a minute, not an hour.

### Interpretability
- `permutation_importance(model, X_test, y_test, n_repeats=20, random_state=42)`
  — model-agnostic; shuffles one feature at a time on the held-out set and
  measures the drop in score. Honest because it doesn't refit and uses test data.
- `PartialDependenceDisplay.from_estimator(model, X, [f1, f2], kind="average")`
  — shows the marginal effect of each chosen feature on the predicted class
  probability with the others integrated out.
- `learning_curve(model, X, y, cv=skf, train_sizes=...)` — train/validation
  score vs. training-set size. Reveals whether the model is data-limited
  (curves still rising) or capacity-limited (curves plateaued and converged).

### Reporting
- `accuracy_score`, `f1_score`, `roc_auc_score` — standard binary metrics.
- `ConfusionMatrixDisplay.from_estimator` — single-call confusion-matrix plot.

## How to read the outputs

- `cv_table.txt` — markdown-ish table of mean ± std accuracy & F1 across the
  five CV folds; sort key for picking the top-2.
- `confusion_matrix.png` — tuned winner on the held-out test set.
- `perm_importance.png` — top-10 features by permutation importance (bars are
  the mean drop in F1, error bars the std across 20 shuffles).
- `pdp.png` — partial-dependence plots for the two most-important features.
- `learning_curve.png` — diagnostic: gap = high variance, low ceiling = bias.
