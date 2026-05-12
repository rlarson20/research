# Experiment 4 — Imbalanced classification: class weights, threshold tuning, calibration

## Goal

On a synthetic 95/5 imbalanced binary classification problem, explore three
classic interventions for the minority class:

1. **Cost-sensitive learning** via `class_weight="balanced"`.
2. **Threshold tuning** — picking the decision threshold that maximizes
   minority-class F1 instead of using the default 0.5.
3. **Probability calibration** — using `CalibratedClassifierCV` to fix the
   over-confident probabilities of an uncalibrated Random Forest, and
   measuring with reliability diagrams + Brier score.

## Dataset

`sklearn.datasets.make_classification(n_samples=10_000, n_features=20,
n_informative=8, weights=[0.95, 0.05], class_sep=0.8, random_state=42)`.
Synthetic so the class-balance and difficulty are precisely controllable.
~500 minority-class samples in 10k total.

## sklearn APIs used and why

### Cost-sensitive learning
- `class_weight="balanced"` — reweights samples inversely proportional to
  class frequency, so the per-sample loss treats minority mistakes as
  ~19× more costly than majority mistakes (`p_majority / p_minority`).
  Supported by LogReg, RF, SVC, GBM (newer), etc.
- The study fits each model in both `class_weight=None` and
  `"balanced"` modes and contrasts minority-class precision/recall/F1
  at threshold 0.5.

### Threshold tuning
- `precision_recall_curve(y_true, y_score)` — sweeps every unique
  predicted probability as a threshold and returns matched
  `(precision, recall, threshold)` arrays. F1 is computed for each
  threshold; the argmax gives the threshold that maximizes minority F1.
- Why this matters: with imbalance, threshold 0.5 almost never matches
  the F1-optimal operating point. The score below picks the
  F1-maximizing threshold on the *test* set after training, which is a
  slight optimism (in production you'd pick on a validation set), but
  fine for illustration.

### Calibration
- `CalibratedClassifierCV(base, method="isotonic", cv=5)` — fits the base
  classifier on each of 5 folds and learns an isotonic regression that
  maps its raw scores to calibrated probabilities on the held-out fold.
  Non-parametric, needs lots of data.
- `CalibratedClassifierCV(base, method="sigmoid", cv=5)` — same wrapper,
  but fits a Platt-style logistic regression to the scores. Parametric,
  works on small data, biases toward sigmoidal calibration curves.
- `CalibrationDisplay.from_estimator(model, X, y, n_bins=10, ax=ax)` —
  plots the reliability diagram: mean predicted probability per bin
  against the actual fraction of positives in that bin. Perfectly
  calibrated model = diagonal.
- `brier_score_loss(y_true, y_proba)` — mean squared error between
  predicted probability and the binary outcome. A scalar summary of
  calibration quality (lower is better). Penalizes both miscalibration
  and discrimination loss.

## How to read the outputs

- `class_weight_table.txt` — minority-class precision/recall/F1 for each
  model in each weighting mode.
- `pr_curve.png` — precision-recall curve of the strongest model, with
  the F1-optimal threshold annotated.
- `threshold_table.txt` — minority metrics at threshold 0.5 vs. the
  F1-optimal threshold.
- `calibration.png` — three-curve reliability diagram (uncalibrated RF,
  isotonic-calibrated RF, sigmoid-calibrated RF).
- `brier.txt` — Brier score for each of the three.
