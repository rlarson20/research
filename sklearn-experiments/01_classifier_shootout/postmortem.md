# Experiment 1 — Postmortem

## Headline

On the breast-cancer dataset (569 × 30), a plain L2-regularised
`LogisticRegression(C=0.1)` won the shootout, beating Random Forest, Gradient
Boosting, SVC-RBF, KNN, and an MLP on both cross-validated F1 and held-out
test F1.

## Cross-validation results (5-fold stratified, on the 80% training split)

| Model         | acc_mean | acc_std | f1_mean | f1_std |
|---            |---       |---      |---      |---     |
| LogReg        | 0.9780   | 0.0098  | 0.9825  | 0.0078 |
| SVC-rbf       | 0.9692   | 0.0146  | 0.9756  | 0.0113 |
| MLP           | 0.9692   | 0.0082  | 0.9756  | 0.0064 |
| RandomForest  | 0.9648   | 0.0176  | 0.9718  | 0.0144 |
| KNN           | 0.9626   | 0.0112  | 0.9705  | 0.0091 |
| GradBoost     | 0.9516   | 0.0149  | 0.9614  | 0.0121 |

LogReg was strongest by both mean F1 and stability (lowest std). The two tree
ensembles trail — interesting because tabular wisdom usually favours them, but
this dataset is small, low-noise, and the classes are nearly linearly
separable in the scaled feature space.

## Tuned head-to-head on the held-out test set

The top-2 by CV F1 (LogReg, MLP) went into `GridSearchCV`.

| Model  | best_params                                                  | test_acc | test_f1 | test_auc |
|---     |---                                                           |---       |---      |---       |
| LogReg | `{C: 0.1}`                                                   | 0.9737   | 0.9793  | 0.9957   |
| MLP    | `{alpha: 1e-3, learning_rate_init: 5e-3}`                    | 0.9561   | 0.9645  | 0.9940   |

LogReg wins on every metric. AUC of 0.996 is at the noise floor of this
test split (114 samples).

## Interpretability findings

Top permutation-importance features for the winning LogReg:

```
worst texture        0.0172
worst smoothness     0.0159
worst concave points 0.0115
mean concave points  0.0113
worst concavity      0.0112
radius error         0.0106
worst radius         0.0103
worst area           0.0098
worst perimeter      0.0093
mean area            0.0079
```

The "worst …" columns (per-tumor max value of each feature) dominate, which
matches clinical intuition — the most-malignant region of the tumor is more
diagnostic than the mean. `PartialDependenceDisplay` on the top two confirms
the model has learned the expected monotonic-ish "higher → more likely
malignant" effect for both.

The learning curve (`outputs/learning_curve.png`) shows train and CV F1
converging tightly above 0.97 by 200 training samples, with the band barely
moving past that — i.e. the model is capacity-saturated for this dataset, and
more data wouldn't help much.

## Surprises and gotchas

- **LogReg beat the ensembles cleanly.** A reminder that linear models with
  proper scaling and a touch of regularization are still very competitive on
  small, well-conditioned tabular data.
- **`probability=True` on `SVC` is slow** — it triggers an internal CV for
  Platt scaling on every fit. Kept it in because it's the only way to get AUC
  out of an RBF SVC.
- **`PartialDependenceDisplay.from_estimator` does not accept a pre-fit
  `GridSearchCV` directly when you pass features as `np.str_` names**;
  passing integer indices and `feature_names=…` works cleanly and is what
  the script does.
- **`FutureWarning` blizzard**: sklearn 1.8 deprecates the explicit
  `penalty=` kwarg for `LogisticRegression` in favour of `l1_ratio` /
  `C`. Removed `penalty` from the grid; warnings cleared.

## What I'd try next

1. Replace `SVC(probability=True)` with `SVC` + isotonic calibration via
   `CalibratedClassifierCV` — faster and usually better-calibrated.
2. Add `HistGradientBoostingClassifier` (the new default histogram booster);
   it's typically faster than `GradientBoostingClassifier` and competitive
   with LightGBM.
3. Sample with `RandomizedSearchCV` over a larger grid for the MLP to see
   if it can catch up to LogReg with more hidden-unit / regularization
   exploration.
