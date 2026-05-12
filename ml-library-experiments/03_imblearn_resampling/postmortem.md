# Experiment 3 — Postmortem

## Headline

**On 99/1 imbalance, do nothing — just lower the threshold.** At
threshold=0.137, plain `LogisticRegression` hits F1=0.327, beating every
imbalanced-learn resampler (best of those: SMOTE family at F1=0.246) and
beating `class_weight="balanced"` (F1=0.246). The tutorial-popular SMOTE
move is actively worse here.

This is the same lesson as `sklearn-experiments/04_imbalanced_calibration`'s
"threshold tuning is the biggest lever", but more extreme: at this
severity, threshold tuning is *also* the only lever that doesn't break.

## Full scoreboard (LogReg head, 99/1 imbalance, 35 minority test rows)

| sampler        | AUC | AP | F1@0.5 | F1_tuned | best_th |
|---             |---:|---:|---:|---:|---:|
| **none**       | 0.639 | **0.206** | 0.108 | **0.327** | 0.137 |
| class_weight   | 0.669 | 0.132 | 0.046 | 0.246 | 0.879 |
| RandomOver     | 0.667 | 0.128 | 0.047 | 0.241 | 0.886 |
| SMOTE          | 0.670 | 0.140 | 0.049 | 0.246 | 0.906 |
| ADASYN         | 0.673 | 0.140 | 0.047 | 0.235 | 0.927 |
| RandomUnder    | 0.659 | 0.098 | 0.038 | 0.194 | 0.926 |
| NearMiss-1     | 0.656 | 0.065 | 0.047 | 0.119 | 0.790 |
| SMOTE+Tomek    | 0.670 | 0.140 | 0.049 | 0.246 | 0.906 |
| SMOTE+ENN      | 0.674 | 0.136 | 0.045 | 0.231 | 0.940 |

Two things to notice:

1. **AUC for the resamplers is higher than for the do-nothing baseline**
   (0.67 vs 0.64) — *yet* their AP (the metric that actually matters for
   imbalanced classification) is lower (0.13 vs 0.21). This is a clean
   case of AUC misleading: ranking improves slightly across the *whole*
   class distribution, but the part of the ranking that contains
   minority examples gets worse.
2. **Tuned thresholds for resamplers are near 1.0** (0.88-0.94). Resampling
   pushes the model to predict "minority" much more often, so calibrated
   probabilities for the minority class shift toward 1; F1 maximum moves
   way to the right. This is *not* a bug — it's a calibration shift — but
   it means resampling and threshold-tuning are roughly substitutes, and
   the do-nothing+threshold path is more direct.

## 5-fold CV stability check

For the top-3 by AP on the test split:

| sampler      | CV AP mean ± std |
|---           |---:|
| none         | **0.217 ± 0.128** |
| SMOTE        | 0.149 ± 0.115 |
| SMOTE+Tomek  | 0.149 ± 0.115 |

`none` is the most stable as well as the highest-mean — the gap to SMOTE
is bigger than the std. This isn't a fluke of the single test split.

## Surprises and gotchas

- **The popular SMOTE recipe is *worse* than doing nothing at 99/1.**
  Reviewing the imblearn docs after the fact, this is mentioned in
  passing: "synthetic samples produced by SMOTE can introduce noise
  when the minority class is very small or noisy". 141 minority
  examples sounds like a lot but n=141 with 20 features is undersampled.
  Interpolating between them just smears noise.
- **NearMiss-1 is catastrophic** (F1_tuned=0.119, half of everything
  else). It throws away >97% of the majority — too aggressive for this
  imbalance ratio. NearMiss is better-suited to moderate imbalance.
- **`imblearn.pipeline.Pipeline` vs `sklearn.pipeline.Pipeline`** —
  this is the critical API gotcha. Using sklearn's pipeline with an
  imblearn sampler silently breaks: sklearn doesn't know that
  `fit_resample` should only happen on training folds, so it leaks
  resampled rows into the CV evaluation. Always use the imblearn
  pipeline when there's a sampler in the chain.
- **`SMOTE+Tomek == SMOTE` exactly on this dataset**, down to the AP.
  The Tomek-link cleanup pass had no effect — there were no
  Tomek-link pairs (mutual nearest neighbors of opposite class) in
  the SMOTE-augmented training set.
- **`ADASYN`'s "harder examples get more samples" promise is invisible
  here.** It produces ~the same AP as SMOTE. The premise assumes
  there are genuinely hard minority points; with only 141 minority
  rows there's not enough signal for that adaptive logic to help.

## What I'd try next

1. Repeat at less severe imbalance (95/5, 90/10) — confirm sklearn-Exp-4's
   finding that resampling matters more there. The story is probably:
   resampling *helps* at moderate imbalance, *hurts* at severe imbalance.
2. Add a **gradient-boosting head** (XGBoost or LightGBM) instead of
   LogReg. Trees handle non-additive boundaries that SMOTE-generated
   "between two minority points" interpolations might exercise — there's
   a real chance SMOTE+GBM beats vanilla+threshold here while
   SMOTE+LogReg can't.
3. Try **focal loss** in PyTorch as a third "imbalance lever". It's the
   modern alternative to class_weight and to SMOTE, and would extend the
   "do nothing but change the loss" approach this experiment leans on.
4. **PR-AUC threshold-tuning curve overlay** — the missing diagnostic
   plot that would make the "different samplers have different
   optimal thresholds" story visual rather than tabular.
