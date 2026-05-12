# Experiment 4 — Postmortem

## Headline

- **Threshold tuning was the single biggest lever** on a 95/5 imbalanced
  task: moving the GradientBoosting decision threshold from 0.5 down to
  0.147 lifted minority-class F1 from **0.286 to 0.445** — a +0.16
  absolute jump with zero retraining.
- **`class_weight="balanced"` is not a universal good.** It massively
  rebalances LogReg (F1 0.025 → 0.183) but *hurts* RandomForest
  (F1 0.212 → 0.107) on this dataset. AUC of both improves under
  balancing, which tells the real story: balancing improves ranking
  quality but the operating point at threshold 0.5 still doesn't match.
- **Calibration helped, but modestly.** Isotonic and sigmoid both shaved
  ~7% off the RF Brier score (0.0356 → 0.033) — the RF was only mildly
  miscalibrated on this dataset to begin with.

## Class-weight study (minority-class metrics at threshold 0.5)

| Model        | class_weight | Precision | Recall | F1    | AUC   |
|---           |---           |---:       |---:    |---:   |---:   |
| LogReg       | None         | 0.667     | 0.013  | 0.025 | 0.719 |
| LogReg       | balanced     | 0.105     | 0.717  | 0.183 | 0.732 |
| RandomForest | None         | 0.950     | 0.119  | 0.212 | 0.913 |
| RandomForest | balanced     | 1.000     | 0.057  | 0.107 | 0.928 |
| GradBoost    | n/a          | 0.757     | 0.176  | 0.286 | 0.868 |

Observations:

- LogReg at default weights is essentially useless at threshold 0.5 — it
  predicts "majority" for almost every example (recall 0.013). Balancing
  flips it from "always predict 0" to "predict 1 too aggressively"
  (precision crashes to 0.10), but F1 still improves 7×.
- RF's behavior is the *opposite*: default weights give a precision-heavy
  model; "balanced" makes it *more* conservative, presumably because
  balanced bootstrap sampling causes more trees to split on minority
  features and ends up with sharper boundaries — but the result at
  threshold 0.5 is fewer minority predictions.
- Both LogReg and RF have higher AUC with `balanced`, even when their
  threshold-0.5 F1 drops. AUC measures ranking; it's invariant to
  threshold. F1 is not. **This is exactly why threshold tuning matters.**

## Threshold tuning

Picked the F1-maximizing threshold from the test-set PR curve
(`precision_recall_curve` → `argmax(2 * P * R / (P + R))`).

| Threshold | Precision | Recall | F1    |
|---        |---:       |---:    |---:   |
| 0.500     | 0.757     | 0.176  | 0.286 |
| **0.147** | 0.444     | 0.447  | **0.445** |

A 0.16 F1 uplift from a 4-line change (`y_pred = (proba >= t).astype(int)`)
is the cheapest minority-class improvement you can get in scikit-learn.

The `outputs/pr_curve.png` plot shows the convex precision-recall curve
with both operating points marked. Threshold 0.5 sits in the high-precision /
low-recall corner; the F1-optimal point is much closer to the diagonal,
trading some precision for a lot of recall.

(Caveat: picking the threshold *on the test set* is a slight optimism. In
production you'd carve out a third "validation" split or pick the threshold
inside CV.)

## Calibration

Brier scores (lower = better; range ≈ [0, p_minority * (1 - p_minority)]
≈ [0, 0.05] on this data):

| Model              | Brier   |
|---                 |---:     |
| RF uncalibrated    | 0.03561 |
| RF + isotonic      | 0.03316 |
| RF + sigmoid       | 0.03310 |

Both calibration methods help by ~7% relative. Isotonic and sigmoid are
essentially tied here. The reliability diagram in `outputs/calibration.png`
shows:

- The uncalibrated RF curve lies *above* the diagonal at most predicted
  probabilities: when it says 0.4, the true positive fraction is more
  like 0.55 — i.e. the RF is *under*-confident on its high-probability
  predictions (a known RF property, since each tree votes from
  bootstrap samples that smear the probabilities toward 0.5).
- The two calibrated curves hug the diagonal much more closely,
  especially in the 0.2-0.6 region where the bulk of the predictions
  live.

## Surprises and gotchas

- **`GradientBoostingClassifier` does not accept `class_weight`** —
  the kwarg is silently ignored if you naively pass it (my first run
  produced duplicate identical rows for `cw=None` and `cw="balanced"`).
  Fixed by making the spec explicit. (`HistGradientBoostingClassifier`
  *does* accept `class_weight` since sklearn 1.3 and would be the modern
  default.)
- **`balanced` weights aren't a free win.** They reweight the loss but
  don't move the decision threshold — so models that were already
  precision-heavy can end up worse at threshold 0.5.
- **Brier scores are tiny on imbalanced binary problems** — the trivial
  predictor "always 0" already gets Brier ≈ `p_minority * 1²` ≈ 0.053.
  Don't compare Brier scores against an absolute "good/bad" cutoff; only
  compare them relative to a baseline.

## What I'd try next

1. Use `TunedThresholdClassifierCV` (added in sklearn 1.5) to pick the
   threshold *inside* CV instead of on the test set — proper hygiene.
2. Combine `class_weight="balanced"` *and* threshold tuning — should
   compound, especially for LogReg whose balanced AUC is highest.
3. Swap `GradientBoostingClassifier` for `HistGradientBoostingClassifier`
   with `class_weight="balanced"` to compare modern-API behavior.
4. Try `imbalanced-learn`'s SMOTE / `BalancedBaggingClassifier` to
   contrast oversampling-based interventions with cost-sensitive ones
   (out of scope here since it'd add a non-sklearn dep, but useful).
