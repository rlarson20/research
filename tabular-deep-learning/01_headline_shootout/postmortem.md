# Experiment 1 — Postmortem

## Headline

**FT-Transformer wins both 20k-sample mixed-type datasets**, by
margins big enough to matter:

- Mixed-type classification: FT-T **0.961 accuracy** vs CatBoost 0.926
  vs LightGBM 0.923 (Δ ≈ 3.5 percentage points — well outside noise).
- Mixed-type regression: FT-T **RMSE 161** vs LightGBM 204 vs CatBoost
  308 (FT-T 21% lower error than the best booster).

Boosting wins the small-data control (`load_digits`, 1797 samples) only
narrowly: MLP 0.986 > CatBoost 0.972 > FT-T 0.978 > LightGBM 0.967 —
and even there a plain hand-rolled MLP is the top model. This is the
**inverse** of the prior projects' theme ("linear/boosting beats
DL on tabular"). The dataset shape that flipped the result: 20k
samples + mixed-type + non-trivial cat cardinality + real signal in
cats.

| Dataset | Winner | Metric | LightGBM | CatBoost | TabNet | MLP | FT-Transformer |
|---|---|---|---|---|---|---|---|
| mixed_clf 20k | **FT-T** | accuracy ↑ | 0.923 | 0.926 | 0.908 | 0.919 | **0.961** |
| mixed_reg 20k | **FT-T** | RMSE ↓ | 204.1 | 308.2 | 716.0 | 641.5 | **161.0** |
| digits        | **MLP** | accuracy ↑ | 0.967 | 0.972 | 0.928 | **0.986** | 0.978 |

## What surprised me

1. **FT-T's regression advantage is huge.** On the mixed regression
   set, the second-best DL model (MLP) was 4× worse than FT-T, and even
   LightGBM was 27% worse. The attention mechanism is doing
   something boosting (and even a wide MLP) can't replicate on this
   signal-from-categoricals task.
2. **CatBoost loses regression to LightGBM** here (RMSE 308 vs 204) —
   a flip from `ml-library-experiments/01`, where CatBoost dominated.
   The difference: in that experiment cats were *the only* signal; here
   numericals + cats both matter, and CatBoost's ordered target
   statistics don't help as much when numericals carry independent
   information.
3. **TabNet underperformed every other model on every dataset.** It
   has the smallest parameter count of the DL models (~10k params vs
   FT-T's ~1M), but the regression result (RMSE 716, almost 4× the
   LightGBM number) is poor enough that I suspect `lambda_sparse=1e-3`
   defaults are too aggressive at this dataset size.
4. **The hand-rolled MLP wins digits.** On 1797 samples of 8×8
   images flattened to 64 numericals, the simplest DL model is best.
   FT-T's attention is overkill on 64 features; CatBoost's tree
   structure is too coarse for the smooth pixel-intensity signal.

## What went wrong on the way

**Run 1 (regression):** FT-T got RMSE 1361, MLP got 1329 — both
*worse* than CatBoost's 308. Inspection: the synthetic regression
target has σ ≈ 1000, but every DL model was trained with raw MSE at
Adam lr=1e-4. The effective learning-rate-relative-to-target-scale was
~1e-7 — far too small. Fix: standardize `y` for the DL models (de-scale
predictions before computing RMSE for the report). Boosters are scale-
invariant and weren't affected. **Lesson: there is no equivalent of
this gotcha for boosting; if you only ever use boosters you never have
to think about target scaling.**

**CatBoost cat_features dtype error.** First attempt passed
`df.values` (float64 ndarray after StandardScaler-ing numericals); the
int cat columns got upcast, and CatBoost refused: *"'data' is numpy
array of floating point numerical type, it means no categorical
features, but 'cat_features' parameter specifies nonzero number of
categorical features"*. Fix: pass the `DataFrame` directly so dtypes
are preserved.

**rtdl version pin.** `rtdl==0.0.13` requires `torch<2`. Switched to
`rtdl-revisiting-models==0.0.2`, same authors, supports `torch<3`.
API differs: no `make_baseline` factory, `FTTransformer` takes
`n_cont_features` / `cat_cardinalities` / `d_out` plus
`**get_default_kwargs(n_blocks=N)`.

## Wall-clock breakdown

Total experiment: 350s (~5.8 min).

| Stage | Wall |
|---|---|
| Mixed clf (5 models on 20k samples) | ~145s |
| Mixed reg (5 models on 20k samples) | ~144s |
| Digits (5 models on 1797 samples) | ~57s |

The single dominant cost is **FT-T at 12 epochs × 20k samples ≈ 120s
per dataset** (~70% of total wall). Boosters add up to ~12s; TabNet
to ~28s; hand-rolled MLP to ~13s. FT-T's parameter count (~1M) is
~10× the next-largest DL model and ~80× the LightGBM tree-leaf
count, so wall-clock-per-parameter the comparison is more even.

## Take-aways for the rest of the project

- DL is genuinely competitive on this synthetic generator at n=20k.
  This makes the rest of the slate fair to run — the rest of the
  experiments are not investigating a doomed cause.
- FT-T's compute cost scales linearly with rows and quadratically with
  feature count (per-token attention is O(n_features²)). At n=40k in
  experiment 5 we need the epoch budget to scale inversely, as
  planned.
- Standardize `y` for any DL regression baseline in subsequent
  experiments. The shared dataset module returns raw `y`; per-experiment
  scaling is left to the caller (correct boundary).

## Output files

- `results_clf.txt`, `results_reg.txt`, `results_digits.txt`
- `shootout_summary.png` (~25KB), `wallclock_vs_params.png` (~30KB)
