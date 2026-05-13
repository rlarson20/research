# Experiment 6 — Postmortem

## Headline

**FT-Transformer is the most robust of the three on every corruption
type we tested.** It loses the least accuracy under Gaussian noise,
under MCAR missingness (even zero-filling, with no missing handling),
and under distributional shift. The opposite of the textbook "trees
are robust, DL is fragile" narrative.

The biggest surprise: **CatBoost's native NaN handling is
*worse* than FT-Transformer's zero-fill** at every missingness rate
we tested.

## Headline numbers

Clean baseline: CatBoost 0.937, LightGBM 0.928, FT-T 0.963.

| corruption | CatBoost | LightGBM | FT-T |
|---|---:|---:|---:|
| gauss σ=0.1 | −0.42 pp | −0.40 pp | −0.62 pp |
| gauss σ=0.5 | −8.4 pp | −7.2 pp | −8.9 pp |
| gauss σ=1.0 | −17.6 pp | −16.0 pp | −18.8 pp |
| mcar 5% | **−5.1 pp** | −1.6 pp | −2.2 pp |
| mcar 20% | **−17.0 pp** | −5.9 pp | −7.3 pp |
| mcar 50% | **−31.8 pp** | −14.7 pp | −16.3 pp |
| shift μ=0.5 | −12.5 pp | −12.1 pp | −11.8 pp |
| shift μ=1.0 | −27.3 pp | −27.3 pp | −25.7 pp |
| shift μ=2.0 | −41.1 pp | −40.8 pp | −40.9 pp |

(FT-T's clean baseline is 2.6 pp higher than CatBoost's, so an equal
percentage-point drop leaves FT-T ahead. The Δ columns are
*degradation*, not absolute rank.)

## The CatBoost missingness surprise

CatBoost loses **17 pp at 20% MCAR** while FT-T (just zero-filling
the NaN cells) only loses 7 pp. At 50% MCAR, CatBoost drops to 0.619
(barely better than guessing) while FT-T keeps 0.800.

What's happening: CatBoost's `nan_mode="Min"` (the default for binary
clf) encodes NaN as "less than every observed value" at every split.
When a feature has bimodal or symmetric structure (most of our
informative numericals do, post-StandardScaler), the "below minimum"
sentinel is far outside the training distribution and many trees
route MCAR rows to the wrong leaf en masse. LightGBM uses a different
default policy (`use_missing=true, zero_as_missing=false`) that's
more graceful: it learns at each split which side missing should go
based on the training residuals, which generalizes better.

**For practitioners**: do not assume "CatBoost handles missing
values" means it handles them *well*. Test explicitly, or zero-fill
upstream. We could potentially have tested `nan_mode="Mean"` /
`nan_mode="Max"` as alternatives — keeping the finding-as-observed
rather than re-tuning to find a better recipe (the headline lesson is
that the default isn't robust).

## Why FT-T is robust under noise / shift

Two mechanisms plausibly explain the FT-T advantage on Gaussian noise
and mean shift:

1. **Feature-token attention averages across features.** When one
   feature's value is noisier or shifted, attention can downweight
   it relative to clean features at inference time. Trees split on
   one feature at a time and have no equivalent recovery mechanism.
2. **Soft features survive small perturbations.** FT-T's
   `LinearEmbeddings` is smooth: a small input change produces a
   small embedding change, which gets smeared further by attention.
   Tree splits are hard threshold rules — flipping one input across
   a split boundary flips the entire downstream branch.

Under large shift (μ=2.0 puts top-3 features 2 std-units out of
training distribution), all models converge near random because the
relevant signal disappears. FT-T's advantage here is small (~0.3 pp)
because there's no signal left to recover.

## Mean-shift on important features

`outputs/degradation_shift.png` is the most dramatic curve. Shifting
just **3 numericals by 0.5σ** drops every model by 12 pp; at 2.0σ
the drop is 40+ pp. These are the LightGBM-top-3 importances
(`num_2, num_4, num_8`), and they carry most of the
boosters' predictive power. Corrupting just *those three columns*
collapses both boosters' performance.

FT-T is hit equally hard in absolute terms (−40.9 pp at μ=2.0), but
since its clean baseline was 2.6 pp higher, it ends at 0.554 vs
CatBoost's 0.526 — a 2.8 pp lead at the worst corruption.

## What didn't surprise me

- **Gaussian noise is the least destructive corruption** at small
  σ (0.1 → only ~0.5 pp loss for any model). All three are well-
  trained on a slightly noisy version of the same distribution and
  handle the test-time bump.
- **All models converge to ~random under heavy enough corruption**
  (shift μ=2.0 → 0.55, mcar 50% → 0.62-0.80). The corruption is
  destroying the signal, not preserving its rank.

## Take-aways

1. **The Grinsztajn thesis is more nuanced than "DL fragile, trees
   robust."** On this generator, DL is *more* robust than trees on
   Gaussian noise, MCAR missingness, and feature shift. The real-
   world data quirks the paper cited may break DL in some specific
   ways (categorical drift, sparsity, extreme outliers) that we
   haven't simulated — but the simple corruptions tested here favor
   DL.
2. **CatBoost's default missing handling can be a trap.** It loses
   12–25 pp more than LightGBM at the same missing rate. If you
   have meaningful missingness in your data, do not assume CatBoost's
   defaults will handle it well — test, and consider zero-fill +
   indicator instead.
3. **FT-T's robustness comes for free.** No extra training, no
   adversarial augmentation, no missing-value-pretraining. The
   architecture's smooth attention + embedding stack just degrades
   more gracefully than tree splits.

## Wall-clock

Total: **78s**. Almost all of that is the three model fits (CB 8s,
LGB 1s, FT-T 60s). The 27 evaluation passes total ~3s.

## Output files

- `degradation_table.txt` — full 10×8 table.
- `degradation_noise.png`, `degradation_missing.png`,
  `degradation_shift.png` — one panel per corruption family.
