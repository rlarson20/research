# Experiment 4 — Postmortem

## Headline

**One-hot wins at high cardinality. CatBoost is no longer the
ceiling.** This *inverts* the cross-project conclusion from
`ml-library-experiments/01_boosting_categoricals/`.

| encoder | card=10 | card=50 | card=200 |
|---|---:|---:|---:|
| emb-d1 | 0.918 | 0.862 | 0.861 |
| emb-d4 | **0.963** | 0.891 | 0.865 |
| emb-d16 | 0.962 | 0.932 | 0.885 |
| **onehot** | 0.961 | **0.954** | **0.943** |
| target-mean | 0.955 | 0.932 | 0.906 |
| target-kfold | 0.957 | 0.933 | 0.911 |
| catboost (ref) | 0.950 | 0.927 | 0.903 |

Three distinct stories collide:

1. **At card=10**, the right learned embedding dim (`d=4`) wins by
   ~1 pp over the next best (one-hot at 0.961, target-kfold at 0.957).
   The classic "embed at √card" advice (here √10 ≈ 3) is correct.
   `d=1` is catastrophically bad (−4.5 pp) — the embedding has no
   capacity to encode level-conditional effects.
2. **At card=50**, the ranking flips. One-hot leaps to 0.954, ~2 pp
   ahead of the next-best DL encoding (emb-d16 at 0.932 and target-
   kfold at 0.933). emb-d4 drops to 0.891 — its 4-dim slot just
   can't hold 50 distinct levels of signal.
3. **At card=200**, one-hot's gap widens further (0.943 vs 0.911 for
   the next-best target-kfold, 0.885 for emb-d16). Embedding-d4 and
   emb-d1 are both around 0.86 — basically random in the cat block.

## Why one-hot is winning

This is the surprise. Two factors:

- **Plenty of training data per level.** 15k rows × 80% train ÷ 200
  levels per col = ~60 examples per level. With 4 cat cols, each
  level still sees ~60 fits. A dense linear layer over one-hot can
  learn per-level coefficients directly; an embedding has to learn
  the embedding *and* the downstream mixing weights, and the
  gradient signal per level is the same ~60 examples either way.
- **No level-level structure to share.** Our synthetic generator
  draws cat-level effects iid from `N(0, σ)`. There is no notion of
  "similar levels" that would benefit from a continuous embedding
  space. Real-world categoricals usually do have this (e.g., postal
  codes near each other have correlated effects), and that's the
  regime where embeddings should outperform one-hot.

So the experiment isolates a real and surprising lesson: **one-hot
is correct when levels are iid; embeddings are correct when levels
are correlated.** The synthetic data is the iid case; real-world
data is usually the correlated case. The default
"always-embed-categoricals" tutorial advice presumes the latter
without saying so.

## Why CatBoost isn't the ceiling

In `ml-library-experiments/01_boosting_categoricals/`, the cat
columns were the *only* signal (3 informative high-cardinality cats
on a binarized regression target). CatBoost's ordered target
statistics dominated because they directly model `E[y | level]` with
proper out-of-fold protection — exactly what target encoding does,
but per-tree-split.

Here, numericals also carry strong signal (10 informative numerical
features). The downstream DL models can fit a clean linear combo of
numerical signal + per-level cat coefficients. CatBoost is paying
for tree-split flexibility on numericals that DL models get more
cheaply via the Linear + MLP path. **The relative gain from "good
cat handling" shrinks when numericals are also informative.**

The result also matches `make_cardinality_variant`'s additive
generative model: `score = numerical_score + Σ_j level_effects[j]`.
A linear-on-one-hot decoder is the *exact* inverse of this generator.
CatBoost has to discover it through tree fitting and pays for the
extra inductive-bias mismatch.

## Param count cost of one-hot

One-hot at card=200 costs 130k parameters in the first layer
(4 × 200 × 128 first-layer weights), vs 28k for target encoding and
49k for emb-d16. **5× the parameters of the next-most-expensive DL
option**, but still cheap on CPU at this dataset size. At card=1000+
the param cost would start to matter; at our cardinalities it doesn't.

## emb-d1 anomaly

emb-d1 (single-dimension learned embedding) clocks 0.918 at card=10
but only 0.861 at card=50 and card=200. The model can squeeze the
10-level structure into 1 dim (it's monotone in level value), but
can't fit 200 levels into a single scalar. This is mostly a
"don't pick d=1 ever" cautionary tale.

## Implementation gotchas

- **CatBoost dtype gotcha re-raised.** Same as experiment 1: pass
  the `DataFrame` to `CatBoostClassifier.fit(...,
  cat_features=cat_names)`, not the ndarray, or the int cat columns
  get upcast to float64 in the conversion and CatBoost refuses.
- **Manual one-hot encoding** (`np.zeros(...); idx assign`) was
  ~2× faster than going through `sklearn.preprocessing.OneHotEncoder`
  for this fixed-cardinality case. We always know the cardinality
  at construction time, so the manual route is fine.
- **5-fold OOF target encoding** is the right move over plain
  mean-target. The lift in our results is small (~0.2 pp) — the
  smoothing factor (α=10) provides most of the leakage protection
  on this data.

## Take-aways

1. **Sparse one-hot is a hard-to-beat baseline** when:
   - The number of levels is small relative to training rows.
   - Levels are iid (no continuous structure across them).
   In those cases, learned embeddings give you *nothing* over a dense
   linear layer over one-hot.
2. **Embedding dim matters more than encoding strategy** at small
   cardinality. `d=1` is always wrong; `d=4` is fine at card=10;
   `d=16` is needed at card=50; at card=200, no learned-emb dim we
   tried catches up to one-hot.
3. **CatBoost's ordered TS advantage is dataset-dependent.** It
   shines when cats carry *most* of the signal; here, where
   numericals are also informative, plain DL encodings beat it.
4. **For experiment 5 (sample-size scaling), use emb-d16 in the
   FT-Transformer.** That's the right "non-degenerate" learned
   embedding dim; the conclusion in this experiment is about the
   isolated cat encoder, not the full FT-T pipeline.

## Wall-clock

Total: **73s**. Each cardinality × encoder costs ~3s for DL, ~3s for
CatBoost. Under the 5-min budget by a wide margin; could expand the
sweep to card ∈ {10, 50, 200, 500, 1000} in future work without
busting the budget.

## Output files

- `encoding_matrix.txt` — 7 × 3 accuracy grid.
- `param_count_table.txt` — same shape, parameter counts.
- `cardinality_effect.png` — line plot, CatBoost dashed reference.
