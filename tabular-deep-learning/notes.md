# Working notes

Diary appended as work proceeds. Each experiment has its own `summary.md`
(APIs + why) and `postmortem.md` (what actually happened) under
`NN_*/`. This file tracks the cross-experiment journey.

## Setup

- Forked the conventions of `sklearn-experiments/` and `ml-library-experiments/`
  exactly: top-level `pyproject.toml` + `uv.lock`, per-experiment
  `run.py` / `summary.md` / `postmortem.md` / `outputs/`.
- `uv sync` pulls ~3GB; CUDA libs come along with PyPI torch even on a
  CPU-only machine, same as last project.

### Network probe

Verified before planning:

- **OpenML** (`openml.org`, `api.openml.org`): 403
- **HuggingFace** (`huggingface.co`, `cdn-lfs.huggingface.co`): 403
- **figshare** (`ndownloader.figshare.com`, used by sklearn's `fetch_*`): 403
- **PyTorch CPU wheel index** (`download.pytorch.org/whl/cpu`): 403
- **PyPI**: 200 — all target libraries available

Net effect: this project uses **only** `sklearn`-bundled datasets
(`load_breast_cancer`, `load_diabetes`, `load_digits`) and synthetic
generators (`make_classification`, `make_regression`). No Adult, no
California Housing, no Grinsztajn benchmark suite, no TabPFN
checkpoint. The synthetic generators in `_shared_datasets.py` are
deterministic and parameterized so the question (mixed-type,
cardinality, sample size, noise) can be isolated.

### Library compat

- **`rtdl==0.0.13` pins `torch>=1.7,<2`** — incompatible with modern
  torch. Switched to **`rtdl-revisiting-models==0.0.2`** (`torch<3`) by
  the same yandex-research authors. The newer package's API is slightly
  different (no `make_baseline`/`make_default` factories — direct
  kwargs), and `FTTransformer` requires `d_out` as a positional arg
  alongside `backbone_kwargs` from `get_default_kwargs(n_blocks=N)`.
- `rtdl-num-embeddings==0.0.12` declares `torch<3,>=1.12` — fine.
- `pytorch-tabnet==4.1.0` installs cleanly. No `__version__` attribute on
  the top-level module — minor.

### Final dep set

```toml
torch>=2.5            # 2.11.0 actually installed
scikit-learn>=1.8.0   # 1.8.0
lightgbm>=4.5         # boosting control
catboost>=1.2         # boosting control + cat-native reference
pytorch-tabnet==4.1.0
rtdl-revisiting-models>=0.0.2
rtdl-num-embeddings>=0.0.12
```

No XGBoost (LightGBM + CatBoost suffice as boosting controls in this
project; ml-library-experiments-01 already covered XGBoost vs LightGBM
vs CatBoost head-to-head on categoricals). No Optuna/SHAP (out of
scope). No pytorch-tabular/pytorch-frame/tabpfn (heavy wrappers / model
download blocked).

## Experiment 1 — Headline shootout

Took 5.8 min (slightly over the 5-min-per-experiment soft cap, but
acceptable given 3 datasets × 5 models). The big surprise: **FT-T won
both 20k-sample mixed-type datasets** by margins outside noise (mixed
clf: 0.961 vs CatBoost 0.926; mixed reg: RMSE 161 vs LightGBM 204).
This *inverts* the cross-project theme from `sklearn-experiments` and
`ml-library-experiments` ("linear/boosting wins tabular"). The
combination that flipped the result: 20k samples + mixed-type +
informative categoricals + non-trivial cat cardinality.

CatBoost reg lost to LightGBM (308 vs 204) — flip from
ml-library-experiments-01. The difference: there cats were the *only*
signal, here numericals and cats both matter, and CatBoost's ordered
TS doesn't help when numericals already carry independent info.

Two methodology bugs caught:
1. `df.values` upcasts int cats to float when other cols are float;
   CatBoost rejects this with `cat_features`. Fix: pass the DataFrame.
2. Regression target std ≈ 1000 broke DL training (effective
   lr-to-target scale ~1e-7). Boosters are scale-invariant and were
   unaffected. Standardize `y` for DL, de-scale the RMSE for reporting.
   **Lesson: there's no equivalent of this gotcha for boosting; you
   only ever discover it because you ran a DL baseline.**

TabNet was last on every dataset. With its tiny parameter count
(~10k) and default `lambda_sparse=1e-3`, it seems undersized for
mixed-type 20k. Wrote up the postmortem; will revisit defaults if it
keeps trailing in later experiments.

## Experiment 2 — rtdl architectures

11 min total — well over the 5-min cap, FT-T at depth=4 dominates the
cost (3–4 min per dropout setting). depth=4, dropout=0.1 wins at
0.961, but depth=2 dropout=0.0 hits 0.957 at 5× lower cost — that's
the "FT-T baseline" I'll reuse in experiment 5 to keep the scaling
sweep within budget.

MLP/ResNet plateau at 0.91 ± 0.005 — depth doesn't help MLP at all,
helps ResNet by 0.3 pp. **The architectural ceiling on this dataset
is ~0.91 for non-attention DL**; FT-T's ~5 pp jump comes from
feature-token attention, not capacity (MLP at depth=4 dropout=0.0
has 28k params vs FT-T-d2 at 335k — but FT-T-d2 still beats MLP-d4 by
~4.5 pp).

Gradient trajectories tell the now-familiar story: ResNet keeps
gradients high through depth-4 thanks to residual connections,
MLP-d4's gradients shrink, FT-T's gradients are well-controlled by the
AdamW + default-optimizer recipe.

## Experiment 3 — Numerical feature embeddings

24s, well under budget. **Surprising result: Gorishniy 2022's headline
claim doesn't reproduce on synthetic data.** Periodic and LinearReLU
are *worse* than plain Linear on regression (RMSE 15.39, 14.86 vs
11.38). Only PiecewiseLinear (PLE) gives a measurable gain (RMSE
11.13, −2.2%). On classification all four cluster within 0.6 pp.

The paper's "use Periodic for everything" advice presumes real-world
data with multimodal / threshold-shaped feature distributions. Our
`make_classification` / `make_regression` synthetic data has mostly-
monotone Gaussian-mixture structure where the linear basis is already
correct. **The encoder is dataset-dependent; PLE is the safest
default if you must pick something fancier than Linear.**

Two compute_bins gotchas: must pass `y=None` and `regression=None`
together (the docstring isn't clear; mixing them with `tree_kwargs=None`
errors). `PeriodicEmbeddings(lite=False)` is now mandatory in 0.0.12 —
no default.

## Experiment 4 — Categorical embeddings × cardinality

73s. The biggest surprise of the project so far: **one-hot wins at
cardinality ≥ 50, and CatBoost is no longer the ceiling.** This
inverts the headline from `ml-library-experiments/01_boosting_categoricals/`.

At card=10, `nn.Embedding(card, d=4)` wins (0.963). At card=50 and
card=200, one-hot dominates (0.954, 0.943). Learned embeddings of
dim 4 or 16 lose ~6-8 pp to one-hot at high cardinality. CatBoost
trails the best DL encoding by 1–3 pp at every cardinality.

The reconciliation with ml-library-experiments-01: there, cats were
*the only* signal. Here, numericals also carry strong signal, so the
relative gain from sophisticated cat handling shrinks. Plus: our cat
levels are iid `N(0, σ)` effects — there's *no* level-level structure
for an embedding to exploit. Real-world cats have such structure
(postal codes, product hierarchies); synthetic ones don't. **One-hot
is correct when levels are iid; embeddings are correct when levels
are correlated.** Tutorial advice "always embed cats" presumes
correlated levels without saying so.

For exp 5: use emb-d16 in FT-T. d=1 is always wrong; d=4 saturates by
card=50. Conclusion about the isolated cat encoder, not the FT-T
pipeline.

## Experiment 5 — Sample-size scaling

212s, the cleanest result of the project: **DL wins at scale and the
gap widens.** Crossover at n=2000, then FT-T pulls steadily away from
CatBoost (Δ +1.7 → +3.4 → +2.7 pp). TabNet — last on every previous
dataset — catches up at n=40k and ties FT-T. **Grinsztajn et al.
2022's headline doesn't hold on this synthetic generator.**

The cost story: CatBoost at n=40k is 7× faster than FT-T (8.8s vs 65s)
and 2× faster than TabNet (31s), at ~3 pp lower accuracy. If you're
cost-bound, CatBoost is still the right pick. If you have the budget,
DL wins.

The Grinsztajn result is dataset-conditional, not universal. Our
generator gives DL ideal conditions: standardized features, dense
signal, no outliers/missingness. Experiment 6 will probe what happens
when those conditions break.

## Experiment 6 — Robustness probe

78s. **FT-T is the most robust on every corruption family** — even
zero-filling NaN beats CatBoost's native NaN handling. The biggest
surprise: CatBoost loses 17 pp at 20% MCAR (FT-T loses 7, LightGBM
loses 6). CatBoost's `nan_mode="Min"` default routes missing rows
to "below-minimum" sentinel branches; with post-StandardScaler
bimodal features, those branches go badly wrong. LightGBM's missing
policy is more forgiving (`use_missing=true, zero_as_missing=false`).

Mean shift on the top-3 LightGBM-importance features is the most
destructive single corruption: 0.5σ shift drops every model by 12 pp;
2σ shift sends everyone near random.

**The Grinsztajn "DL fragile, trees robust" narrative doesn't hold on
this generator.** On every corruption type, FT-T degrades least.
Real-world quirks that break DL may be more specific than what we
simulated here — categorical drift, sparse high-cardinality cats,
extreme outliers — but the simple corruptions tested favor DL.

## Cross-cutting

- **Total wall**: ~23 minutes for all six experiments. Within the
  6–8 × 2–5 min plan budget (just over the soft cap on experiments 1
  and 2; well under for 3, 4, 5, 6). FT-Transformer's training loop
  dominates wall in experiments 1, 2, 5, 6.
- **Determinism**: `random_state=42` / `torch.manual_seed(42)` / seeded
  `DataLoader` generator throughout. Verified via the smoke test in
  `_shared_datasets.py` (same args → same output). TabNet has its own
  `seed=42`; we accept "deterministic within tabnet's guarantees"
  per the project's prior conventions.
- **Library-determinism flags** reused from `ml-library-experiments`:
  LightGBM `deterministic=True, force_row_wise=True`; CatBoost
  defaults; PyTorch `torch.manual_seed(42)` + seeded loader generator
  (no `use_deterministic_algorithms(True)` — crashes on some ops).
- **`OMP/MKL/OPENBLAS_NUM_THREADS=4`** set at the top of every `run.py`
  and `torch.set_num_threads(4)` to prevent oversubscription.
- **`matplotlib.use("Agg")`** and headless image writes throughout —
  same convention as the two sibling projects.
- **CatBoost output dir**: every experiment writes to
  `outputs/_catboost_info/`, which is `.gitignore`-d. The
  `allow_writing_files=False` flag suppresses most of the chatter;
  the train_dir override catches the rest.

### Big-picture take-away

The two sibling projects concluded "linear / boosting wins tabular."
This one concluded the opposite — **on the synthetic generator we
were forced to use because the sandbox blocked every real dataset
host**. That contrast is the most useful lesson of the trio:
*which* tabular data you're testing on completely determines whether
DL or boosting wins. The literature splits on this for exactly the
same reason.
