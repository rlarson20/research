# Tabular deep learning experiments

Six self-contained experiments asking a single question: **does
tabular deep learning beat boosting?** Sibling project to
[`../sklearn-experiments/`](../sklearn-experiments/) and
[`../ml-library-experiments/`](../ml-library-experiments/), both of
which concluded *"linear / boosting is hard to beat on tabular."* This
project takes that finding head-on, using six methodological slices.

The short answer: **yes, on this synthetic data, DL beats boosting —
and the gap widens with scale and survives realistic corruption.**
The cross-project "boosting wins tabular" theme inverts here. The
details are dataset-conditional and laid out below.

Each experiment lives in its own subfolder containing:

- `run.py` — the experiment script (`uv run python NN_*/run.py`)
- `summary.md` — *what* APIs are used and *why*
- `postmortem.md` — *what actually happened* when the script ran
- `outputs/` — PNG plots and tabular `.txt` summaries

## Reproducing

```
cd tabular-deep-learning
uv sync
uv run python 01_headline_shootout/run.py     # ~6 min
uv run python 02_rtdl_architectures/run.py    # ~11 min
uv run python 03_numerical_embeddings/run.py  # ~25 s
uv run python 04_categorical_embeddings/run.py# ~75 s
uv run python 05_sample_size_scaling/run.py   # ~4 min
uv run python 06_robustness/run.py            # ~80 s
```

Tested on Python 3.11 with `torch==2.11.0`, `rtdl-revisiting-models==0.0.2`,
`rtdl-num-embeddings==0.0.12`, `pytorch-tabnet==4.1.0`,
`scikit-learn==1.8.0`, `lightgbm==4.6.0`, `catboost==1.2.10`. Every
script uses `random_state=42` and runs on a single CPU.

## Experiments at a glance

| # | Folder | Topic | Dataset | Headline finding |
|---|---|---|---|---|
| 1 | [`01_headline_shootout/`](01_headline_shootout/) | Boosting vs DL across 3 datasets (mixed clf 20k, mixed reg 20k, digits) | synthetic mixed-type + `load_digits` | **FT-Transformer wins both 20k mixed-type sets** (clf 0.961 vs CatBoost 0.926; reg RMSE 161 vs LightGBM 204). Hand-rolled MLP wins `load_digits`. |
| 2 | [`02_rtdl_architectures/`](02_rtdl_architectures/) | MLP × ResNet × FT-T, depth × dropout sweep, full instrumentation | `make_mixed_clf(20_000)` reused | **MLP/ResNet plateau at ~0.91; FT-T spans 0.93–0.96.** Capacity isn't the bottleneck — inductive bias is. depth=4 dropout=0.1 is the FT-T peak (0.961), depth=2 dropout=0.0 is the cost-efficient near-peak (0.957, 5× cheaper). |
| 3 | [`03_numerical_embeddings/`](03_numerical_embeddings/) | Numerical feature embeddings ablation (Gorishniy 2022) | `make_pure_num_clf(10k×20)` + `_reg` | **Gorishniy's headline doesn't reproduce on synthetic data.** Periodic and LinearReLU are *worse* than plain Linear on regression. Only PiecewiseLinear gives a small (~2%) RMSE gain. The paper's effect requires real-world data with multimodal feature distributions. |
| 4 | [`04_categorical_embeddings/`](04_categorical_embeddings/) | DL cat encodings × cardinality vs CatBoost reference | `make_cardinality_variant` at {10, 50, 200} | **One-hot wins at card ≥ 50, CatBoost is no longer the ceiling.** Inverts `ml-library-experiments/01`. Reason: cat levels are iid here (no continuous structure for an embedding to exploit), and numericals carry independent signal that erodes CatBoost's ordered-TS advantage. |
| 5 | [`05_sample_size_scaling/`](05_sample_size_scaling/) | Does DL catch boosting as n grows? (Grinsztajn 2022) | mixed clf, n ∈ {500, 2k, 10k, 40k} | **The crossover happens at n=2000 and widens with n.** FT-T Δ over CatBoost: −0.3 pp → +1.7 → +3.4 → +2.7. TabNet (last on every previous dataset) catches up at n=40k and ties FT-T. |
| 6 | [`06_robustness/`](06_robustness/) | Boosting vs FT-T under corruption (noise / MCAR / shift) | n=40k clean train, corrupted test | **FT-T is the most robust on every corruption type.** Biggest surprise: CatBoost's default `nan_mode="Min"` loses 17 pp at 20% MCAR while FT-T zero-fills with only 7 pp loss. LightGBM's missing-policy is more forgiving than CatBoost's. |

## Cross-experiment observations

### 1. The "boosting wins tabular" cross-project theme inverts here

Both `sklearn-experiments` and `ml-library-experiments` concluded
boosting / linear models are hard to beat. This project found the
**opposite** on the same project conventions:

- **Experiment 1**: FT-Transformer beats CatBoost / LightGBM by 3.5 pp
  on classification and reduces RMSE by 21% on regression.
- **Experiment 4**: One-hot encoding into a plain MLP beats CatBoost
  at every cardinality tested.
- **Experiment 5**: At n=10_000, FT-T leads CatBoost by 3.4 pp; at
  n=40_000, both FT-T and TabNet beat CatBoost by 2.8 pp.
- **Experiment 6**: FT-T degrades less than CatBoost under noise,
  MCAR missingness, and feature shift.

What changed: **the dataset**. The two prior projects used either
sklearn-bundled small datasets (where DL has no chance) or synthetic
generators where cats carried almost all the signal (where CatBoost's
ordered TS dominated). This project's mixed-type synthetic generator
has **both** strong numerical and categorical signal at moderate
scale (20k+ rows), which is exactly the regime where DL — especially
FT-Transformer — is supposed to shine. The cross-project lesson is
**not** "boosting wins"; it's **"the dataset shape decides the
winner, and FT-T wins more dataset shapes than the literature
suggests."**

### 2. Two reputable 2022 results don't reproduce on our synthetic generator

- **Gorishniy et al. 2022** ("On Embeddings for Numerical Features
  in Tabular DL"): claimed Periodic embeddings are a big lever for
  tabular DL. **On our pure-numerical synthetic regression, Periodic
  is 35% *worse* than plain Linear.** It works on real-world data
  with multimodal feature distributions; our `make_regression`
  features are unimodal and standardized.
- **Grinsztajn et al. 2022** ("Why do tree-based models still
  outperform DL on tabular?"): claimed the gap holds even at scale.
  **On our mixed-type generator, DL crosses CatBoost at n=2000 and
  the gap widens with n.** Real-world data quirks may explain the
  paper's finding; the basic n-scaling argument doesn't.

Both findings are dataset-conditional. Synthetic data is more
DL-friendly than the real-world benchmark suites the papers used.
That's a genuine limitation of the project and a real lesson: **a
'tabular DL benchmark' on synthetic data tells you something
different from the same benchmark on real-world data.**

### 3. CatBoost's defaults are not always the right defaults

The `ml-library-experiments` headline was "CatBoost dominates on
high-cardinality cats." Two findings here qualify that:

- **Experiment 4**: When numericals also carry signal, CatBoost is no
  longer the cat-encoding ceiling. Plain one-hot into a DL backbone
  beats it.
- **Experiment 6**: CatBoost's `nan_mode="Min"` (default for binary
  clf) is *worse* than FT-T's zero-fill at every missingness rate.
  LightGBM's `use_missing=true` policy generalizes better.

CatBoost is excellent at one thing (high-cardinality cats with
strong cat-signal); it isn't the universal best choice.

### 4. The instrumentation case for hand-rolled training continues

`ml-library-experiments/02_pytorch_training_loop/` argued the value of
a hand-rolled training loop is the *intermediate quantities* you can
expose — gradient norms, per-epoch LR, per-batch loss — not the final
accuracy. Experiment 2 here extends that to the rtdl architecture
family: per-architecture gradient trajectories *do* tell different
stories (`MLP shrinking grads`, `ResNet stable through depth-4`, `FT-T
flat well-regularized`) that you can't see inside a wrapped fit-method
like TabNet's. The shared training loop in `_shared_torch.py` was
reusable across experiments 1, 2, 3, 4, 5, and 6 — proof that the
hand-rolled investment pays off when the project has more than one
experiment.

### 5. Sandbox-driven choices changed the project

The sandbox blocks OpenML, HuggingFace, and figshare — so no Adult,
no Higgs, no Grinsztajn benchmark, no TabPFN checkpoint. We pivoted
to deterministic synthetic generators in `_shared_datasets.py`. The
upside: **complete control over the question**. We can vary
cardinality, sample size, signal strength, noise, missingness in
isolation. The downside: **none of our findings translate directly to
real-world data**. The findings here are about *methodological
structure* (when does DL beat boosting? when does cat encoding
matter? when do trees handle missing well?) rather than benchmark
numbers.

## Reused infrastructure

- `_shared_datasets.py` — `make_mixed_clf`, `make_mixed_reg`,
  `make_pure_num_clf`, `make_pure_num_reg`,
  `make_cardinality_variant`, `corrupt`. Five generators, all
  deterministic, used across experiments 1/2/3/4/5/6.
- `_shared_torch.py` — `set_seeds`, `make_loader`, `train_one`
  (history with grad_norm), and metric/forward closures. Lifted
  from `ml-library-experiments/02_pytorch_training_loop/run.py`.

## Library surface area covered

- **Boosting (controls)**: `lightgbm.LGBMClassifier/Regressor(deterministic=, force_row_wise=, categorical_feature=)`,
  `catboost.CatBoostClassifier/Regressor(cat_features=, nan_mode=, allow_writing_files=False, train_dir=)`.
- **DL contenders**:
  - `rtdl_revisiting_models.{MLP, ResNet, FTTransformer}` —
    direct-kwargs constructors (no `make_baseline`/`make_default`),
    `FTTransformer.get_default_kwargs(n_blocks=)` and
    `FTTransformer.make_default_optimizer()`.
  - `rtdl_num_embeddings.{LinearEmbeddings, LinearReLUEmbeddings,
    PeriodicEmbeddings, PiecewiseLinearEmbeddings, compute_bins}`.
  - `pytorch_tabnet.tab_model.{TabNetClassifier, TabNetRegressor}` —
    `cat_idxs=`, `cat_dims=`, `cat_emb_dim=`, `device_name="cpu"`,
    explicit `seed=42`.
- **Shared training**: `torch.nn.{Embedding, BatchNorm1d, Dropout,
  Linear}`, `torch.optim.{Adam, AdamW}`, `torch.utils.data.{DataLoader,
  TensorDataset}` with a seeded `torch.Generator`,
  `torch.no_grad()` for eval.
- **sklearn**: `make_classification`, `make_regression`, `load_digits`,
  `StandardScaler`, `train_test_split`, `KFold`, `accuracy_score`,
  `mean_squared_error`.

For per-API explanations see each experiment's `summary.md`; for
what each one *actually did* see the `postmortem.md`.

## Note on the network sandbox

This project was developed in a sandbox that blocks
`openml.org`, `huggingface.co`, `figshare.com`, and
`download.pytorch.org/whl/cpu`. Consequences documented at the top of
`notes.md`. The TLDR: we cannot benchmark against any community
tabular dataset (Adult, Higgs, Covtype, etc.) or use any pretrained
model (TabPFN). All findings are on synthetic data we control.
