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
