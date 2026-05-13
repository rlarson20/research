# Experiment 1 — Headline shootout

## Goal

Establish a baseline answer to the project's central question: *on
representative tabular data, does deep learning actually beat
boosting?* Run five contenders head-to-head on three datasets that
exercise different regimes:

- **mixed-type binary classification, 20k samples** — the canonical
  modern-tabular setting (cat + numerical, real-world cardinality
  range).
- **mixed-type regression, 20k samples** — same shape, continuous
  target; tests whether boosting's leaf-wise residual fitting beats
  DL's gradient descent on non-binary targets.
- **`load_digits` (1797 × 64)** — the small-data, pure-numerical
  control. The same dataset used in `ml-library-experiments/02_pytorch_training_loop/`.

## Models

| Family | Model | API |
|---|---|---|
| Boosting | LightGBM | `lightgbm.LGBMClassifier/Regressor(n_estimators=400, deterministic=True, force_row_wise=True, n_jobs=4)` with `categorical_feature=cat_names` |
| Boosting | CatBoost | `catboost.CatBoostClassifier/Regressor(iterations=400, depth=6, allow_writing_files=False, train_dir=outputs/_catboost_info)` with `cat_features=cat_names` |
| DL | TabNet | `pytorch_tabnet.tab_model.TabNetClassifier/Regressor(n_d=8, n_a=8, n_steps=3, cat_idxs=, cat_dims=, cat_emb_dim=4, seed=42, device_name="cpu")`, `fit(max_epochs=30, patience=8, batch_size=2048, virtual_batch_size=256)` |
| DL | Hand-rolled MLP | 4-layer, hidden=128, BN + ReLU + dropout 0.1, learned `nn.Embedding(card, min(16, card//2))` per cat. Adam lr=1e-3, weight_decay=1e-5, 15 epochs, batch=1024. |
| DL | FT-Transformer | `rtdl_revisiting_models.FTTransformer(n_cont_features=, cat_cardinalities=, d_out=1, **get_default_kwargs(n_blocks=3))`. Optimizer from `FTTransformer.make_default_optimizer(model)`, 12 epochs, batch=512. |

## Implementation notes

- **`rtdl_revisiting_models` (not `rtdl`).** The legacy `rtdl==0.0.13`
  package pins `torch<2`; the modern replacement by the same authors
  drops that pin. API differs: there are no `make_baseline` /
  `make_default` factories — `MLP/ResNet/FTTransformer` are
  constructed by direct kwargs, and `FTTransformer` takes
  `n_cont_features`, `cat_cardinalities`, `d_out`, plus
  `**get_default_kwargs(n_blocks=N)`.
- **CatBoost dtype gotcha.** Passing a float `ndarray` with
  `cat_features=` raises `'data' is numpy array of floating point …
  but cat_features parameter specifies nonzero number of categorical
  features`. Fix: pass the `DataFrame` directly so int dtypes on cat
  columns survive.
- **Regression target standardization.** The synthetic regression
  target has std ≈ 1000. Boosters tolerate this; DL models trained
  with raw MSE have effective learning rates wildly mismatched to that
  scale. The fix is the conventional one: standardize `y` (zero-mean,
  unit-stddev) before training DL and de-scale predictions before
  computing the RMSE so the metric remains comparable to the booster
  baseline. This is the right "fair fight" baseline; the un-scaled
  run was already a lesson on its own (see postmortem).
- **TabNet's `device_name="cpu"`** is necessary or the model probes
  for CUDA on import. `seed=42` is set on the constructor; the
  package doesn't fully honor `torch.manual_seed`, so we document
  "deterministic within TabNet's guarantees" rather than promise
  bit-exact runs (same convention as `ml-library-experiments`).
- **Parameter counts.** For boosters we use a rough proxy
  (`num_trees × leaves_per_tree`); for DL models we sum
  `p.numel()` over `model.parameters()`. Not directly comparable
  units, but useful as a "model capacity" ordering.

## Outputs

- `results_clf.txt`, `results_reg.txt`, `results_digits.txt` — per-dataset table of (model, metric, fit_s, params).
- `shootout_summary.png` — grouped bar chart of normalized "winner score" by model × dataset.
- `wallclock_vs_params.png` — log-log scatter of fit time vs model parameter count, colored by model, marker shape by dataset.
