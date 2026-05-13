# Experiment 3 — Numerical feature embeddings

## Goal

Gorishniy et al. 2022 ("On Embeddings for Numerical Features in
Tabular Deep Learning") argued the single biggest lever for tabular
DL is *how each scalar numerical feature is mapped to a vector before
the MLP*. Test their claim on pure-numerical synthetic data with one
fixed downstream backbone, varying only the encoder.

## Setup

- **Datasets:** `make_pure_num_clf(n_samples=10_000, n_features=20)`
  and `make_pure_num_reg(n_samples=10_000, n_features=20)`. Both
  `make_classification` / `make_regression`-based, 60% informative
  features, 20% redundant for clf. `StandardScaler` on features; the
  regression `y` is also standardized for the DL training and the RMSE
  is reported in original units.
- **Backbone (fixed):** 2-block `rtdl_revisiting_models.MLP`,
  `d_block=128`, `dropout=0.1`, `d_out=1`. AdamW lr=1e-3,
  weight_decay=1e-5, 20 epochs, batch=512.
- **Encoders (the sweep):**
  1. `LinearEmbeddings(n_features, d_embedding=8)` — baseline per-feature affine.
  2. `LinearReLUEmbeddings(n_features, d_embedding=8)` — same shape + ReLU.
  3. `PeriodicEmbeddings(n_features, d_embedding=8, n_frequencies=24, frequency_init_scale=0.01, activation=True, lite=False)` — sin/cos frequency features (the paper's "PL" recipe).
  4. `PiecewiseLinearEmbeddings(bins=compute_bins(X_tr, n_bins=16), d_embedding=8, activation=True, version="B")` — feature-wise quantile bins, piecewise-linear interpolation across them.

## APIs

- `rtdl_num_embeddings.LinearEmbeddings`, `LinearReLUEmbeddings`, `PeriodicEmbeddings`, `PiecewiseLinearEmbeddings`.
- `rtdl_num_embeddings.compute_bins(X_train, n_bins=16)` — pure quantile bins. Note: passing `y=None` *together with* `regression=` non-None raises `ValueError: If tree_kwargs is None, then y must be None, regression must be None and verbose must be False`. Pass only the input tensor and `n_bins` for quantile binning.

## Outputs

- `encoder_table.txt` — accuracy / RMSE for each encoder × task.
- `clf_curves.png` — val accuracy vs epoch for the 4 encoders on classification.
- `reg_curves.png` — val RMSE (standardized scale) vs epoch on regression.
