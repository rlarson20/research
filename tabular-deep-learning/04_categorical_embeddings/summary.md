# Experiment 4 — Categorical encoding × cardinality

## Goal

Direct callback to `ml-library-experiments/01_boosting_categoricals/`,
which found CatBoost's *ordered target statistics* dominate
LightGBM (integer-coded) and XGBoost (one-hot) on high-cardinality
categoricals (log-loss 0.50 vs 0.80+). That comparison was between
*boosters*. The follow-up question, taken head-on here: among the
**DL-side** categorical encodings, which one actually matches
CatBoost — and how does the ranking shift with cardinality?

## Setup

- **Datasets:** three uniform-cardinality variants from
  `_shared_datasets.make_cardinality_variant(level)`. Each is 15k
  rows × (10 numerical + 4 categorical) with cardinality fixed across
  all 4 cat cols at `level ∈ {10, 50, 200}`.
- **DL backbone (shared across all DL encodings):** 2-layer MLP with
  hidden=128, BN + ReLU + dropout 0.1. Numericals go through a
  `rtdl_num_embeddings.LinearEmbeddings(d_emb=8)`. The cat encoding
  is the only thing that varies. AdamW lr=1e-3, weight_decay=1e-5,
  15 epochs, batch=1024.
- **Encodings (7 total):**
  - `emb-d1`, `emb-d4`, `emb-d16` — per-column `nn.Embedding(card, d)`.
  - `onehot` — manual `np.zeros + assign`; flattens to `(n_cat × card)`.
  - `target-mean` — train-set `E[y | level]` with additive smoothing (α=10) toward the global mean.
  - `target-kfold` — 5-fold OOF mean target encoding (train rows get out-of-fold means; test rows get the full-train means).
  - `catboost` — CatBoost native (`cat_features=`) as the reference "ceiling".

## APIs

- `nn.Embedding(num_embeddings=card, embedding_dim=d)`.
- `sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)` for OOF folds.
- `catboost.CatBoostClassifier(iterations=400, depth=6, cat_features=cat_names, allow_writing_files=False, train_dir=outputs/_catboost_info)`.

## Outputs

- `encoding_matrix.txt` — 7 encoders × 3 cardinalities of test accuracy.
- `param_count_table.txt` — same shape, parameter counts (so the cost of
  one-hot at high cardinality is explicit).
- `cardinality_effect.png` — line plot of accuracy vs cardinality, one
  line per encoder, CatBoost dashed for the reference.
