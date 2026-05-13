# Experiment 5 — Sample-size scaling

## Goal

Grinsztajn et al. 2022 ("Why do tree-based models still outperform
deep learning on tabular data?") argued the answer holds even at
scale. Test that on our synthetic mixed-type clf generator by sweeping
n_train and seeing whether any DL model crosses the CatBoost line.

## Setup

- **Pool:** `make_mixed_clf(n_samples=45_000, seed=42)` shuffled with
  `np.random.default_rng(42).permutation`. First 5000 rows held out as
  a fixed test set used at every n_train. Remaining 40k as the
  training pool.
- **n_train sweep:** {500, 2000, 10_000, 40_000}. Each n uses the
  *cumulative-prefix* slice of the shuffled training pool so larger
  n is a strict superset of smaller n (best-case for learning curves).
- **Models:**
  - **CatBoost** (control): `iterations=400, depth=6, learning_rate=0.05,
    allow_writing_files=False`. Same config as experiments 1 and 4.
  - **FT-Transformer**: experiment 2's depth=2 winner config — `n_blocks=2`,
    default dropouts from `get_default_kwargs(n_blocks=2)`. Epoch
    budget scales inversely with n: 30 epochs @ 500, 20 @ 2k, 12 @ 10k,
    6 @ 40k (keeps total experiment under budget).
  - **TabNet**: same config as experiment 1.

## Why depth=2 FT-T (not depth=4)

Experiment 2 found depth=4 best at 0.961, depth=2 close behind at 0.957.
Depth=4 takes ~5× the wall time. At n=40k that would put one
configuration at 10+ minutes — past the project's budget. Depth=2 is
the right operating point for an n-sweep.

## Outputs

- `crossover_table.txt` — n × model accuracy matrix with FT-CB and
  TabNet-CB deltas.
- `acc_vs_n.png` — test accuracy vs log-n_train.
- `wallclock_vs_n.png` — fit time vs log-n_train (log-y).
