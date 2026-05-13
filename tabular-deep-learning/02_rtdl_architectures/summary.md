# Experiment 2 — rtdl architectures

## Goal

The "Revisiting Deep Learning Models for Tabular Data" paper
(Gorishniy et al. 2021) introduced three workhorse architectures —
MLP, ResNet, FT-Transformer — and argued you should always benchmark
against all three rather than just an MLP. Reproduce that triad on the
same mixed-type 20k-sample dataset used in experiment 1 and run a
depth × dropout sweep with full per-epoch instrumentation. Same
training loop as `ml-library-experiments/02_pytorch_training_loop/` so
the gradient-norm / loss-curve story carries across the project.

## Setup

- **Dataset:** `make_mixed_clf(n_samples=20_000, seed=42)` — byte-identical to experiment 1's mixed-clf split.
- **Sweep:** `arch ∈ {MLP, ResNet, FT-Transformer} × depth ∈ {2, 4} × dropout ∈ {0.0, 0.1, 0.3}` = 18 configs.
- **Embeddings for MLP/ResNet:** the rtdl `MLP` and `ResNet` modules accept a single tensor of size `d_in`. Cat columns get a learned `nn.Embedding(card, d_emb=8)` each; the concatenation of standardized numericals + cat embeddings is fed in as `d_in`. FT-Transformer accepts `(x_cont, x_cat)` natively.
- **Training:** AdamW, weight_decay=1e-5, batch=1024, MLP/ResNet at lr=1e-3 for 10 epochs, FT-T at lr=1e-4 for 8 epochs (its `make_default_optimizer` recipe). Loss: BCE-with-logits.
- **Determinism:** `set_seeds(42)` and a seeded `DataLoader` generator per config — runs are reproducible within library guarantees.

## APIs

- `rtdl_revisiting_models.MLP(d_in, d_out, n_blocks, d_block, dropout)`
- `rtdl_revisiting_models.ResNet(d_in, d_out, n_blocks, d_block, d_hidden_multiplier, dropout1, dropout2)`
- `rtdl_revisiting_models.FTTransformer(n_cont_features, cat_cardinalities, d_out, **get_default_kwargs(n_blocks=N))` — the `attention_dropout` / `ffn_dropout` / `residual_dropout` knobs are overridden to the sweep value (residual at half rate so we don't double-dip).

## Instrumentation

`_shared_torch.train_one(..., log_grad=True)` returns a per-epoch
history dict with `train_loss`, `val_loss`, `val_metric`, `grad_norm`,
`lr`. Reuses the recipe from `ml-library-experiments/02`. Gradient
L2 is computed as `sqrt(sum_p ||grad_p||²)` averaged over batches
within each epoch.

## Outputs

- `arch_table.txt` — 18 rows of `(arch, depth, dropout, val_acc, fit_s, params)`.
- `training_curves.png` — 3 panels (one per arch family), each with depth×dropout overlay on val accuracy vs epoch.
- `grad_norms.png` — same layout, log y, gradient L2 trajectories.
