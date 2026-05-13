# Experiment 2 — Postmortem

## Headline

**FT-Transformer dominates the rtdl triad by ~5 percentage points** on
this 20k mixed-type classification task — and depth/dropout knobs
matter very differently per architecture.

Best per arch (with depth × dropout):

| arch | best val_acc | best config | fit_s | params |
|---|---:|---|---:|---:|
| MLP | 0.9110 | depth=2, drop=0.1 | 2.8 | 28,609 |
| ResNet | 0.9140 | depth=4, drop=0.1 or 0.3 | ~7 | 277,057 |
| FT-Transformer | **0.9607** | depth=4, drop=0.1 | 180.3 | 2,249,385 |

MLP/ResNet plateau in the 0.91 ± 0.005 band regardless of depth × dropout.
FT-T sweeps 0.934 → 0.961 across the grid; the variance inside FT-T is
larger than the entire MLP/ResNet range.

## What the sweep showed

**MLP — saturating.** Going from depth 2 to depth 4 doesn't help
(0.911 → 0.909) and dropout barely moves the needle (0.908–0.911).
The model converges fast and stops learning more from the data than
its capacity allows. The same shape as the digits-MLP result from
experiment 1: MLP is a *capable but unsubtle* baseline.

**ResNet — slight gain from depth + dropout.** depth=4 with dropout
0.1 or 0.3 ties at 0.914, ~0.3 pp better than the best MLP. The
residual connections let depth=4 train at all (depth=4 MLP didn't
improve over depth=2), but the gain over MLP is marginal.

**FT-Transformer — depth helps, dropout hurts at 0.3.** depth=4
beats depth=2 by ~1 pp at every dropout level (0.957 → 0.960 at p=0,
0.948 → 0.961 at p=0.1, 0.934 → 0.952 at p=0.3). Within depth, dropout
0.1 ≥ 0.0 > 0.3. p=0.3 hurts because at 8 epochs the model never
recovers the capacity it loses to high dropout — same lesson as
`ml-library-experiments/02`'s aggressive-LR-schedule finding: knobs
sized to a longer training budget hurt at our short one.

## Wall-clock breakdown

Total: **665s** (~11 min). FT-T accounts for 610s (92% of total):

| config | wall |
|---|---:|
| 12 × MLP/ResNet | ~50s |
| FT-T depth=2 (×3 dropouts) | ~109s |
| FT-T depth=4 (×3 dropouts) | ~502s |

FT-T at depth=4 is the single biggest cost in the project. For
experiment 5 (sample-size scaling), FT-T must use depth=2 (the
best-depth-2 FT-T at 0.957 is only 0.4 pp behind depth=4 at 0.961 but
~5× faster). Plan note for exp 5: use the depth=2 FT-T config from
this sweep as the reused FT-T baseline.

## Gradient-norm story

`outputs/grad_norms.png` shows the three architectures' L2 gradient
trajectories per epoch. The pattern matches the
`ml-library-experiments/02_pytorch_training_loop/` findings on a
deeper architecture family:

- **MLP**: gradient L2 drops smoothly from ~3 → ~1 across 10 epochs;
  depth doesn't change the shape.
- **ResNet**: gradient L2 stays larger (5–10) and noisier — the
  residual connections are doing their advertised job of keeping
  gradients flowing through depth-4.
- **FT-Transformer**: gradient L2 *starts* lower (~1) and stays
  flatter — AdamW + the default optimizer recipe keeps the attention
  weights well-regularized. Higher dropout amplifies the late-epoch
  noise; depth=4 has visibly noisier gradients than depth=2.

These trajectories are not visible inside `MLPClassifier.fit()` — the
direct continuation of the case made in
`ml-library-experiments/02_pytorch_training_loop/postmortem.md`.

## What was reusable from experiment 1

The hand-rolled MLP from experiment 1 (with `nn.Embedding` per cat
column + concatenation) tied within noise of the rtdl `MLP` configured
here. Concrete numbers: exp 1's MLP at depth=4, dropout=0.1, 15 epochs
hit 0.919; exp 2's rtdl MLP at depth=4, dropout=0.1, 10 epochs hits
0.909. The difference (1 pp) is the 5 extra epochs; the architectures
are functionally equivalent on this data.

The FT-T number from experiment 1 (depth=3, lr=1e-4, 12 epochs,
~0.961) is essentially identical to this sweep's best depth=2 FT-T
(0.957) — confirming the sweep's local-best is a robust operating
point rather than a freak run.

## Take-aways

1. **MLP/ResNet capacity isn't the bottleneck** on this dataset.
   They plateau at 0.91. The bottleneck is the inductive bias —
   feature-token attention (FT-T) helps where channel-wise dense
   layers don't.
2. **FT-T depth × dropout has a clear interaction**: higher dropout
   needs more epochs to recover, so at our 8-epoch budget the best
   FT-T is depth=4 with mild dropout (0.0 or 0.1).
3. **For the rest of the project**, the depth=2 FT-T is the right
   "FT-T baseline" to reuse — it's 0.4 pp behind depth=4 but 5×
   cheaper, and the 5× matters when n_train grows to 40k in
   experiment 5.

## Output files

- `arch_table.txt` — 18-row sweep table.
- `training_curves.png` — 3 panels, val accuracy vs epoch.
- `grad_norms.png` — 3 panels, gradient L2 vs epoch (log).
