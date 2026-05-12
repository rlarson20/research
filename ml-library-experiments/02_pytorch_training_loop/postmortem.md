# Experiment 2 — Postmortem

## Headline

A hand-rolled PyTorch MLP (`xavier_bnOn`) hits 0.980 val accuracy on
`load_digits` in 12 epochs vs sklearn's `MLPClassifier` at 0.973 with the
same width/LR — but the *real* value of the hand-rolled loop is the
gradient-norm trajectory: BN keeps gradient L2 near 2-4 throughout
training, whereas no-BN runs drop **~10×** by epoch 12. That's invisible
in `MLPClassifier`.

## Init × BN grid (val accuracy, 12 epochs)

|                | BN off | BN on |
|---             |--------|-------|
| Xavier init    | 0.973  | **0.980** |
| Kaiming init   | 0.969  | **0.980** |

Init choice barely matters on this dataset (1 pp range BN-off, dead tie
BN-on). **BN is the lever** — and the gradient-norm panel shows why:

| Config         | grad-norm ep0 | grad-norm ep11 |
|---             |---:|---:|
| xavier_bnOff   | 1.75 | 0.22 |
| xavier_bnOn    | 2.70 | 1.69 |
| kaiming_bnOff  | 3.18 | 0.29 |
| kaiming_bnOn   | 2.72 | 3.94 |

Without BN, the gradient norm drops by ~7-10× over 12 epochs, suggesting
saturating activations and slowing learning. With BN, the gradient stays
healthy (1.6-3.9), so the model can still adapt at epoch 12 — and indeed
its final loss is lower and accuracy higher.

## LR schedule (best config: xavier_bnOn)

| schedule  | final val_acc | final LR |
|---        |--:|--:|
| constant  | **0.982** | 1e-3 |
| step      | 0.969 | 1.3e-4 |
| cosine    | 0.971 | ~0    |

Constant wins. At 12 epochs, both schedules anneal LR aggressively before
the model has converged — they would help on a longer run but here they
just truncate learning. **Lesson**: LR schedules need to be sized to the
number of epochs you actually train for; you can't drop a schedule that
was tuned for 100 epochs into a 12-epoch budget and expect a free lunch.

## PyTorch MLP vs CNN vs sklearn

| model                       | val_acc | fit time |
|---                          |---:|---:|
| sklearn MLPClassifier       | 0.973 | 0.63s |
| PyTorch MLP (xavier_bnOn)   | **0.980** | ~0.5s/run |
| PyTorch CNN                 | 0.971 | 0.83s |

The CNN brings no improvement on 8×8 digits — there's not enough spatial
structure in 64-pixel hand-drawn digits for the conv inductive bias to
help. **Pixels-as-flat-vector is fine when "the image" is already a tiny
flat thing.** A reminder that "use a CNN for images" is a heuristic, not
a law; on truly small images the MLP can match it with less compute.

## Surprises and gotchas

- **Gradient norms aren't bounded by accuracy.** kaiming_bnOn ends with
  gradient norm 3.94 — *higher* than epoch 0 (2.72). It's still finding
  useful directions to update in. xavier_bnOff finishes at 0.22 and only
  slightly underperforms. Gradient norm is a process metric, not an
  outcome metric.
- **`float(loss)` emits a `UserWarning`** about tensor-with-requires-grad
  → scalar conversion. Use `float(loss.detach())` or `loss.item()`.
- **`torch.use_deterministic_algorithms(True)` crashes on some ops** —
  skipped. `torch.manual_seed(42)` + DataLoader generator with
  `manual_seed(42)` is sufficient for repeated runs to match.
- **First CNN architecture (with `AdaptiveAvgPool2d(2)` after 4×4 conv)
  underperformed dramatically (0.84)** — pooling away too much spatial
  signal in an already-tiny feature map. Replaced with a flat
  Linear(32·4·4 → 64) head, which jumped to 0.971. CNN architecture
  decisions on small inputs are surprisingly fragile.

## What I'd try next

1. **Train for more epochs (50-100)** so the LR schedules have room to
   matter. Cosine annealing routinely wins at that length.
2. **Add `torch.optim.lr_scheduler.OneCycleLR`** — it's the modern "no
   tuning required" choice and would likely beat both Step and Cosine
   at this epoch count.
3. **Gradient clipping** (`torch.nn.utils.clip_grad_norm_`) — to see what
   it does to the kaiming_bnOn gradient-norm growth.
4. **Mixed precision (CPU bfloat16)** via `torch.autocast(device_type="cpu",
   dtype=torch.bfloat16)` — likely a no-op on this dataset size but worth
   timing on a larger one.
5. **Use `pytorch-lightning` to see how much boilerplate the abstraction
   eats** vs the ~80-line hand-rolled loop here.
