# Experiment 2 — PyTorch training-loop instrumentation

## Goal

Sklearn's `MLPClassifier` is a useful baseline but it's a black box — you
get a final accuracy and not much else. The point of writing a PyTorch
training loop is the instrumentation it exposes: per-epoch loss, validation
trajectory, learning-rate schedule, **gradient norms**, weight-init effects,
and the ability to mix in batch normalization or other architectural pieces.

This experiment runs the same 3-ablation design (init × BN, then LR
schedule, then MLP → CNN) on `load_digits` and reports what each ablation
*actually* changes in the training dynamics.

## Dataset

`sklearn.datasets.load_digits()` (1797 × 64, 10 classes), standard-scaled.
75/25 train/test stratified split with `random_state=42`. Reshaped to
`1 × 8 × 8` images for the CNN.

## APIs used and why

### PyTorch core
- `torch.nn.Module` subclasses for `MLP` and `CNN`.
- `torch.utils.data.{DataLoader, TensorDataset}` — standard loader idiom;
  the `generator` is seeded for shuffle reproducibility.
- `torch.optim.Adam` with `lr=1e-3` across all runs.
- `torch.nn.CrossEntropyLoss` (combines log-softmax + NLL).

### Instrumentation
- Per-batch: `loss.backward()` then *before* `opt.step()` we walk
  `model.parameters()` and sum `p.grad.pow(2).sum()` to compute a true L2
  gradient norm. This is the kind of signal sklearn doesn't surface.
- Per-epoch: train loss, val loss, val accuracy, gradient norm, current LR.

### Ablation knobs
- **Weight init**: `nn.init.xavier_uniform_` vs `nn.init.kaiming_uniform_(nonlinearity="relu")`.
  Applied via `model.apply(init_fn)` to `nn.Linear` only — conv layers
  keep PyTorch's default (Kaiming).
- **Batch norm**: `nn.BatchNorm1d(hidden)` inserted between Linear and ReLU.
- **LR schedule**: `None` (constant), `StepLR(step_size=4, gamma=0.5)`,
  `CosineAnnealingLR(T_max=EPOCHS)`. Each calls `sched.step()` per epoch.

### Architectures
- **MLP**: 64 → 128 → 64 → 10, optional BN, ReLU.
- **CNN**: Conv2d(1→16, 3, pad=1) → ReLU → MaxPool(2) → Conv2d(16→32, 3, pad=1)
  → ReLU → Flatten → Linear(512→64) → ReLU → Linear(64→10).

### sklearn baseline
- `MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, batch_size=64,
  learning_rate_init=1e-3, alpha=1e-4)` — same width and base LR as the
  PyTorch MLP so the comparison isolates "what does the loop expose" rather
  than "which library has a better architecture".

### CPU pitfalls handled
- `OMP/MKL/OPENBLAS_NUM_THREADS=4` set at the top to avoid oversubscription.
- `torch.set_num_threads(4)` is set explicitly — PyTorch's CPU thread count
  is not always automatic.
- `torch.manual_seed(42)` set globally and per-model before each run for
  apples-to-apples comparison.
- `torch.use_deterministic_algorithms(True)` was *not* enabled — it crashes
  on a number of ops; the seed alone is sufficient here.

## How to read the outputs

- `mlp_init_bn_grid.png` — left: val-accuracy curves for the 2×2 init×BN
  grid; right: gradient-norm curves (log y-axis). The gradient panel is
  the interesting one: it shows BN keeping gradient flow high through
  training where no-BN runs decay.
- `lr_schedule_curves.png` — left: LR vs epoch for constant/step/cosine;
  right: val-loss vs epoch under each schedule.
- `cnn_vs_mlp.png` — best PyTorch MLP, the CNN, and the sklearn baseline
  on one set of axes.
- `results.txt` — flat scoreboard of all configurations.
