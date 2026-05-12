"""Experiment 2 — PyTorch training-loop instrumentation.

What does a hand-rolled training loop expose that sklearn's MLPClassifier
hides? Train an MLP and a small CNN on load_digits with three deliberate
ablations:

  1. weight init: Xavier vs Kaiming
  2. batch norm: on vs off
  3. learning-rate schedule: constant vs StepLR vs CosineAnnealingLR

Log per-epoch train/val loss, accuracy, and gradient norm. Compare against
a sklearn MLPClassifier baseline number.
"""
from __future__ import annotations

import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "4")

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DEVICE = torch.device("cpu")
EPOCHS = 12
BATCH = 64
LR = 1e-3

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.set_num_threads(4)

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def get_data():
    data = load_digits()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    sc = StandardScaler().fit(X_tr)
    X_tr = sc.transform(X_tr).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)
    return X_tr, X_te, y_tr, y_te


def make_loader(X, y, batch=BATCH, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    g = torch.Generator(); g.manual_seed(RANDOM_STATE)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, generator=g)


class MLP(nn.Module):
    def __init__(self, in_dim=64, hidden=(128, 64), n_out=10, bn=False):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, n_out=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),  # 8×8 → 8×8
            nn.MaxPool2d(2),                            # 8→4
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), # 4×4 → 4×4
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64), nn.ReLU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)


def init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)


def train_one(model, loader_tr, loader_va, opt, sched=None, epochs=EPOCHS):
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "grad_norm": [], "lr": []}
    for ep in range(epochs):
        model.train()
        tot, n_b, grad_sq = 0.0, 0, 0.0
        for xb, yb in loader_tr:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            # accumulate gradient L2 norm pre-step
            gn = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    gn += float(p.grad.detach().pow(2).sum())
            grad_sq += gn
            opt.step()
            tot += float(loss.detach()); n_b += 1
        if sched is not None:
            sched.step()
        history["lr"].append(opt.param_groups[0]["lr"])
        history["train_loss"].append(tot / max(n_b, 1))
        history["grad_norm"].append((grad_sq / max(n_b, 1)) ** 0.5)

        model.eval()
        with torch.no_grad():
            v_tot, v_n, v_correct, v_count = 0.0, 0, 0, 0
            for xb, yb in loader_va:
                out = model(xb)
                v_tot += float(loss_fn(out, yb).detach()); v_n += 1
                v_correct += int((out.argmax(1) == yb).sum())
                v_count += len(yb)
            history["val_loss"].append(v_tot / max(v_n, 1))
            history["val_acc"].append(v_correct / max(v_count, 1))
    return history


def main():
    X_tr, X_te, y_tr, y_te = get_data()
    print(f"Dataset: digits  X_tr={X_tr.shape}  X_te={X_te.shape}")
    loader_tr = make_loader(X_tr, y_tr)
    loader_va = make_loader(X_te, y_te, shuffle=False)

    # ----- (a) sklearn MLP baseline -----
    print("\n[1] sklearn MLPClassifier baseline …")
    t0 = time.perf_counter()
    skl = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                        random_state=RANDOM_STATE, batch_size=BATCH,
                        learning_rate_init=LR, alpha=1e-4)
    skl.fit(X_tr, y_tr)
    skl_acc = skl.score(X_te, y_te)
    skl_t = time.perf_counter() - t0
    print(f"  sklearn MLP test_acc={skl_acc:.4f}  fit={skl_t:.2f}s")

    # ----- (b) 2×2 init × BN grid -----
    print("\n[2] Init × BatchNorm grid (4 configs, 12 epochs each) …")
    grid = {}
    for init_name, init_fn in [("xavier", init_xavier), ("kaiming", init_kaiming)]:
        for bn in (False, True):
            torch.manual_seed(RANDOM_STATE)
            m = MLP(bn=bn).apply(init_fn)
            opt = torch.optim.Adam(m.parameters(), lr=LR)
            hist = train_one(m, loader_tr, loader_va, opt)
            label = f"{init_name}_bn{'On' if bn else 'Off'}"
            grid[label] = hist
            print(f"  {label:20s} val_acc={hist['val_acc'][-1]:.4f}  "
                  f"grad_norm(ep0)={hist['grad_norm'][0]:.2f} →(ep-1)={hist['grad_norm'][-1]:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for label, h in grid.items():
        ax1.plot(h["val_acc"], label=label)
        ax2.plot(h["grad_norm"], label=label)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("val accuracy"); ax1.legend()
    ax1.set_title("Init × BN — val accuracy")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("grad L2 norm (mean over batches)")
    ax2.set_yscale("log"); ax2.legend()
    ax2.set_title("Init × BN — gradient norm")
    fig.tight_layout(); fig.savefig(OUT / "mlp_init_bn_grid.png", dpi=120); plt.close(fig)

    # ----- (c) LR schedule comparison -----
    print("\n[3] LR-schedule comparison (constant, StepLR, Cosine) on best config …")
    best_label = max(grid, key=lambda k: grid[k]["val_acc"][-1])
    print(f"  Using best config: {best_label}")
    init_fn = init_kaiming if "kaiming" in best_label else init_xavier
    bn = "bnOn" in best_label

    scheds = {}
    for sched_name in ("constant", "step", "cosine"):
        torch.manual_seed(RANDOM_STATE)
        m = MLP(bn=bn).apply(init_fn)
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        if sched_name == "constant":
            s = None
        elif sched_name == "step":
            s = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.5)
        else:
            s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        hist = train_one(m, loader_tr, loader_va, opt, sched=s)
        scheds[sched_name] = hist
        print(f"  {sched_name:9s} val_acc={hist['val_acc'][-1]:.4f}  final_lr={hist['lr'][-1]:.5f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for n, h in scheds.items():
        ax1.plot(h["lr"], label=n); ax2.plot(h["val_loss"], label=n)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("LR"); ax1.legend()
    ax1.set_title("LR schedule")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("val loss"); ax2.legend()
    ax2.set_title("Val loss by LR schedule")
    fig.tight_layout(); fig.savefig(OUT / "lr_schedule_curves.png", dpi=120); plt.close(fig)

    # ----- (d) tiny CNN -----
    print("\n[4] Tiny CNN on 8×8 digits …")
    X_tr_img = X_tr.reshape(-1, 1, 8, 8)
    X_te_img = X_te.reshape(-1, 1, 8, 8)
    loader_tr_img = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_img), torch.from_numpy(y_tr)),
        batch_size=BATCH, shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE))
    loader_va_img = DataLoader(
        TensorDataset(torch.from_numpy(X_te_img), torch.from_numpy(y_te)),
        batch_size=BATCH, shuffle=False)

    torch.manual_seed(RANDOM_STATE)
    cnn = CNN()
    cnn.apply(init_kaiming)  # for the Linear layers; conv keeps default
    opt = torch.optim.Adam(cnn.parameters(), lr=LR)
    t0 = time.perf_counter()
    cnn_hist = train_one(cnn, loader_tr_img, loader_va_img, opt)
    cnn_t = time.perf_counter() - t0
    print(f"  CNN val_acc={cnn_hist['val_acc'][-1]:.4f}  fit={cnn_t:.2f}s")

    # ----- summary plot: MLP vs CNN vs sklearn -----
    fig, ax = plt.subplots(figsize=(7, 4))
    best_mlp_hist = grid[best_label]
    ax.plot(best_mlp_hist["val_acc"], label=f"PyTorch MLP ({best_label})")
    ax.plot(cnn_hist["val_acc"], label="PyTorch CNN")
    ax.axhline(skl_acc, color="gray", ls="--", label=f"sklearn MLP final={skl_acc:.3f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy"); ax.legend()
    ax.set_title("PyTorch MLP vs CNN vs sklearn MLPClassifier")
    fig.tight_layout(); fig.savefig(OUT / "cnn_vs_mlp.png", dpi=120); plt.close(fig)

    lines = [
        f"sklearn MLP   test_acc={skl_acc:.4f}  fit={skl_t:.2f}s",
        f"PyTorch MLP   ({best_label})  val_acc={best_mlp_hist['val_acc'][-1]:.4f}",
        f"PyTorch CNN   val_acc={cnn_hist['val_acc'][-1]:.4f}  fit={cnn_t:.2f}s",
        "",
        "Init × BN grid (final val_acc):",
        *[f"  {k:20s} {v['val_acc'][-1]:.4f}" for k, v in grid.items()],
        "",
        "LR schedules (final val_acc):",
        *[f"  {k:9s} {v['val_acc'][-1]:.4f}" for k, v in scheds.items()],
    ]
    (OUT / "results.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
