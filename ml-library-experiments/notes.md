# Working notes

Diary appended as work proceeds. Each experiment has its own `summary.md`
(APIs + why) and `postmortem.md` (what actually happened) under
`NN_*/`. This file tracks the cross-experiment journey.

## Setup

- Forked the conventions of `sklearn-experiments/` exactly: top-level
  `pyproject.toml` + `uv.lock`, per-experiment `run.py` / `summary.md` /
  `postmortem.md` / `outputs/`.
- `uv sync` with the full dep set (xgboost, lightgbm, catboost, optuna,
  shap, torch, transformers, sentence-transformers, datasets, numpyro,
  jax, statsmodels, imbalanced-learn) pulled ~3GB; CUDA libs come along
  with PyPI torch even on a CPU machine.

### Network and torch
- Tried to use the `https://download.pytorch.org/whl/cpu` index for
  CPU-only torch to skip the NVIDIA libs — that endpoint is blocked
  by the sandbox (`x-deny-reason: host_not_allowed`). Fell back to the
  PyPI torch wheel, which pulls in ~2GB of CUDA libs. `torch.cuda.is_available()`
  is still False (no `/dev/nvidia*` device), so this is wasted disk
  but otherwise fine.
- **`huggingface.co` is also blocked** — the API endpoint returns 403.
  This forced a pivot for Experiment 3: planned HF zero-shot vs linear
  probe → actual imbalanced-learn resampling deep dive. The pivot
  preserves the "tooling layer" coverage from the user's library
  picks and pairs nicely with sklearn-experiment-4's calibration work.
- `pypi.org` reachable; `s3.amazonaws.com/models.huggingface.co/` reachable
  (legacy bucket); `cdn-lfs.huggingface.co`, `api-inference.huggingface.co`,
  `datasets-server.huggingface.co` all blocked.

## Experiment 1 — boosting + categoricals

Initial plan was a regression+classification booster shootout, but
sklearn-experiment-1 already concluded that small clean tabular favors
linear models. The Plan-agent critique was right: three boosters on
569×30 will tie within noise. Reframed around the **one place the three
libraries genuinely diverge**: high-cardinality categorical handling.

Built a synthetic dataset: `load_diabetes` binarized on the median, plus
3 informative high-cardinality categorical columns. Swept cardinality
{5, 20, 50, 80}. **CatBoost dominates at every level** with log-loss
~0.5 vs ~0.8-0.9 for LightGBM (integer-coded) and XGBoost (one-hot).
That's the kind of sharp finding that justifies the experiment.

Optuna's `LightGBMPruningCallback` moved to a separate
`optuna-integration[lightgbm]` package — used LightGBM's built-in
`early_stopping(20)` instead, which gets most of the benefit without
adding another dep. SHAP's TreeExplainer worked fine after handling its
old "list of arrays" return shape (current LightGBM returns a single
2-D ndarray).

CatBoost writes a `catboost_info/` folder by default — moved before the
final commit. (Same as Optuna's `study_*.db` if you use SQLite storage.)

## Experiment 2 — PyTorch training loop instrumentation

Goal was "what does a hand-rolled training loop expose that
MLPClassifier hides". The 2×2 init×BN grid + 3-way LR schedule + CNN
comparison fits in 9s end-to-end.

The clean finding: **gradient norms reveal what accuracy cannot**.
Without BN, gradient L2 drops ~10× over 12 epochs (saturating
activations); with BN it stays healthy at 2-4. Final accuracy difference
between BN-off and BN-on is only ~1pp on this small dataset, but the
gradient story tells you BN is doing something real — and predicts that
on a harder/longer training run, the gap would widen.

LR schedules (Step, Cosine) *hurt* relative to constant LR at 12 epochs:
they decay too aggressively before the model has converged. Schedules
need to be sized to your epoch budget.

The CNN underperformed the MLP in my first version (84% vs 98%) because
`AdaptiveAvgPool2d(2)` on a 4×4 feature map averaged away too much
signal. Replaced with a direct Flatten → Linear head; CNN jumped to
97%. Real lesson on small images: pooling decisions are very fragile,
and the "always use a CNN on images" heuristic doesn't earn its keep
on 8×8 inputs.

`torch.use_deterministic_algorithms(True)` crashes on some ops; the
`manual_seed(42)` + DataLoader generator with `manual_seed(42)` combo
is enough for repeatable runs.

`float(loss)` emits a `UserWarning` about converting tensor-with-grad
to a Python scalar — used `float(loss.detach())` instead.

## Experiment 3 — imbalanced-learn (pivoted from HF transformers)

Network ate the HF transformers plan; pivoted to imbalanced-learn.
The new question — **does explicit resampling beat threshold tuning at
severe imbalance?** — turned out to give a very clean finding:

> At 99/1 imbalance, plain LogReg + threshold-tuning (F1=0.327, th=0.137)
> beats every imblearn resampler (best of those: SMOTE-family at
> F1=0.246) AND beats `class_weight="balanced"` (F1=0.246).

The popular SMOTE tutorial advice is **actively wrong** at this severity.
Reading the imblearn docs after the fact, this is mentioned in passing.
The PR curves and CV-stability check both confirm it.

The other data point: AUC for the resampled fits is *higher* than for
the do-nothing baseline (0.67 vs 0.64), but their AP (the metric that
actually matters for imbalanced) is *lower* (0.13 vs 0.21). Same lesson
as sklearn-experiment-4 made: **AUC measures ranking, threshold-dependent
metrics measure the operating point**, and they routinely disagree.

API gotcha: **always use `imblearn.pipeline.Pipeline`** when a sampler is
in the chain. Sklearn's pipeline would leak resampled rows into CV folds.

## Experiment 4 — NumPyro Bayesian regression

Used NumPyro over PyMC for JIT/CPU speed (Plan agent's recommendation).
JAX device probing is real — set `JAX_PLATFORMS=cpu` env var *and*
`jax.config.update("jax_platform_name", "cpu")` *before* any other JAX
import or use, otherwise it tries (and fails) to find a GPU.

Three model fits in ~23s wall:

- Raw `load_diabetes`: Bayesian Normal-prior CIs essentially match
  statsmodels OLS CIs (weakly-informative prior + n=442 makes the
  posterior look frequentist). **The numerical value of Bayes is small
  here; the interpretive value remains.**
- Noise-augmented (10 N(0,1) cols): four-way comparison Bayes
  Normal-prior vs Bayes Horseshoe vs Lasso vs Ridge. Three different
  winners depending on metric: **Lasso wins noise-rejection by a hair
  (0.92 vs Horseshoe 1.03), Ridge wins signal recovery 4× over (2.16
  vs ~9), and Horseshoe is the only sparsity method that ships
  calibrated uncertainty intervals.** That feels like the right summary
  of "why use Bayes": it's not strictly better, it's just a different
  trade.
- Posterior predictive check on the raw model shows the observed `y`
  has heavier right tails than the Normal-likelihood replicates —
  generative diagnostic only available with a full posterior.

`chain_method="sequential"` faster than `"parallel"` on CPU at this size
(process-spawn overhead > parallelism gain). Horseshoe takes ~50% longer
per sample than Normal-prior due to its funnel-y geometry, even with
half the budget; the regularized variant (Piironen & Vehtari) is much
better than plain horseshoe.

Lasso α=1.0 over-shrinks the signal coefficients; would use `LassoCV`
for a fairer comparison but kept the fixed α to make the noise-rejection
ranking visually clear.

## Cross-cutting

- All four scripts run in **<25s** on a single CPU (combined ~45s).
- `random_state=42` / `torch.manual_seed(42)` / `jax.random.PRNGKey(42)`
  throughout. Runs are deterministic within their library's guarantees
  (XGBoost + `tree_method="hist"`; LightGBM + `deterministic=True,
  force_row_wise=True`; CatBoost default; statsmodels exact).
- `OMP/MKL/OPENBLAS_NUM_THREADS=4` set at the top of each `run.py` to
  prevent oversubscription on small data.
- `matplotlib.use("Agg")` and headless image writes — same convention
  as sklearn-experiments.
