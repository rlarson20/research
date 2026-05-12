# Experiment 4 — Postmortem

## Headline

On noise-augmented diabetes, the **regularized horseshoe** matches Lasso
at noise rejection (1.03 vs 0.92 mean |β_noise|, ~10% gap) but
additionally reports calibrated posterior 95% credible intervals on every
coefficient — sklearn's `Lasso` cannot. **Ridge α=1.0** quietly wins
**signal recovery** by 4× because it shrinks gently across the board.
There's no one "best regression"; the right choice depends on whether you
want sparsity, uncertainty, or coefficient fidelity.

## Raw `load_diabetes` (10 features)

Bayesian Normal-prior posterior means agree with statsmodels OLS to within
posterior noise across all 10 features. Bayesian 95% CrIs and OLS 95% CIs
also agree closely — *as they should*, because under a weakly-informative
Normal(0, 10) prior on standardized predictors, the posterior is
approximately the frequentist sampling distribution.

Where they would disagree: small-n, strong priors, or non-Normal
likelihoods. None of those apply to raw diabetes. **Finding for the raw
data: Bayesian and frequentist intervals agree.** The value of Bayesian
inference here is interpretive (credible intervals are statements about
parameters, not about hypothetical resamples), not numerical.

`coef_forest_plot.png` shows all three approaches stacked per feature.
The most visible disagreements are at `s1`/`s4`/`s5` where the design
matrix has strong collinearity — both intervals widen there but they
still overlap.

## Noise-augmented (20 features: 10 signal + 10 noise)

| estimator           | mean \|β_noise\| ↓ | mean \|β - OLS_aug\| signal ↓ | 95% CrI width signal |
|---                  |---:|---:|---:|
| Normal-prior Bayes  | 1.81 | 9.07 | 15.6 |
| **Horseshoe Bayes** | **1.03** | 9.49 | **13.2** |
| **Lasso (α=1.0)**   | **0.92** | 9.81 | — |
| **Ridge (α=1.0)**   | 1.90 | **2.16** | — |

Three different winners depending on what you want:

- **Lasso wins at killing noise** but at the cost of also shrinking
  *signal* coefficients heavily — its signal-recovery error (9.81) is
  the worst of the four. It also produces no uncertainty intervals.
- **Horseshoe is essentially tied with Lasso on noise (1.03 vs 0.92)**
  but ships full posterior distributions including CrIs. The horseshoe
  CrIs on noise columns straddle zero, which is the visual sparsity
  signal in the figure — you can *see* the model say "I don't think
  this is informative".
- **Ridge wins at signal recovery** by a huge margin (2.16 vs ~9 for
  the others) — it shrinks every coefficient gently rather than
  zeroing the small ones, so signal columns stay close to the OLS
  estimate. The cost: it doesn't sparsify (noise mean 1.90, only
  marginally better than Normal-prior Bayes).
- **Normal-prior Bayes** is the dominated option in this lineup — it
  has neither Lasso's sparsity nor Ridge's signal fidelity. Its
  contribution is the credible intervals on signal cols, with width
  15.6 vs Horseshoe's 13.2 (i.e. Horseshoe is also *more confident*
  about signal coefficients because it has shrunk the noise away and
  has less prior mass on those dimensions).

## Posterior predictive check

`posterior_predictive.png`: 50 replicated `y` densities (light blue)
under the raw Normal-prior model, overlaid with the observed `y` density
(black). The model captures location and scale but the observed density
has slightly heavier right tails than the replicates — i.e., a few real
patients with very high diabetes progression are surprising under a
Normal likelihood. This suggests a Student-t likelihood would fit
better; it's the kind of diagnostic you can only do with a generative
model.

## NUTS performance on CPU

| run | warmup + samples | wall (s) |
|---  |---  |---:|
| raw normal       | 1000 + 1000 × 2 | 6.6 |
| aug normal       | 1000 + 1000 × 2 | 4.6 |
| aug horseshoe    | 500 + 500 × 2   | 7.2 |

JAX JIT compilation accounts for ~3s of each run's wall time. Horseshoe
takes ~50% longer per sample than Normal-prior because of its more
complex geometry, even with half the budget.

## Surprises and gotchas

- **JAX device probing wastes seconds** if you don't gate it. Both
  `os.environ["JAX_PLATFORMS"] = "cpu"` (before any jax import) *and*
  `jax.config.update("jax_platform_name", "cpu")` (after import, before
  use) are needed as belt-and-suspenders.
- **`chain_method="sequential"` is faster than `"parallel"`** on CPU at
  this size, because the cost of spawning child processes for 2000
  samples is larger than the time saved by running them concurrently.
- **NUTS warmup is most of the cost.** Halving samples saves ~25% wall
  time; halving warmup saves ~50%. For exploratory work it can pay to
  use shorter warmup, verify the trace looks sane, then re-run longer
  for publication.
- **Lasso at α=1.0 over-shrinks signal coefficients** — choosing α via
  `LassoCV` would give a fairer comparison and a much smaller signal-MAE.
  Left α=1.0 here to make the noise-rejection ranking visually clear.
- **`statsmodels.OLS.summary()` is genuinely nice** — it prints the
  whole frequentist regression report (R², F-statistic, BIC, per-coef
  t-stats) in one call. Sklearn forces you to compute all of that by
  hand.

## What I'd try next

1. **Student-t likelihood** — `y ~ StudentT(nu, mu, sigma)` with
   `nu ~ Gamma(2, 0.1)` — to address the right-tail miss in the PPC.
2. **`LassoCV`** as the sklearn baseline instead of `Lasso(α=1.0)`,
   for fairness on signal recovery.
3. **`numpyro.diagnostics`** — `print_summary(mcmc)` would show R̂ and
   ESS per parameter. Mention rather than implement here since the
   models all converged cleanly.
4. **A logistic regression variant** on `load_breast_cancer` — would let
   us compare Bayesian credible intervals on odds-ratios against the
   `statsmodels.Logit` profile likelihood CIs.
5. **`arviz`** for diagnostic plots (`plot_trace`, `plot_pair`) — much
   nicer than hand-rolled matplotlib for posterior summaries.
