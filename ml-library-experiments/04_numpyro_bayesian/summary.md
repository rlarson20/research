# Experiment 4 — Bayesian regression with NumPyro

## Goal

Three things sklearn's regression APIs can't do, but a Bayesian model can:

1. Report **calibrated uncertainty** (95% credible intervals) on every
   coefficient, not just point estimates.
2. Use **sparsity-inducing priors** (regularized horseshoe) that decide
   per-coefficient whether to shrink toward zero — and report posterior
   uncertainty about that decision.
3. Do **posterior predictive checks** to confirm the model is generating
   data that looks like the observations.

Quantify each by direct comparison to sklearn `LinearRegression` / `Lasso`
/ `Ridge` and to `statsmodels.OLS` (which *does* have CIs) on
`load_diabetes` raw and noise-augmented.

## Library choice: NumPyro over PyMC

NumPyro uses JAX/JIT for CPU NUTS sampling and is 3-10× faster than PyMC
at this problem size. On CPU, single-chain `chain_method="sequential"`
avoids process-overhead from parallel chains. Set
`JAX_PLATFORMS=cpu` *and* `jax.config.update("jax_platform_name", "cpu")`
before any other JAX import or it probes for GPU (slow).

## Models

### `bayes_linear` (Normal prior)
- `sigma ~ HalfNormal(5)`
- `alpha ~ Normal(0, 10)` (intercept)
- `beta ~ Normal(0, 10).expand([p])` (weakly informative coefficient prior)
- `y ~ Normal(alpha + X·beta, sigma)`

### `bayes_horseshoe` (regularized horseshoe, Piironen & Vehtari 2017)
- `tau ~ HalfCauchy(1)` (global scale)
- `lam_j ~ HalfCauchy(1)` per coefficient (local scale)
- `c2 ~ InverseGamma(2, 8)` (slab / regularization)
- `lam_tilde_j = sqrt(c2) · lam_j / sqrt(c2 + tau²·lam_j²)`
- `beta_j ~ Normal(0, tau · lam_tilde_j)`
- `y ~ Normal(alpha + X·beta, sigma)`

The regularized variant is much better-behaved under NUTS than the plain
horseshoe (which has a notorious funnel geometry).

## Sampling

- `NUTS` kernel inside `MCMC(num_chains=2, chain_method="sequential",
  progress_bar=False)`.
- Normal-prior models: 1000 warmup + 1000 samples × 2 chains = 4000 total.
- Horseshoe: 500 warmup + 500 samples × 2 chains = 2000 total. Horseshoe
  geometry is harder; cut budget to keep within ~30s total experiment time.
- Wall-clock on CPU (this machine): ~6s raw, ~5s normal-aug, ~7s horseshoe.

## What's being compared

Three model-vs-sklearn comparisons:

1. **Raw `load_diabetes`**: Bayes Normal-prior vs sklearn `LinearRegression`
   point estimates vs statsmodels OLS 95% CIs. Plotted as a forest plot.
2. **Noise-augmented diabetes** (10 N(0, 1) columns appended): Bayes
   Normal-prior, Bayes Horseshoe, sklearn `Lasso(α=1.0)`, sklearn
   `Ridge(α=1.0)`. Noise-rejection metric: mean |β_noise|.
3. **Posterior predictive check** on the raw model: 50 replicated
   datasets sampled from the posterior; overlay vs observed `y` density.

## How to read the outputs

- `coef_forest_plot.png` — Bayesian 95% CrI (blue), OLS 95% CI (orange),
  sklearn point (green X). Vertical reference at 0. Bands that don't
  cross 0 = "this feature has a credible effect".
- `posterior_predictive.png` — 50 light blue replicated-dataset densities
  vs the observed `y` density in black. They should be visually similar;
  systematic miss = model misspecification.
- `horseshoe_vs_lasso.png` — 4-way coefficient comparison on
  noise-augmented data, with the 10 noise rows shaded pink. Shows how
  each estimator handles the spurious features.
- `coef_comparison_table.txt` — per-feature numeric table (raw, model A).
- `shrinkage_table.txt` — three sub-tables: noise-rejection, signal-recovery
  vs OLS-on-augmented, and Bayesian credible-interval width.
- `winner.txt` — wall-clock times + summary metrics.
