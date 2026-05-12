"""Experiment 4 — NumPyro Bayesian regression: where do credible intervals
disagree with frequentist CIs?

Three model fits on `load_diabetes`:

  (A) Normal-prior Bayesian linear regression — compare posterior 95% CIs to
      sklearn point estimates and statsmodels OLS 95% CIs.
  (B) Same regression on noise-augmented data (raw + 10 N(0,1) columns).
      Compare Normal prior vs a Finnish-style regularized horseshoe prior
      vs sklearn Lasso/Ridge. Hypothesis: the horseshoe identifies noise
      columns more cleanly than Lasso's hard sparsity while reporting
      calibrated uncertainty.
  (C) Posterior predictive check (PPC) overlay for (A).
"""
from __future__ import annotations

import os
os.environ["JAX_PLATFORMS"] = "cpu"
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "4")

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

import statsmodels.api as sm
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
N_WARMUP = 1000
N_SAMPLES = 1000
HORSESHOE_WARMUP = 500
HORSESHOE_SAMPLES = 500

numpyro.set_host_device_count(1)

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def bayes_linear(X, y):
    n, p = X.shape
    sigma = numpyro.sample("sigma", dist.HalfNormal(5.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 10.0).expand([p]).to_event(1))
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 10.0))
    mu = alpha + X @ beta
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)


def bayes_horseshoe(X, y):
    # Regularized horseshoe prior (Piironen & Vehtari 2017).
    n, p = X.shape
    sigma = numpyro.sample("sigma", dist.HalfNormal(5.0))
    tau = numpyro.sample("tau", dist.HalfCauchy(1.0))
    lam = numpyro.sample("lam", dist.HalfCauchy(1.0).expand([p]).to_event(1))
    c2 = numpyro.sample("c2", dist.InverseGamma(2.0, 8.0))
    lam_tilde = jnp.sqrt(c2) * lam / jnp.sqrt(c2 + tau ** 2 * lam ** 2)
    beta = numpyro.sample("beta", dist.Normal(0.0, tau * lam_tilde).to_event(1))
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 10.0))
    mu = alpha + X @ beta
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)


def fit_nuts(model, X, y, n_warmup=N_WARMUP, n_samples=N_SAMPLES, seed=RANDOM_STATE):
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=2, chain_method="sequential", progress_bar=False)
    key = jax.random.PRNGKey(seed)
    t0 = time.perf_counter()
    mcmc.run(key, X, y)
    elapsed = time.perf_counter() - t0
    return mcmc, elapsed


def quantiles(arr, q=(0.025, 0.5, 0.975)):
    return np.quantile(np.asarray(arr), q, axis=0)


def main():
    print("[A] Bayesian linear regression on raw load_diabetes …")
    data = load_diabetes()
    feat = list(data.feature_names)
    X_raw = data.data
    y_raw = data.target.astype(np.float64)

    # standardize predictors so priors / CI comparisons are on the same scale
    sc = StandardScaler().fit(X_raw)
    X = sc.transform(X_raw).astype(np.float64)
    y = (y_raw - y_raw.mean())  # center y; we'll learn alpha but it will be ~0

    mcmc, t_a = fit_nuts(bayes_linear, jnp.asarray(X), jnp.asarray(y))
    samples = mcmc.get_samples()
    print(f"  NUTS done in {t_a:.1f}s  | beta shape {samples['beta'].shape}")

    beta_q = quantiles(samples["beta"])  # (3, p)
    beta_post_mean = np.asarray(samples["beta"]).mean(axis=0)

    # sklearn baseline
    skl = LinearRegression().fit(X, y)
    skl_beta = skl.coef_

    # statsmodels frequentist CIs
    Xc = sm.add_constant(X)
    ols = sm.OLS(y, Xc).fit()
    ols_beta = ols.params[1:]
    ols_ci = ols.conf_int(alpha=0.05)[1:]  # (p, 2)

    # forest plot: per-feature, bayesian 95% CI vs frequentist 95% CI vs sklearn point
    p = X.shape[1]
    idx = np.arange(p)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(beta_post_mean, idx, xerr=[beta_post_mean - beta_q[0],
                                            beta_q[2] - beta_post_mean],
                fmt="o", color="C0", label="Bayesian (NumPyro, 95% CrI)", capsize=3)
    ax.errorbar(ols_beta, idx + 0.15,
                xerr=[ols_beta - ols_ci[:, 0], ols_ci[:, 1] - ols_beta],
                fmt="s", color="C1", label="statsmodels OLS (95% CI)", capsize=3)
    ax.plot(skl_beta, idx - 0.15, "x", color="C2", label="sklearn (point)")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_yticks(idx); ax.set_yticklabels(feat)
    ax.set_xlabel("coefficient")
    ax.set_title("Bayesian vs frequentist vs sklearn coefficients (raw diabetes)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "coef_forest_plot.png", dpi=120); plt.close(fig)

    lines = ["feature       bayes_mean  bayes_lo  bayes_hi  ols_mean  ols_lo   ols_hi   sklearn"]
    lines.append("-" * len(lines[0]))
    for i, name in enumerate(feat):
        lines.append(f"{name:12s} {beta_post_mean[i]:+9.3f}  {beta_q[0, i]:+8.3f}  "
                     f"{beta_q[2, i]:+8.3f}  {ols_beta[i]:+8.3f}  "
                     f"{ols_ci[i, 0]:+8.3f} {ols_ci[i, 1]:+8.3f}  {skl_beta[i]:+8.3f}")
    (OUT / "coef_comparison_table.txt").write_text("\n".join(lines) + "\n")

    # ---- (C) Posterior predictive check on raw model ----
    print("\n[C] Posterior predictive check …")
    predictive = Predictive(bayes_linear, samples)
    ppc = predictive(jax.random.PRNGKey(RANDOM_STATE + 1), jnp.asarray(X), None)
    y_rep = np.asarray(ppc["y"])  # (S, n)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(50):
        ax.hist(y_rep[i], bins=30, color="C0", alpha=0.03, density=True)
    ax.hist(y, bins=30, color="black", alpha=0.7, histtype="step",
            linewidth=2, density=True, label="observed y")
    ax.set_xlabel("y (centered)"); ax.set_ylabel("density")
    ax.set_title("Posterior predictive check — y_obs vs y_rep")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "posterior_predictive.png", dpi=120); plt.close(fig)

    # ---- (B) Horseshoe on noise-augmented diabetes ----
    print("\n[B] Horseshoe vs Normal vs Lasso/Ridge on noise-augmented diabetes …")
    rng = np.random.default_rng(RANDOM_STATE)
    n = X.shape[0]
    noise = rng.standard_normal((n, 10))
    X_aug = np.concatenate([X, noise], axis=1)
    feat_aug = feat + [f"noise_{i}" for i in range(10)]
    p_aug = X_aug.shape[1]

    print("  fit Normal-prior Bayes …")
    mcmc_n, t_n = fit_nuts(bayes_linear, jnp.asarray(X_aug), jnp.asarray(y),
                            n_warmup=N_WARMUP, n_samples=N_SAMPLES,
                            seed=RANDOM_STATE + 7)
    s_n = mcmc_n.get_samples()
    beta_n = quantiles(s_n["beta"]);  beta_n_mean = np.asarray(s_n["beta"]).mean(axis=0)
    print(f"    NUTS {t_n:.1f}s")

    print("  fit Horseshoe Bayes (smaller sample budget) …")
    mcmc_h, t_h = fit_nuts(bayes_horseshoe, jnp.asarray(X_aug), jnp.asarray(y),
                            n_warmup=HORSESHOE_WARMUP, n_samples=HORSESHOE_SAMPLES,
                            seed=RANDOM_STATE + 13)
    s_h = mcmc_h.get_samples()
    beta_h = quantiles(s_h["beta"]); beta_h_mean = np.asarray(s_h["beta"]).mean(axis=0)
    print(f"    NUTS {t_h:.1f}s")

    print("  fit sklearn Lasso (alpha=1.0) + Ridge (alpha=1.0) …")
    lasso = Lasso(alpha=1.0, max_iter=10000, random_state=RANDOM_STATE).fit(X_aug, y)
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(X_aug, y)

    # plot: 4-way coefficient comparison, focus on the 10 noise columns
    fig, ax = plt.subplots(figsize=(9, 5))
    idx = np.arange(p_aug)[::-1]
    ax.errorbar(beta_n_mean, idx + 0.2,
                xerr=[beta_n_mean - beta_n[0], beta_n[2] - beta_n_mean],
                fmt="o", color="C0", label="Bayes Normal-prior 95% CrI", capsize=2)
    ax.errorbar(beta_h_mean, idx,
                xerr=[beta_h_mean - beta_h[0], beta_h[2] - beta_h_mean],
                fmt="s", color="C2", label="Bayes Horseshoe 95% CrI", capsize=2)
    ax.plot(lasso.coef_, idx - 0.15, "x", color="C1", label="Lasso (α=1.0)")
    ax.plot(ridge.coef_, idx - 0.30, "+", color="C3", label="Ridge (α=1.0)")
    ax.axvline(0, color="gray", lw=0.8)
    # shade noise rows
    ax.axhspan(-0.5, 9.5, color="lightcoral", alpha=0.08)
    ax.set_yticks(idx); ax.set_yticklabels(feat_aug)
    ax.set_xlabel("coefficient")
    ax.set_title("Sparse-signal recovery: noise rows shaded (lower 10)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "horseshoe_vs_lasso.png", dpi=120); plt.close(fig)

    # OLS on augmented data for fair signal-recovery reference
    ols_aug = sm.OLS(y, sm.add_constant(X_aug)).fit().params[1:]

    def noise_mae(coef):
        return float(np.abs(coef[-10:]).mean())
    def signal_mae(coef):
        return float(np.abs(coef[:10] - ols_aug[:10]).mean())
    def signal_credint_width(samples):
        q = quantiles(samples)
        return float((q[2, :10] - q[0, :10]).mean())

    lines2 = [
        f"noise-coefficient mean |β| (lower is better — wants to be 0):",
        f"  Normal-prior Bayes  {noise_mae(beta_n_mean):.3f}",
        f"  Horseshoe Bayes     {noise_mae(beta_h_mean):.3f}",
        f"  Lasso α=1.0         {noise_mae(lasso.coef_):.3f}",
        f"  Ridge α=1.0         {noise_mae(ridge.coef_):.3f}",
        "",
        f"signal-recovery |β - OLS_aug| (lower is better):",
        f"  Normal-prior Bayes  {signal_mae(beta_n_mean):.3f}",
        f"  Horseshoe Bayes     {signal_mae(beta_h_mean):.3f}",
        f"  Lasso α=1.0         {signal_mae(lasso.coef_):.3f}",
        f"  Ridge α=1.0         {signal_mae(ridge.coef_):.3f}",
        "",
        f"Bayesian 95% CrI mean width on signal cols (calibrated uncertainty):",
        f"  Normal-prior Bayes  {signal_credint_width(s_n['beta']):.3f}",
        f"  Horseshoe Bayes     {signal_credint_width(s_h['beta']):.3f}",
    ]
    (OUT / "shrinkage_table.txt").write_text("\n".join(lines2) + "\n")

    summary = [
        f"raw NUTS time: {t_a:.1f}s",
        f"horseshoe NUTS time: {t_h:.1f}s",
        f"normal NUTS time (aug): {t_n:.1f}s",
        f"posterior mean |β_noise| Normal:   {noise_mae(beta_n_mean):.3f}",
        f"posterior mean |β_noise| Horseshoe: {noise_mae(beta_h_mean):.3f}",
        f"Lasso α=1.0 |β_noise| mean: {noise_mae(lasso.coef_):.3f}",
        f"Ridge α=1.0 |β_noise| mean: {noise_mae(ridge.coef_):.3f}",
    ]
    (OUT / "winner.txt").write_text("\n".join(summary) + "\n")
    print("\n" + "\n".join(summary))


if __name__ == "__main__":
    main()
