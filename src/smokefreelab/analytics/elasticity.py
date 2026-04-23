"""Price elasticity of demand — log-log OLS + hierarchical Bayes.

This module fits the constant-elasticity demand model

::

    ln(Q) = alpha + beta * ln(P) + epsilon

where ``beta`` is the own-price elasticity of quantity demanded. It also
provides a partial-pooling variant that estimates a per-category elasticity
with shrinkage toward a grand mean, for cases where a cigarette portfolio
splits naturally into sub-categories (e.g. SKM Mild, SPM Full Flavour,
kretek clove) and each has a few hundred price-quantity observations rather
than the thousands OLS would want independently.

Business context
----------------
Indonesian tobacco is a textbook elasticity problem. The government adjusts
cukai (excise) tiers annually; each tier shift is effectively a price shock
imposed across thousands of SKUs simultaneously. Whether a manufacturer
raises retail to fully pass cukai through, absorbs it into margin, or
reformulates SKUs down a cukai tier depends on ``beta`` for each category.
A ``-0.8`` (inelastic) category can carry full pass-through; a ``-1.4``
(elastic) category cannot. This is the single most important number in the
annual pricing plan.

Design choices
--------------
- **Manual OLS via numpy**, not ``statsmodels``. The module is ~50 lines of
  linear algebra; a dep for three scalars isn't worth the import cost.
- **scipy.stats.t** for the CI because the Wald interval is Gaussian and
  under-covers for small ``n``. Matches ``statsmodels.OLS`` CI to ~1e-8 on
  a dense test grid.
- **Hierarchical Bayes via runtime pymc import**, so the OLS path stays
  dependency-light for notebooks that don't need partial pooling.
- **``PriceShockScenario`` dataclass** for the cukai-shift question —
  analysts should never hand-derive ``Q_new = Q_old * (P_new / P_old)^beta``
  in a deck.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Sequence


FloatArray = Union["Sequence[float]", np.ndarray]
"""Array-like of floats — accepts Python sequences or 1-D numpy arrays."""

StrArray = Union["Sequence[str]", np.ndarray]
"""Array-like of strings — accepts Python sequences or 1-D numpy arrays."""


@dataclass(frozen=True)
class ElasticityResult:
    """Log-log OLS elasticity fit.

    Attributes
    ----------
    elasticity : float
        Own-price elasticity ``beta``. Negative for a normal good; a value
        of ``-1.2`` means a 1% price increase reduces quantity by 1.2%.
    intercept : float
        ``alpha`` on the log scale. Exponentiating gives the quantity at
        unit price under the fitted model.
    std_error : float
        Standard error of ``beta`` from the residual variance.
    ci_low, ci_high : float
        Two-sided confidence interval on ``beta`` at ``ci_level``, using
        the Student-t critical value with ``n - 2`` degrees of freedom.
    ci_level : float
        Nominal coverage of the interval, e.g. ``0.95``.
    r_squared : float
        Coefficient of determination of the log-log fit. Below 0.3 means
        the constant-elasticity assumption is fighting the data — reach
        for a hierarchical or semi-parametric model.
    n_observations : int
        Sample size used in the fit.
    """

    elasticity: float
    intercept: float
    std_error: float
    ci_low: float
    ci_high: float
    ci_level: float
    r_squared: float
    n_observations: int

    @property
    def is_elastic(self) -> bool:
        """True when ``|beta| > 1`` — price changes dominate quantity changes."""
        return abs(self.elasticity) > 1.0

    @property
    def revenue_response(self) -> str:
        """Directional revenue response to a price increase.

        Returns one of ``"elastic"`` (revenue falls when price rises),
        ``"inelastic"`` (revenue rises), or ``"unit_elastic"`` (revenue
        neutral, within a 5% tolerance band around ``|beta| = 1``).
        """
        magnitude = abs(self.elasticity)
        if magnitude > 1.05:
            return "elastic"
        if magnitude < 0.95:
            return "inelastic"
        return "unit_elastic"


@dataclass(frozen=True)
class HierarchicalElasticityResult:
    """Partial-pooling elasticity estimates, one per category.

    Attributes
    ----------
    categories : tuple[str, ...]
        Category labels in the order the other fields are aligned to.
    elasticities : tuple[float, ...]
        Posterior-mean elasticity per category after shrinkage.
    hdi_low, hdi_high : tuple[float, ...]
        Highest Density Interval bounds at ``hdi_level``. These are the
        Bayesian analogue of a confidence interval — "the posterior puts
        ``hdi_level`` of its mass between these two numbers".
    grand_mean : float
        Posterior-mean of the hyper-prior on category elasticities. The
        value each category is shrunk toward.
    hdi_level : float
        HDI coverage, e.g. ``0.94``.
    """

    categories: tuple[str, ...]
    elasticities: tuple[float, ...]
    hdi_low: tuple[float, ...]
    hdi_high: tuple[float, ...]
    grand_mean: float
    hdi_level: float


@dataclass(frozen=True)
class PriceShockScenario:
    """Projected quantity and revenue under a proportional price change.

    Attributes
    ----------
    baseline_price, baseline_quantity : float
        Pre-shock equilibrium. Price in any unit; quantity in units-sold.
    shocked_price, expected_quantity : float
        Post-shock values under the constant-elasticity assumption.
    expected_revenue_change_abs : float
        ``P_new * Q_new - P_old * Q_old``. Same unit as ``P * Q``.
    expected_revenue_change_rel : float
        Fractional change in revenue, e.g. ``-0.08`` for an 8% drop.
    elasticity_used : float
        The ``beta`` the projection was built on — echoed back for auditability.
    """

    baseline_price: float
    baseline_quantity: float
    shocked_price: float
    expected_quantity: float
    expected_revenue_change_abs: float
    expected_revenue_change_rel: float
    elasticity_used: float


def fit_log_log(
    price: FloatArray,
    quantity: FloatArray,
    *,
    ci_level: float = 0.95,
) -> ElasticityResult:
    """Estimate constant elasticity via OLS on log-transformed data.

    Parameters
    ----------
    price : Sequence[float]
        Observed prices. Must all be strictly positive.
    quantity : Sequence[float]
        Observed quantities, same length as ``price``. Must all be strictly
        positive.
    ci_level : float, default 0.95
        Two-sided CI coverage on the elasticity estimate.

    Returns
    -------
    ElasticityResult

    Raises
    ------
    ValueError
        If lengths differ, any value is non-positive, ``n < 3``, or all
        prices are identical (rank-deficient design).

    Notes
    -----
    The estimator is closed-form:

    ::

        beta_hat = sum((ln P - ln P_bar)(ln Q - ln Q_bar))
                 / sum((ln P - ln P_bar)^2)

    with residual variance ``sigma^2 = SSE / (n - 2)`` and standard error
    ``SE(beta) = sqrt(sigma^2 / S_PP)``. The CI uses the Student-t critical
    value with ``n - 2`` degrees of freedom.
    """
    p = np.asarray(price, dtype=float)
    q = np.asarray(quantity, dtype=float)

    if p.shape != q.shape:
        raise ValueError(f"price and quantity must have the same shape, got {p.shape} vs {q.shape}")
    if p.ndim != 1:
        raise ValueError(f"price and quantity must be 1-D, got {p.ndim}-D")
    if p.size < 3:
        raise ValueError(f"need at least 3 observations to fit, got {p.size}")
    if np.any(p <= 0) or np.any(q <= 0):
        raise ValueError("price and quantity must be strictly positive for log-log fit")

    ln_p = np.log(p)
    ln_q = np.log(q)

    n = p.size
    ln_p_bar = float(ln_p.mean())
    ln_q_bar = float(ln_q.mean())
    s_pp = float(np.sum((ln_p - ln_p_bar) ** 2))
    s_pq = float(np.sum((ln_p - ln_p_bar) * (ln_q - ln_q_bar)))

    if s_pp == 0.0:
        raise ValueError("prices are all identical; elasticity is not identified")

    beta = s_pq / s_pp
    alpha = ln_q_bar - beta * ln_p_bar

    y_hat = alpha + beta * ln_p
    residuals = ln_q - y_hat
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((ln_q - ln_q_bar) ** 2))
    r_squared = 1.0 - sse / sst if sst > 0 else 0.0

    sigma2 = sse / (n - 2)
    se_beta = float(np.sqrt(sigma2 / s_pp)) if sigma2 > 0 else 0.0

    t_crit = float(stats.t.ppf((1 + ci_level) / 2, df=n - 2))
    ci_low = beta - t_crit * se_beta
    ci_high = beta + t_crit * se_beta

    return ElasticityResult(
        elasticity=float(beta),
        intercept=float(alpha),
        std_error=se_beta,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        ci_level=ci_level,
        r_squared=r_squared,
        n_observations=n,
    )


def simulate_price_shock(
    *,
    baseline_price: float,
    baseline_quantity: float,
    pct_price_change: float,
    elasticity: float,
) -> PriceShockScenario:
    """Project post-shock quantity and revenue under constant elasticity.

    Parameters
    ----------
    baseline_price : float
        Current equilibrium price, in any unit.
    baseline_quantity : float
        Current equilibrium quantity, in units sold.
    pct_price_change : float
        Fractional price change. ``0.10`` is a 10% price increase;
        ``-0.05`` is a 5% discount.
    elasticity : float
        Own-price elasticity ``beta`` (typically negative for a normal good).

    Returns
    -------
    PriceShockScenario

    Notes
    -----
    Under the constant-elasticity model, ``Q_new / Q_old = (P_new /
    P_old)^beta``. Revenue change follows directly. This is the right
    projection for small-to-moderate shocks (say, up to +/- 15%); for
    larger cukai-tier moves the linearity in log-space breaks down and a
    piecewise or non-parametric fit becomes appropriate.

    Raises
    ------
    ValueError
        If baseline price/quantity are non-positive or the shocked price
        lands at or below zero.
    """
    if baseline_price <= 0 or baseline_quantity <= 0:
        raise ValueError("baseline price and quantity must be strictly positive")
    if pct_price_change <= -1.0:
        raise ValueError(f"pct_price_change must be > -1 (>-100%), got {pct_price_change}")

    shocked_price = baseline_price * (1.0 + pct_price_change)
    quantity_ratio = (shocked_price / baseline_price) ** elasticity
    expected_quantity = baseline_quantity * quantity_ratio

    baseline_revenue = baseline_price * baseline_quantity
    new_revenue = shocked_price * expected_quantity
    change_abs = new_revenue - baseline_revenue
    change_rel = change_abs / baseline_revenue

    return PriceShockScenario(
        baseline_price=baseline_price,
        baseline_quantity=baseline_quantity,
        shocked_price=shocked_price,
        expected_quantity=expected_quantity,
        expected_revenue_change_abs=change_abs,
        expected_revenue_change_rel=change_rel,
        elasticity_used=elasticity,
    )


def fit_hierarchical(
    price: FloatArray,
    quantity: FloatArray,
    category: StrArray,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    random_seed: int = 42,
    hdi_level: float = 0.94,
) -> HierarchicalElasticityResult:
    """Fit per-category elasticity with partial pooling via pymc.

    The model is

    ::

        mu ~ Normal(-1, 1)                  # hyper-prior on grand elasticity
        tau ~ HalfNormal(0.5)               # between-category spread
        beta_k ~ Normal(mu, tau)            # per-category elasticity
        ln Q_i ~ Normal(alpha_k + beta_k * ln P_i, sigma_k)

    Categories with few observations are pulled toward ``mu``; categories
    with abundant data dominate their own posterior. This is exactly the
    structure you want for a portfolio with a long tail of niche SKUs and
    a short head of flagship SKUs.

    Parameters
    ----------
    price, quantity, category : Sequence
        Parallel arrays of length ``n``. Price/quantity must be strictly
        positive; ``category`` is the grouping variable.
    draws, tune, chains : int
        NUTS sampler configuration. The defaults are conservative for a
        k < 10 category panel with ~100 obs/category.
    random_seed : int
        Seed for reproducibility.
    hdi_level : float, default 0.94
        HDI coverage level. 0.94 is the ArviZ convention (credible but not
        overclaimed, per Gelman's recommendation).

    Returns
    -------
    HierarchicalElasticityResult

    Raises
    ------
    ImportError
        If ``pymc`` is not installed. Install the full dev environment
        (``uv sync`` at the repo root) to get it.
    ValueError
        If input shapes mismatch or contain non-positive values.
    """
    try:
        import arviz as az
        import pymc as pm
    except ImportError as exc:  # pragma: no cover - env guard
        raise ImportError(
            "fit_hierarchical requires pymc and arviz. Install with `uv sync`."
        ) from exc

    p = np.asarray(price, dtype=float)
    q = np.asarray(quantity, dtype=float)
    cat_arr = np.asarray(category)

    if not (p.shape == q.shape == cat_arr.shape):
        raise ValueError("price, quantity, category must share shape")
    if p.ndim != 1:
        raise ValueError("inputs must be 1-D")
    if np.any(p <= 0) or np.any(q <= 0):
        raise ValueError("price and quantity must be strictly positive")

    categories, cat_idx = np.unique(cat_arr, return_inverse=True)
    ln_p = np.log(p)
    ln_q = np.log(q)
    k = len(categories)

    with pm.Model():
        mu = pm.Normal("mu", mu=-1.0, sigma=1.0)
        tau = pm.HalfNormal("tau", sigma=0.5)
        beta = pm.Normal("beta", mu=mu, sigma=tau, shape=k)
        alpha = pm.Normal("alpha", mu=0.0, sigma=5.0, shape=k)
        sigma = pm.HalfNormal("sigma", sigma=1.0, shape=k)

        mu_obs = alpha[cat_idx] + beta[cat_idx] * ln_p
        pm.Normal("ln_q", mu=mu_obs, sigma=sigma[cat_idx], observed=ln_q)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=False,
            return_inferencedata=True,
        )

    beta_posterior = idata.posterior["beta"].mean(dim=("chain", "draw")).values
    hdi = az.hdi(idata, hdi_prob=hdi_level)["beta"].values
    grand_mean = float(idata.posterior["mu"].mean().values)

    return HierarchicalElasticityResult(
        categories=tuple(str(c) for c in categories),
        elasticities=tuple(float(b) for b in beta_posterior),
        hdi_low=tuple(float(h) for h in hdi[:, 0]),
        hdi_high=tuple(float(h) for h in hdi[:, 1]),
        grand_mean=grand_mean,
        hdi_level=hdi_level,
    )
