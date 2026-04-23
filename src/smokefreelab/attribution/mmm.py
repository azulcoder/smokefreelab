"""Marketing Mix Modelling — adstock + Hill saturation + Bayesian regression.

This module is the "how should we split next quarter's IDR 50B media budget
across TV, digital, and trade promotion?" engine. It implements the three
pillars that any credible MMM needs:

1. **Adstock transform** — a geometric-decay carry-over on media spend, so
   today's brand impact reflects the last several weeks of GRPs rather than
   just the current flight. Single-parameter (decay rate ``lambda``), which
   is what a weekly-grain panel can actually identify.
2. **Hill saturation** — a two-parameter (``k``, ``alpha``) S-curve that
   turns raw adstocked spend into effective spend. Enforces diminishing
   returns at high budgets — the single most important non-linearity in the
   media ROI story.
3. **Bayesian regression** via PyMC — partial-pooling-friendly, so the
   posterior on each channel's coefficient comes with an HDI that a media
   planner can turn into a risk-adjusted budget split.

Business context
----------------
Indonesian tobacco media strategy has shifted hard toward trade promotion
and in-store activation (the advertising ban on tobacco products makes TV
and digital uplift essentially impossible to buy directly). The question
the brand team asks is thus not "what's the ROI of Facebook?" but "what's
the ROI of a Rp 1 B shift from trade incentives to branded merchandise
drops?" This module's output — per-channel incremental response curves —
is exactly what answers that question.

Design choices
--------------
- **Separate transform + model** — callers can pipe ``apply_adstock`` and
  ``apply_hill`` into a pandas frame for EDA before committing to the
  pymc fit. This matches the workflow in ``elasticity.py``.
- **Weekly grain** — the adstock math is written for equally-spaced time
  steps; monthly or daily work too, but the decay rate interpretation
  changes and the module docstring deliberately does not default to one.
- **Runtime pymc import** so the transforms stay dependency-light for EDA-
  only notebooks. Matches ``fit_hierarchical`` in elasticity.
- **Frozen dataclass results** — serialise cleanly to JSON/Streamlit.
- **Zero media-spend units** are assumed throughout; callers should pass
  currency-denominated spend (not GRPs, not impressions), because the
  downstream response-curve reading is rupiah-per-rupiah.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


FloatArray = Union["Sequence[float]", np.ndarray]
"""Array-like of floats — Python sequence or 1-D numpy array."""


@dataclass(frozen=True)
class MMMChannelResult:
    """Per-channel MMM fit with response curve.

    Attributes
    ----------
    name : str
        Channel label, echoed back from input.
    coefficient : float
        Posterior-mean regression coefficient on the Hill-transformed
        adstocked spend. Interpreted as incremental sales per unit of
        saturated-adstock media.
    coefficient_hdi_low, coefficient_hdi_high : float
        HDI bounds on ``coefficient`` at the level returned by ``fit_mmm``.
    adstock_decay : float
        Posterior-mean geometric decay rate ``lambda`` on this channel.
    hill_k, hill_alpha : float
        Posterior-mean saturation parameters. ``k`` is the half-saturation
        spend (spend level at which response hits 50% of its asymptote).
        ``alpha`` is the curvature — 1.0 is Michaelis-Menten, higher is
        more step-like.
    total_contribution : float
        Sum of predicted contribution from this channel across the panel.
        Unit is the same as the target (e.g. IDR revenue if target is
        rupiah-denominated sales).
    share_of_contribution : float
        ``total_contribution / total_predicted_sales``. Dimensionless; sums
        to less than 1 because the baseline (intercept) absorbs the rest.
    roi : float
        ``total_contribution / total_spend`` for this channel. Unitless.
        Report as "IDR 2.7 of incremental revenue per IDR 1 of spend".
    """

    name: str
    coefficient: float
    coefficient_hdi_low: float
    coefficient_hdi_high: float
    adstock_decay: float
    hill_k: float
    hill_alpha: float
    total_contribution: float
    share_of_contribution: float
    roi: float


@dataclass(frozen=True)
class MMMResult:
    """Full MMM fit with per-channel results and model fit diagnostics.

    Attributes
    ----------
    channels : tuple[MMMChannelResult, ...]
        One entry per channel, aligned to the input order of ``fit_mmm``.
    baseline : float
        Posterior-mean intercept. Interpreted as sales at zero marketing
        spend — the "what the brand would have sold with no media" number.
    baseline_share : float
        ``baseline * n_periods / total_predicted_sales``. Typical FMCG
        decompositions land 40-70% of volume on baseline, with the balance
        on media and promotions. A baseline share below 20% is a red flag
        that the model is over-attributing to media.
    total_spend : float
        Sum of raw (un-transformed) spend across all channels and periods.
    total_predicted_sales : float
        Sum of the fitted posterior-mean response across the panel.
    r_squared : float
        In-sample R² of the posterior-mean fit on the observed target.
    hdi_level : float
        HDI coverage of the per-channel intervals, e.g. ``0.94``.
    n_periods, n_channels : int
        Panel dimensions echoed back for sanity checking.
    """

    channels: tuple[MMMChannelResult, ...]
    baseline: float
    baseline_share: float
    total_spend: float
    total_predicted_sales: float
    r_squared: float
    hdi_level: float
    n_periods: int
    n_channels: int


def apply_adstock(spend: FloatArray, decay: float) -> np.ndarray:
    """Geometric-decay adstock transform on a single-channel spend series.

    ``adstocked[t] = spend[t] + decay * adstocked[t-1]``, with
    ``adstocked[0] = spend[0]``.

    Parameters
    ----------
    spend : FloatArray
        Non-negative spend by period. Length ``n``.
    decay : float
        Geometric decay rate in ``[0, 1)``. A value of ``0`` disables
        adstock (pure flow effect); ``0.5`` means 50% of last week's
        impact persists into this week.

    Returns
    -------
    np.ndarray
        Adstocked spend series, same length as ``spend``.

    Raises
    ------
    ValueError
        If ``decay`` is outside ``[0, 1)`` or ``spend`` has negative values.

    Notes
    -----
    The convention here is "current + carry-over from prior" rather than
    the Koyck "current weighted by geometric series", which halves the
    identification stability when ``decay`` is near 1. For interpretable
    half-life, use ``half_life = -ln(2) / ln(decay)`` for ``decay > 0``.
    """
    x = np.asarray(spend, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"spend must be 1-D, got shape {x.shape}")
    if np.any(x < 0):
        raise ValueError("spend must be non-negative")
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must be in [0, 1), got {decay}")

    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + decay * out[t - 1]
    return out


def apply_hill(spend: FloatArray, k: float, alpha: float) -> np.ndarray:
    """Hill / Michaelis-Menten saturation on an adstocked-spend series.

    ``y = x^alpha / (k^alpha + x^alpha)``. Output is always in ``[0, 1)``
    regardless of input scale; multiply by a channel coefficient downstream
    to recover units.

    Parameters
    ----------
    spend : FloatArray
        Non-negative (typically adstocked) spend.
    k : float
        Half-saturation constant, same units as ``spend``. At ``spend = k``
        the response is exactly 0.5.
    alpha : float
        Curvature exponent. ``1.0`` → Michaelis-Menten; ``2-3`` → S-shaped
        with a lag and then a cliff; ``<1`` → concave-everywhere.

    Returns
    -------
    np.ndarray
        Saturated response in ``[0, 1)``.

    Raises
    ------
    ValueError
        If ``k <= 0`` or ``alpha <= 0`` or ``spend`` has negative values.
    """
    x = np.asarray(spend, dtype=float)
    if k <= 0:
        raise ValueError(f"k must be strictly positive, got {k}")
    if alpha <= 0:
        raise ValueError(f"alpha must be strictly positive, got {alpha}")
    if np.any(x < 0):
        raise ValueError("spend must be non-negative")

    x_alpha = np.power(x, alpha)
    k_alpha = k**alpha
    return np.asarray(x_alpha / (k_alpha + x_alpha), dtype=float)


def fit_mmm(
    sales: FloatArray,
    spend_by_channel: "Mapping[str, FloatArray]",
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    random_seed: int = 42,
    hdi_level: float = 0.94,
) -> MMMResult:
    """Fit a Bayesian MMM on weekly sales vs per-channel spend.

    The model is

    ::

        adstock_c,t = spend_c,t + lambda_c * adstock_c,t-1
        hill_c,t    = adstock_c,t^alpha_c / (k_c^alpha_c + adstock_c,t^alpha_c)
        sales_t     = baseline + sum_c beta_c * hill_c,t + noise
        noise       ~ Normal(0, sigma)

    with weakly-informative priors on each channel's ``(lambda, k, alpha,
    beta)`` quad and a HalfNormal on ``sigma``. This is MMM-lite — no
    seasonality regressors, no price control, no geography pooling. It is
    intentionally the simplest specification that still carries the right
    business story (adstock + saturation are the two must-haves).

    Parameters
    ----------
    sales : FloatArray
        Target series — typically weekly revenue in rupiah. Length ``n``.
    spend_by_channel : dict[str, FloatArray]
        Raw (not adstocked, not saturated) media spend per channel. All
        values must be the same length as ``sales``.
    draws, tune, chains : int
        NUTS sampler config. Defaults are conservative for a k=3..8
        channel panel with 52..156 weekly observations.
    random_seed : int
        Sampler seed for reproducibility.
    hdi_level : float, default 0.94
        HDI coverage level. 0.94 is the ArviZ convention.

    Returns
    -------
    MMMResult

    Raises
    ------
    ImportError
        If ``pymc`` is not installed.
    ValueError
        If inputs are ill-shaped or any channel has non-positive total
        spend (unidentifiable).
    """
    try:
        import arviz as az
        import pymc as pm
    except ImportError as exc:  # pragma: no cover - env guard
        raise ImportError("fit_mmm requires `pymc` and `arviz`. Install via `uv sync`.") from exc

    sales_arr = np.asarray(sales, dtype=float)
    n = len(sales_arr)
    if sales_arr.ndim != 1:
        raise ValueError(f"sales must be 1-D, got shape {sales_arr.shape}")
    if n < 8:
        raise ValueError(f"need at least 8 observations, got {n}")
    if not spend_by_channel:
        raise ValueError("spend_by_channel must have at least one channel")

    channel_names = list(spend_by_channel.keys())
    spend_matrix = np.zeros((n, len(channel_names)), dtype=float)
    for j, name in enumerate(channel_names):
        series = np.asarray(spend_by_channel[name], dtype=float)
        if series.ndim != 1:
            raise ValueError(f"spend for {name!r} must be 1-D, got {series.shape}")
        if len(series) != n:
            raise ValueError(f"spend for {name!r} has length {len(series)}, expected {n}")
        if np.any(series < 0):
            raise ValueError(f"spend for {name!r} must be non-negative")
        if float(series.sum()) <= 0.0:
            raise ValueError(f"spend for {name!r} is all-zero; channel is unidentifiable")
        spend_matrix[:, j] = series

    n_ch = len(channel_names)
    sales_scale = float(sales_arr.mean()) if sales_arr.mean() > 0 else 1.0
    spend_scale = spend_matrix.mean(axis=0)
    spend_scale = np.where(spend_scale > 0, spend_scale, 1.0)

    with pm.Model():
        # Adstock decay ∈ (0, 1), Beta-prior centred on 0.3 (moderate carry-over).
        lam = pm.Beta("lam", alpha=2.0, beta=4.0, shape=n_ch)
        # Hill half-saturation in units of spend; prior centred on the channel mean.
        k = pm.HalfNormal("k", sigma=spend_scale, shape=n_ch)
        # Hill curvature; prior keeps alpha inside (0.3, 3.0).
        alpha = pm.TruncatedNormal("alpha", mu=1.0, sigma=0.5, lower=0.3, upper=3.0, shape=n_ch)
        # Per-channel coefficient — scaled against mean sales so posterior is on O(1).
        beta = pm.HalfNormal("beta", sigma=sales_scale, shape=n_ch)
        baseline = pm.Normal("baseline", mu=sales_scale, sigma=sales_scale)
        sigma = pm.HalfNormal("sigma", sigma=0.1 * sales_scale)

        channel_contribs = []
        for j in range(n_ch):
            spend_j = spend_matrix[:, j]
            ads_terms: list[object] = [spend_j[0]]
            for t in range(1, n):
                ads_terms.append(spend_j[t] + lam[j] * ads_terms[t - 1])
            adstocked_tensor = pm.math.stack(ads_terms)
            hill = adstocked_tensor ** alpha[j] / (k[j] ** alpha[j] + adstocked_tensor ** alpha[j])
            channel_contribs.append(beta[j] * hill)

        mu = baseline + sum(channel_contribs)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=sales_arr)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=False,
            target_accept=0.9,
        )

    summary = az.summary(idata, hdi_prob=hdi_level, kind="stats")

    lam_mean = np.asarray([summary.loc[f"lam[{j}]", "mean"] for j in range(n_ch)], dtype=float)
    k_mean = np.asarray([summary.loc[f"k[{j}]", "mean"] for j in range(n_ch)], dtype=float)
    alpha_mean = np.asarray([summary.loc[f"alpha[{j}]", "mean"] for j in range(n_ch)], dtype=float)
    beta_mean = np.asarray([summary.loc[f"beta[{j}]", "mean"] for j in range(n_ch)], dtype=float)
    hdi_low_col = f"hdi_{(1 - hdi_level) / 2 * 100:.1f}%"
    hdi_high_col = f"hdi_{(1 + hdi_level) / 2 * 100:.1f}%"
    beta_hdi_low = np.asarray(
        [summary.loc[f"beta[{j}]", hdi_low_col] for j in range(n_ch)], dtype=float
    )
    beta_hdi_high = np.asarray(
        [summary.loc[f"beta[{j}]", hdi_high_col] for j in range(n_ch)], dtype=float
    )
    baseline_mean = float(summary.loc["baseline", "mean"])

    contributions = np.zeros((n, n_ch), dtype=float)
    for j in range(n_ch):
        adstocked = apply_adstock(spend_matrix[:, j], float(lam_mean[j]))
        hill = apply_hill(adstocked, float(k_mean[j]), float(alpha_mean[j]))
        contributions[:, j] = beta_mean[j] * hill

    predicted = baseline_mean + contributions.sum(axis=1)
    total_predicted = float(predicted.sum())
    ss_res = float(np.sum((sales_arr - predicted) ** 2))
    ss_tot = float(np.sum((sales_arr - sales_arr.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    channel_results: list[MMMChannelResult] = []
    for j, name in enumerate(channel_names):
        total_contrib = float(contributions[:, j].sum())
        spend_total = float(spend_matrix[:, j].sum())
        share = total_contrib / total_predicted if total_predicted > 0 else 0.0
        roi = total_contrib / spend_total if spend_total > 0 else 0.0
        channel_results.append(
            MMMChannelResult(
                name=name,
                coefficient=float(beta_mean[j]),
                coefficient_hdi_low=float(beta_hdi_low[j]),
                coefficient_hdi_high=float(beta_hdi_high[j]),
                adstock_decay=float(lam_mean[j]),
                hill_k=float(k_mean[j]),
                hill_alpha=float(alpha_mean[j]),
                total_contribution=total_contrib,
                share_of_contribution=share,
                roi=roi,
            )
        )

    return MMMResult(
        channels=tuple(channel_results),
        baseline=baseline_mean,
        baseline_share=(baseline_mean * n / total_predicted) if total_predicted > 0 else 0.0,
        total_spend=float(spend_matrix.sum()),
        total_predicted_sales=total_predicted,
        r_squared=r_squared,
        hdi_level=hdi_level,
        n_periods=n,
        n_channels=n_ch,
    )


def response_curve(
    channel_result: MMMChannelResult, *, spend_grid: FloatArray | None = None, n_points: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a response curve for a single MMM channel.

    Given a fitted channel's ``(beta, k, alpha)``, emit a grid of ``(spend,
    incremental_sales)`` pairs that can be plotted as the "how much lift
    do I get for each IDR 1 of additional spend in this channel" curve.

    Parameters
    ----------
    channel_result : MMMChannelResult
        Per-channel fit from ``fit_mmm``.
    spend_grid : FloatArray or None
        Optional custom spend grid. Defaults to ``linspace(0, 3 * k,
        n_points)`` which covers the saturation elbow for reasonable priors.
    n_points : int, default 50
        Grid size when ``spend_grid`` is None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(spend_grid, incremental_sales)``.
    """
    if spend_grid is None:
        spend_grid_arr = np.linspace(0.0, 3.0 * channel_result.hill_k, n_points)
    else:
        spend_grid_arr = np.asarray(spend_grid, dtype=float)
    hill = apply_hill(spend_grid_arr, channel_result.hill_k, channel_result.hill_alpha)
    return spend_grid_arr, channel_result.coefficient * hill
