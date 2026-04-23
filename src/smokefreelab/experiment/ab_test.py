"""Two-proportion A/B testing — frequentist + Bayesian + diagnostics.

This module is the statistical engine behind the SmokeFreeLab experiment
workflow. Every public entry point returns a frozen dataclass (never a raw
tuple) so notebooks and dashboards can serialise results without hand-rolling
schemas.

Business context
----------------
The GA4 sample shows the worst proportional leak at ``view_item ->
add_to_cart`` (80%). The realistic lever there is a UI-level treatment on the
product-detail page. Sizing, running, and reading out that experiment cleanly
is the core craft of FMCG product analytics — and of this module.

Design choices
--------------
- **Analytical Beta conjugate** for Bayesian inference, Monte-Carlo only for
  the posterior of the *difference* (no closed form). PyMC would be overkill
  for two-proportion and would slow notebooks down by orders of magnitude.
- **Pooled SE** for the z-statistic (textbook H0: p_c = p_t), **unpooled SE**
  for the confidence interval on the difference (standard practice — see
  Casella & Berger §10.4, Kohavi 2020 §17).
- **Sample-size formula** from Fleiss (1981): conservative unpooled variance
  under H1. Matches ``statsmodels.stats.power.NormalIndPower`` to within 1
  unit across the tested grid.
- **Peeking simulator** demonstrates why the "just check daily and stop when
  p < 0.05" heuristic is dangerous — observed Type-I inflates from 5% toward
  ~15-25% depending on peek count. This is the interview-grade proof that
  motivates Bayesian or sequential designs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class ArmStats:
    """One arm of a two-proportion experiment.

    Parameters
    ----------
    name : str
        Human-readable label ("Control", "Treatment A").
    n : int
        Number of assigned users (denominator).
    conversions : int
        Number of users who hit the success event (numerator).

    Raises
    ------
    ValueError
        If ``n <= 0`` or ``conversions`` is outside ``[0, n]``.
    """

    name: str
    n: int
    conversions: int

    def __post_init__(self) -> None:
        """Validate invariants at construction time."""
        if self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}")
        if not 0 <= self.conversions <= self.n:
            raise ValueError(f"conversions must be in [0, {self.n}], got {self.conversions}")

    @property
    def rate(self) -> float:
        """Observed conversion rate."""
        return self.conversions / self.n

    @property
    def standard_error(self) -> float:
        """Unpooled Wald standard error of the rate."""
        p = self.rate
        return math.sqrt(p * (1 - p) / self.n)


@dataclass(frozen=True)
class FrequentistResult:
    """Two-proportion z-test summary.

    The p-value uses the pooled-variance z-statistic (H0: p_c = p_t). The CI
    on the absolute lift uses the unpooled Wald SE — standard split per
    Casella & Berger.
    """

    control: ArmStats
    treatment: ArmStats
    alpha: float
    lift_abs: float
    lift_rel: float
    z_stat: float
    p_value: float
    ci_low_abs: float
    ci_high_abs: float

    @property
    def significant(self) -> bool:
        """Two-sided rejection of H0 at the stated alpha."""
        return self.p_value < self.alpha


@dataclass(frozen=True)
class BayesianResult:
    """Beta-Binomial posterior summary on the difference in rates.

    ``prob_treatment_beats_control`` is the posterior probability that the
    treatment arm's true CVR exceeds the control's — the number a business
    stakeholder intuitively wants. ``expected_loss_choose_treatment`` is the
    posterior mean of ``max(0, p_control - p_treatment)``, i.e. the expected
    CVR we forfeit if we ship the treatment and it is actually worse. A
    0.1pp loss on a 20% baseline is typically acceptable; a 1pp loss is not.
    """

    control: ArmStats
    treatment: ArmStats
    prior_alpha: float
    prior_beta: float
    n_draws: int
    prob_treatment_beats_control: float
    expected_loss_choose_treatment: float
    expected_loss_choose_control: float
    credible_interval_abs: tuple[float, float]
    credible_level: float


@dataclass(frozen=True)
class SRMResult:
    """Sample Ratio Mismatch check via chi-square goodness-of-fit.

    An SRM failure (p below threshold, default 0.01) usually means the
    randomisation is broken — the bucketing logic, a redirect that drops
    one arm, or a logging gap. Never read the primary metric until SRM
    passes. The 0.01 threshold is the industry default (Kohavi et al. 2020).
    """

    observed: tuple[int, ...]
    expected: tuple[float, ...]
    chi2: float
    p_value: float
    alpha: float

    @property
    def passed(self) -> bool:
        """True when p >= alpha (fail to reject the null of matched ratios)."""
        return self.p_value >= self.alpha


@dataclass(frozen=True)
class PowerResult:
    """Sample-size planner output for a two-proportion test.

    Parameters
    ----------
    baseline_rate : float
        Expected control CVR.
    mde_abs : float
        Minimum Detectable Effect in absolute points (0.01 = 1pp).
    alpha : float
        Two-sided significance level.
    power : float
        Target power (1 - beta).
    sample_size_per_arm : int
        Users required per arm to detect ``mde_abs`` at the stated alpha and
        power, assuming equal allocation.
    """

    baseline_rate: float
    mde_abs: float
    alpha: float
    power: float
    sample_size_per_arm: int

    @property
    def total_sample_size(self) -> int:
        """Both arms combined."""
        return 2 * self.sample_size_per_arm


def frequentist_test(
    control: ArmStats,
    treatment: ArmStats,
    *,
    alpha: float = 0.05,
) -> FrequentistResult:
    """Two-proportion z-test with unpooled Wald CI on the absolute lift.

    Parameters
    ----------
    control, treatment : ArmStats
        The two arms.
    alpha : float, default 0.05
        Two-sided significance level.

    Returns
    -------
    FrequentistResult

    Notes
    -----
    Under H0: p_c = p_t we pool the variance using the combined success rate;
    under H1 the variance is unpooled. We return both statistics in the
    result so downstream readers can audit the stats without re-deriving.
    """
    p_c, p_t = control.rate, treatment.rate
    n_c, n_t = control.n, treatment.n
    diff = p_t - p_c

    pooled = (control.conversions + treatment.conversions) / (n_c + n_t)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    if se_pooled == 0:
        z = 0.0
        p_value = 1.0
    else:
        z = diff / se_pooled
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    se_unpooled = math.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = diff - z_crit * se_unpooled
    ci_high = diff + z_crit * se_unpooled

    lift_rel = diff / p_c if p_c > 0 else math.nan

    return FrequentistResult(
        control=control,
        treatment=treatment,
        alpha=alpha,
        lift_abs=diff,
        lift_rel=lift_rel,
        z_stat=z,
        p_value=p_value,
        ci_low_abs=ci_low,
        ci_high_abs=ci_high,
    )


def bayesian_test(
    control: ArmStats,
    treatment: ArmStats,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    credible_level: float = 0.95,
    n_draws: int = 200_000,
    rng: np.random.Generator | None = None,
) -> BayesianResult:
    """Beta-Binomial posterior inference on the lift.

    Parameters
    ----------
    control, treatment : ArmStats
    prior_alpha, prior_beta : float, default 1.0
        Beta prior hyperparameters. The default ``Beta(1, 1)`` is uniform on
        ``[0, 1]`` — uninformative, and recovers the maximum-likelihood
        point estimate at the posterior mode. For shrinkage toward a known
        baseline CVR ``p0`` with prior strength ``s``, set
        ``prior_alpha = s * p0`` and ``prior_beta = s * (1 - p0)``.
    credible_level : float, default 0.95
        Width of the returned credible interval on the absolute lift.
    n_draws : int, default 200_000
        Monte-Carlo sample count. 200k keeps the tail CIs stable to the
        fourth decimal place — good enough for a CVR posterior.
    rng : numpy.random.Generator, optional
        Pass one for deterministic tests; default seeds from ``np.random``.

    Returns
    -------
    BayesianResult

    Notes
    -----
    The posterior of the *difference* ``p_t - p_c`` has no closed form, so
    we draw from each marginal Beta and subtract. Expected loss is
    ``E[max(0, p_c - p_t)]`` and ``E[max(0, p_t - p_c)]`` respectively.
    """
    if rng is None:
        rng = np.random.default_rng()

    post_c_alpha = prior_alpha + control.conversions
    post_c_beta = prior_beta + control.n - control.conversions
    post_t_alpha = prior_alpha + treatment.conversions
    post_t_beta = prior_beta + treatment.n - treatment.conversions

    draws_c = rng.beta(post_c_alpha, post_c_beta, size=n_draws)
    draws_t = rng.beta(post_t_alpha, post_t_beta, size=n_draws)
    diff = draws_t - draws_c

    prob_t_beats_c = float((diff > 0).mean())
    loss_choose_t = float(np.maximum(0.0, draws_c - draws_t).mean())
    loss_choose_c = float(np.maximum(0.0, draws_t - draws_c).mean())

    tail = (1 - credible_level) / 2
    ci_low, ci_high = np.quantile(diff, [tail, 1 - tail])

    return BayesianResult(
        control=control,
        treatment=treatment,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        n_draws=n_draws,
        prob_treatment_beats_control=prob_t_beats_c,
        expected_loss_choose_treatment=loss_choose_t,
        expected_loss_choose_control=loss_choose_c,
        credible_interval_abs=(float(ci_low), float(ci_high)),
        credible_level=credible_level,
    )


def check_srm(
    observed: Sequence[int],
    *,
    expected_ratios: Sequence[float] = (0.5, 0.5),
    alpha: float = 0.01,
) -> SRMResult:
    """Sample Ratio Mismatch chi-square goodness-of-fit check.

    Parameters
    ----------
    observed : sequence of int
        Observed per-arm assignment counts. Most commonly length 2.
    expected_ratios : sequence of float, default (0.5, 0.5)
        Target allocation ratios. Must sum to 1.
    alpha : float, default 0.01
        SRM detection threshold. 0.01 is the industry default — tighter
        than 0.05 because SRM false alarms are expensive (pause an
        experiment for nothing) and true SRMs usually have p << 0.001.

    Returns
    -------
    SRMResult

    Raises
    ------
    ValueError
        If ``expected_ratios`` does not sum to 1 or lengths mismatch.
    """
    observed_arr = np.asarray(observed, dtype=float)
    ratios_arr = np.asarray(expected_ratios, dtype=float)
    if observed_arr.shape != ratios_arr.shape:
        raise ValueError(
            f"length mismatch: observed={observed_arr.shape}, " f"ratios={ratios_arr.shape}"
        )
    if not math.isclose(ratios_arr.sum(), 1.0, abs_tol=1e-9):
        raise ValueError(f"expected_ratios must sum to 1, got {ratios_arr.sum()}")

    total = observed_arr.sum()
    expected_arr = ratios_arr * total
    chi2, p_value = stats.chisquare(observed_arr, expected_arr)

    return SRMResult(
        observed=tuple(int(x) for x in observed_arr),
        expected=tuple(float(x) for x in expected_arr),
        chi2=float(chi2),
        p_value=float(p_value),
        alpha=alpha,
    )


def sample_size_per_arm(
    baseline_rate: float,
    mde_abs: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> PowerResult:
    r"""Minimum per-arm sample size for a two-proportion test.

    Parameters
    ----------
    baseline_rate : float
        Expected control CVR in ``(0, 1)``.
    mde_abs : float
        Minimum Detectable Effect as an absolute difference (0.01 = 1pp).
        Must be positive and keep ``baseline_rate + mde_abs`` in ``(0, 1)``.
    alpha : float, default 0.05
    power : float, default 0.8
    two_sided : bool, default True

    Returns
    -------
    PowerResult

    Notes
    -----
    Uses the unpooled-variance Fleiss (1981) approximation:

    .. math::

       n = \frac{(z_{1-\alpha/q} + z_{power})^2
           \cdot [p_1(1-p_1) + p_2(1-p_2)]}{\delta^2}

    with ``q = 2`` for a two-sided test, ``q = 1`` for one-sided. The result
    matches ``statsmodels.stats.power.NormalIndPower`` within one unit on a
    dense grid of ``(baseline, mde)`` pairs.
    """
    if not 0 < baseline_rate < 1:
        raise ValueError(f"baseline_rate must be in (0, 1), got {baseline_rate}")
    if mde_abs <= 0:
        raise ValueError(f"mde_abs must be positive, got {mde_abs}")
    p_treatment = baseline_rate + mde_abs
    if not 0 < p_treatment < 1:
        raise ValueError(f"baseline_rate + mde_abs must be in (0, 1), got {p_treatment}")

    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_power = stats.norm.ppf(power)
    var = baseline_rate * (1 - baseline_rate) + p_treatment * (1 - p_treatment)
    n = ((z_alpha + z_power) ** 2) * var / (mde_abs**2)

    return PowerResult(
        baseline_rate=baseline_rate,
        mde_abs=mde_abs,
        alpha=alpha,
        power=power,
        sample_size_per_arm=math.ceil(n),
    )


def experiment_duration_days(
    sample_size_per_arm_value: int,
    daily_traffic_per_arm: int,
) -> int:
    """Round-up the calendar days needed to reach a planned per-arm sample size.

    Parameters
    ----------
    sample_size_per_arm_value : int
        Required sample per arm (from ``sample_size_per_arm``).
    daily_traffic_per_arm : int
        Users eligible and allocated to each arm per day.

    Returns
    -------
    int
        Minimum calendar days needed. ``ceil(n / traffic)`` with a floor of 1.

    Raises
    ------
    ValueError
        If ``daily_traffic_per_arm <= 0``.
    """
    if daily_traffic_per_arm <= 0:
        raise ValueError(f"daily_traffic_per_arm must be positive, got {daily_traffic_per_arm}")
    return max(1, math.ceil(sample_size_per_arm_value / daily_traffic_per_arm))


def simulate_peeking_inflation(
    baseline_rate: float,
    n_total_per_arm: int,
    *,
    n_peeks: int = 10,
    alpha: float = 0.05,
    n_sims: int = 2000,
    rng: np.random.Generator | None = None,
) -> float:
    """Empirical Type-I rate when a fixed-horizon z-test is peeked at.

    Simulates ``n_sims`` A/A experiments (same true rate for both arms) and
    stops each as soon as any of ``n_peeks`` evenly-spaced interim looks
    rejects at the nominal ``alpha``. Returns the fraction that rejected —
    i.e., the *true* Type-I rate achieved by the peek-and-stop policy.

    Parameters
    ----------
    baseline_rate : float
        True CVR for both arms under A/A (null).
    n_total_per_arm : int
        Horizon. The final peek hits this count.
    n_peeks : int, default 10
        Number of evenly-spaced interim analyses (includes the final).
    alpha : float, default 0.05
    n_sims : int, default 2000
        Simulation draws. 2000 gives SE of roughly 0.005 on the observed
        alpha estimate — tight enough to make the inflation point
        unambiguous, cheap enough to run in <5s.
    rng : numpy.random.Generator, optional

    Returns
    -------
    float
        Observed Type-I rate under peek-and-stop. Under a correctly-sized
        fixed-horizon test this would be exactly ``alpha``; for ``n_peeks >
        1`` it inflates monotonically.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_peeks < 1:
        raise ValueError(f"n_peeks must be >= 1, got {n_peeks}")
    if n_total_per_arm < n_peeks:
        raise ValueError(f"n_total_per_arm ({n_total_per_arm}) must be >= n_peeks ({n_peeks})")

    peek_points = np.linspace(n_total_per_arm // n_peeks, n_total_per_arm, n_peeks).astype(int)
    peek_points = np.unique(peek_points)

    rejections = 0
    for _ in range(n_sims):
        c_draws = rng.binomial(1, baseline_rate, size=n_total_per_arm)
        t_draws = rng.binomial(1, baseline_rate, size=n_total_per_arm)
        c_cum = np.cumsum(c_draws)
        t_cum = np.cumsum(t_draws)
        for k in peek_points:
            sc, st = int(c_cum[k - 1]), int(t_cum[k - 1])
            result = frequentist_test(
                ArmStats("C", int(k), sc),
                ArmStats("T", int(k), st),
                alpha=alpha,
            )
            if result.p_value < alpha:
                rejections += 1
                break

    return rejections / n_sims
