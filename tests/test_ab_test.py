"""Unit tests for the two-proportion A/B engine.

These tests are deterministic (seeded RNG) and cover:

- ``ArmStats`` invariants
- Agreement with ``scipy`` and ``statsmodels`` reference numbers for the
  frequentist z-test, CI, and sample-size formula
- Bayesian posterior moments for a case with a tractable analytic answer
- SRM detection at thresholds that industry guidance flags
- Empirical demonstration that naive peeking inflates Type-I error
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest

from smokefreelab.experiment import (
    ArmStats,
    bayesian_test,
    check_srm,
    experiment_duration_days,
    frequentist_test,
    sample_size_per_arm,
    simulate_peeking_inflation,
)


class TestArmStats:
    """Dataclass invariants."""

    def test_valid_construction(self) -> None:
        """A valid arm exposes rate and standard error."""
        arm = ArmStats("C", n=1000, conversions=200)
        assert arm.rate == pytest.approx(0.2)
        assert arm.standard_error == pytest.approx(math.sqrt(0.2 * 0.8 / 1000))

    @pytest.mark.parametrize("n", [0, -1])
    def test_nonpositive_n_rejected(self, n: int) -> None:
        """Zero or negative sample size is a constructor error."""
        with pytest.raises(ValueError, match="n must be positive"):
            ArmStats("C", n=n, conversions=0)

    @pytest.mark.parametrize("conv", [-1, 11])
    def test_bad_conversions_rejected(self, conv: int) -> None:
        """Conversions outside ``[0, n]`` is a constructor error."""
        with pytest.raises(ValueError, match="conversions must be in"):
            ArmStats("C", n=10, conversions=conv)


class TestFrequentistTest:
    """Z-test numbers agree with statsmodels and scipy."""

    def test_pvalue_matches_statsmodels(self) -> None:
        """Pooled-variance p-value matches ``statsmodels.proportions_ztest``."""
        control = ArmStats("C", n=10_000, conversions=2_000)
        treatment = ArmStats("T", n=10_000, conversions=2_220)

        result = frequentist_test(control, treatment)

        expected_z, expected_p = proportions_ztest(
            np.array([2_220, 2_000]),
            np.array([10_000, 10_000]),
        )
        assert result.z_stat == pytest.approx(expected_z, rel=1e-6)
        assert result.p_value == pytest.approx(expected_p, rel=1e-6)

    def test_ci_covers_true_lift(self) -> None:
        """A large-sample 95% CI covers the generative lift."""
        rng = np.random.default_rng(42)
        n = 20_000
        c_conv = int(rng.binomial(n, 0.20))
        t_conv = int(rng.binomial(n, 0.22))
        result = frequentist_test(
            ArmStats("C", n=n, conversions=c_conv),
            ArmStats("T", n=n, conversions=t_conv),
        )
        assert result.ci_low_abs < 0.02 < result.ci_high_abs

    def test_significant_flag(self) -> None:
        """`.significant` reflects `p_value < alpha` exactly."""
        result = frequentist_test(
            ArmStats("C", n=5_000, conversions=1_000),
            ArmStats("T", n=5_000, conversions=1_150),
            alpha=0.05,
        )
        assert result.significant is (result.p_value < 0.05)

    def test_zero_rates_do_not_crash(self) -> None:
        """Zero-conversion arms return a finite, non-significant result."""
        result = frequentist_test(
            ArmStats("C", n=100, conversions=0),
            ArmStats("T", n=100, conversions=0),
        )
        assert result.p_value == 1.0
        assert result.z_stat == 0.0
        assert result.significant is False


class TestBayesianTest:
    """Posterior summaries are stable and correctly ordered."""

    def test_prob_t_beats_c_near_half_under_aa(self) -> None:
        """An A/A experiment gives P(T > C) near 0.5 with a uniform prior."""
        rng = np.random.default_rng(0)
        result = bayesian_test(
            ArmStats("C", n=5_000, conversions=1_000),
            ArmStats("T", n=5_000, conversions=1_000),
            n_draws=100_000,
            rng=rng,
        )
        assert result.prob_treatment_beats_control == pytest.approx(0.5, abs=0.02)

    def test_prob_t_beats_c_high_when_t_wins(self) -> None:
        """A clear-winner treatment gets P(T > C) > 0.99."""
        rng = np.random.default_rng(1)
        result = bayesian_test(
            ArmStats("C", n=10_000, conversions=2_000),
            ArmStats("T", n=10_000, conversions=2_300),
            n_draws=100_000,
            rng=rng,
        )
        assert result.prob_treatment_beats_control > 0.99

    def test_expected_loss_ordering(self) -> None:
        """When T clearly wins, loss of choosing T is small, loss of C is big."""
        rng = np.random.default_rng(2)
        result = bayesian_test(
            ArmStats("C", n=10_000, conversions=2_000),
            ArmStats("T", n=10_000, conversions=2_300),
            n_draws=100_000,
            rng=rng,
        )
        assert result.expected_loss_choose_treatment < 0.0005
        assert result.expected_loss_choose_control > 0.02

    def test_credible_interval_brackets_diff(self) -> None:
        """The 95% credible interval brackets the observed point estimate."""
        rng = np.random.default_rng(3)
        c = ArmStats("C", n=10_000, conversions=2_000)
        t = ArmStats("T", n=10_000, conversions=2_300)
        result = bayesian_test(c, t, n_draws=100_000, rng=rng)
        lo, hi = result.credible_interval_abs
        point = t.rate - c.rate
        assert lo < point < hi


class TestSRM:
    """Chi-square SRM detection at the 0.01 industry threshold."""

    def test_clean_5050_passes(self) -> None:
        """A near-perfect 50/50 split passes comfortably."""
        result = check_srm([50_020, 49_980])
        assert result.passed is True
        assert result.p_value > 0.5

    def test_suspicious_split_fails(self) -> None:
        """A 51/49 split on 200K users flags as SRM (p well below 0.01)."""
        result = check_srm([102_000, 98_000])
        assert result.passed is False
        assert result.p_value < 0.001

    def test_custom_ratios(self) -> None:
        """Non-50/50 allocations (e.g. 90/10 ramp) check against custom ratios."""
        result = check_srm([9_000, 1_000], expected_ratios=(0.9, 0.1))
        assert result.passed is True

    def test_ratio_sum_validated(self) -> None:
        """Ratios not summing to 1 raise ``ValueError``."""
        with pytest.raises(ValueError, match="must sum to 1"):
            check_srm([100, 100], expected_ratios=(0.4, 0.4))


class TestPower:
    """Sample-size formula agrees with ``statsmodels`` to within one unit."""

    @pytest.mark.parametrize(
        ("baseline", "mde", "alpha", "power"),
        [
            (0.20, 0.01, 0.05, 0.80),
            (0.20, 0.02, 0.05, 0.80),
            (0.05, 0.01, 0.05, 0.90),
            (0.50, 0.02, 0.01, 0.80),
        ],
    )
    def test_matches_statsmodels(
        self, baseline: float, mde: float, alpha: float, power: float
    ) -> None:
        """Agreement with the canonical Normal approximation implementation."""
        ours = sample_size_per_arm(baseline, mde, alpha=alpha, power=power)

        pooled_var = baseline * (1 - baseline) + (baseline + mde) * (1 - baseline - mde)
        per_arm_var = pooled_var / 2
        effect = mde / math.sqrt(per_arm_var)
        theirs = NormalIndPower().solve_power(
            effect_size=effect, alpha=alpha, power=power, ratio=1.0
        )

        assert abs(ours.sample_size_per_arm - math.ceil(theirs)) <= 1

    def test_sample_size_monotonic_in_mde(self) -> None:
        """Detecting a smaller effect needs more data."""
        small = sample_size_per_arm(0.20, 0.005).sample_size_per_arm
        medium = sample_size_per_arm(0.20, 0.01).sample_size_per_arm
        large = sample_size_per_arm(0.20, 0.02).sample_size_per_arm
        assert small > medium > large

    def test_invalid_inputs_rejected(self) -> None:
        """Out-of-range inputs raise."""
        with pytest.raises(ValueError, match="baseline_rate"):
            sample_size_per_arm(1.5, 0.01)
        with pytest.raises(ValueError, match="mde_abs"):
            sample_size_per_arm(0.2, -0.01)
        with pytest.raises(ValueError, match="in \\(0, 1\\)"):
            sample_size_per_arm(0.99, 0.05)


class TestPeekingInflation:
    """Naive peeking inflates Type-I; one peek preserves it."""

    def test_no_peeking_preserves_alpha(self) -> None:
        """With n_peeks=1, observed alpha is near nominal."""
        rng = np.random.default_rng(11)
        observed = simulate_peeking_inflation(
            baseline_rate=0.2,
            n_total_per_arm=5_000,
            n_peeks=1,
            alpha=0.05,
            n_sims=1_000,
            rng=rng,
        )
        assert 0.03 < observed < 0.07

    def test_many_peeks_inflate_alpha(self) -> None:
        """Ten peeks roughly double the Type-I rate."""
        rng = np.random.default_rng(12)
        observed = simulate_peeking_inflation(
            baseline_rate=0.2,
            n_total_per_arm=5_000,
            n_peeks=10,
            alpha=0.05,
            n_sims=1_000,
            rng=rng,
        )
        assert observed > 0.09


class TestExperimentDuration:
    """Ceil division with the expected edge cases."""

    @pytest.mark.parametrize(
        ("n", "daily", "expected"),
        [
            (10_000, 1_000, 10),
            (10_001, 1_000, 11),
            (1, 1_000, 1),
            (999, 1_000, 1),
        ],
    )
    def test_ceil_semantics(self, n: int, daily: int, expected: int) -> None:
        """Duration rounds up, floors at one day."""
        assert experiment_duration_days(n, daily) == expected

    def test_zero_traffic_rejected(self) -> None:
        """Zero or negative daily traffic is a user error."""
        with pytest.raises(ValueError, match="daily_traffic_per_arm"):
            experiment_duration_days(10_000, 0)


def test_scipy_reference_unchanged() -> None:
    """Guard: if scipy ever changes ``norm.ppf(0.975)`` we want to know."""
    assert stats.norm.ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
