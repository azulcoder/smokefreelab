"""Unit tests for the price-elasticity module.

These tests are deterministic (seeded RNG) and cover:

- ``fit_log_log`` recovers a known elasticity from synthetic data
- Shape, positivity, and rank validations raise on bad input
- ``ElasticityResult`` semantic properties (``is_elastic``,
  ``revenue_response``) match the interpretation quoted in the docstring
- ``simulate_price_shock`` respects the constant-elasticity identity
- ``fit_hierarchical`` is smoke-tested behind a pymc availability guard
  (pymc is heavy and optional at test time, per the xgboost pattern in
  ``test_propensity.py``)
"""

from __future__ import annotations

import numpy as np
import pytest

from smokefreelab.analytics.elasticity import (
    ElasticityResult,
    PriceShockScenario,
    fit_log_log,
    simulate_price_shock,
)


def _synthetic_panel(
    *,
    true_elasticity: float,
    true_intercept: float = 10.0,
    n: int = 200,
    sigma: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a price/quantity panel from the constant-elasticity DGP.

    ln Q = alpha + beta ln P + noise.
    """
    rng = np.random.default_rng(seed)
    ln_p = rng.normal(loc=np.log(25_000.0), scale=0.15, size=n)
    noise = rng.normal(loc=0.0, scale=sigma, size=n)
    ln_q = true_intercept + true_elasticity * ln_p + noise
    return np.exp(ln_p), np.exp(ln_q)


class TestFitLogLog:
    """Recovery and CI behavior of the OLS fit."""

    def test_recovers_known_elasticity_inelastic(self) -> None:
        """Inelastic DGP (beta = -0.7) is recovered to within 0.03."""
        price, quantity = _synthetic_panel(true_elasticity=-0.7, n=400, sigma=0.03)
        result = fit_log_log(price, quantity)
        assert result.elasticity == pytest.approx(-0.7, abs=0.03)
        assert result.r_squared > 0.8

    def test_recovers_known_elasticity_elastic(self) -> None:
        """Elastic DGP (beta = -1.4) is recovered to within 0.05."""
        price, quantity = _synthetic_panel(true_elasticity=-1.4, n=400, sigma=0.05)
        result = fit_log_log(price, quantity)
        assert result.elasticity == pytest.approx(-1.4, abs=0.05)

    def test_ci_covers_truth_most_of_the_time(self) -> None:
        """Nominal 95% CI covers the true elasticity in >=18 of 20 seeds."""
        hits = 0
        for seed in range(20):
            price, quantity = _synthetic_panel(true_elasticity=-1.1, n=150, sigma=0.08, seed=seed)
            result = fit_log_log(price, quantity)
            if result.ci_low <= -1.1 <= result.ci_high:
                hits += 1
        assert hits >= 18  # 18/20 = 90% empirical; 95% nominal is fine with slack

    def test_n_observations_reported(self) -> None:
        """``n_observations`` echoes input length."""
        price, quantity = _synthetic_panel(true_elasticity=-1.0, n=123)
        result = fit_log_log(price, quantity)
        assert result.n_observations == 123

    def test_rejects_mismatched_lengths(self) -> None:
        """Different-length inputs raise."""
        with pytest.raises(ValueError, match="same shape"):
            fit_log_log([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_rejects_too_few_observations(self) -> None:
        """Fewer than 3 obs is a rank/dof error."""
        with pytest.raises(ValueError, match="at least 3"):
            fit_log_log([1.0, 2.0], [3.0, 4.0])

    def test_rejects_nonpositive_values(self) -> None:
        """Zero or negative price/quantity breaks the log transform."""
        with pytest.raises(ValueError, match="strictly positive"):
            fit_log_log([1.0, 2.0, 0.0], [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="strictly positive"):
            fit_log_log([1.0, 2.0, 3.0], [1.0, -1.0, 3.0])

    def test_rejects_constant_prices(self) -> None:
        """Identical prices give a rank-deficient design."""
        with pytest.raises(ValueError, match="not identified"):
            fit_log_log([10.0, 10.0, 10.0, 10.0], [1.0, 2.0, 3.0, 4.0])

    def test_rejects_multidim_input(self) -> None:
        """2-D input is rejected explicitly."""
        with pytest.raises(ValueError, match="1-D"):
            fit_log_log(np.ones((3, 2)), np.ones((3, 2)))


class TestElasticityResultProperties:
    """Semantic properties of the result dataclass."""

    def _make(self, elasticity: float) -> ElasticityResult:
        return ElasticityResult(
            elasticity=elasticity,
            intercept=0.0,
            std_error=0.1,
            ci_low=elasticity - 0.2,
            ci_high=elasticity + 0.2,
            ci_level=0.95,
            r_squared=0.9,
            n_observations=100,
        )

    @pytest.mark.parametrize(
        ("elasticity", "expected"),
        [(-0.5, False), (-0.99, False), (-1.01, True), (-1.5, True), (-2.3, True)],
    )
    def test_is_elastic(self, elasticity: float, expected: bool) -> None:
        """``|beta| > 1`` flips the is_elastic flag."""
        assert self._make(elasticity).is_elastic is expected

    @pytest.mark.parametrize(
        ("elasticity", "expected"),
        [
            (-0.5, "inelastic"),
            (-0.8, "inelastic"),
            (-1.0, "unit_elastic"),
            (-1.04, "unit_elastic"),
            (-1.3, "elastic"),
            (-2.0, "elastic"),
        ],
    )
    def test_revenue_response(self, elasticity: float, expected: str) -> None:
        """Revenue band matches textbook interpretation."""
        assert self._make(elasticity).revenue_response == expected


class TestSimulatePriceShock:
    """Projected quantity and revenue under a price change."""

    def test_elastic_shock_reduces_revenue(self) -> None:
        """Price up + elastic demand -> revenue down."""
        scenario = simulate_price_shock(
            baseline_price=25_000.0,
            baseline_quantity=1_000_000,
            pct_price_change=0.10,
            elasticity=-1.4,
        )
        assert scenario.expected_revenue_change_rel < 0
        assert scenario.expected_quantity < 1_000_000

    def test_inelastic_shock_lifts_revenue(self) -> None:
        """Price up + inelastic demand -> revenue up."""
        scenario = simulate_price_shock(
            baseline_price=25_000.0,
            baseline_quantity=1_000_000,
            pct_price_change=0.10,
            elasticity=-0.5,
        )
        assert scenario.expected_revenue_change_rel > 0

    def test_unit_elasticity_revenue_near_flat(self) -> None:
        """At beta = -1 exactly, revenue is unchanged."""
        scenario = simulate_price_shock(
            baseline_price=25_000.0,
            baseline_quantity=1_000_000,
            pct_price_change=0.10,
            elasticity=-1.0,
        )
        assert scenario.expected_revenue_change_rel == pytest.approx(0.0, abs=1e-9)

    def test_negative_shock_raises_quantity(self) -> None:
        """A price cut (negative shock) lifts quantity for a normal good."""
        scenario = simulate_price_shock(
            baseline_price=25_000.0,
            baseline_quantity=1_000_000,
            pct_price_change=-0.05,
            elasticity=-1.2,
        )
        assert scenario.expected_quantity > 1_000_000
        assert scenario.shocked_price < 25_000.0

    def test_echo_back_elasticity(self) -> None:
        """The scenario echoes the elasticity for audit trail."""
        scenario = simulate_price_shock(
            baseline_price=100.0,
            baseline_quantity=10.0,
            pct_price_change=0.0,
            elasticity=-0.8,
        )
        assert scenario.elasticity_used == -0.8
        # Zero shock is a no-op.
        assert scenario.expected_revenue_change_abs == pytest.approx(0.0, abs=1e-9)

    def test_rejects_nonpositive_baseline(self) -> None:
        """Zero or negative baselines are rejected."""
        with pytest.raises(ValueError, match="strictly positive"):
            simulate_price_shock(
                baseline_price=0.0,
                baseline_quantity=10.0,
                pct_price_change=0.1,
                elasticity=-1.0,
            )

    def test_rejects_price_to_zero_or_below(self) -> None:
        """A -100% shock crashes price to zero — meaningless in constant-elasticity land."""
        with pytest.raises(ValueError, match="-100"):
            simulate_price_shock(
                baseline_price=100.0,
                baseline_quantity=10.0,
                pct_price_change=-1.0,
                elasticity=-1.0,
            )

    def test_is_scenario_type(self) -> None:
        """Return type is the frozen dataclass."""
        scenario = simulate_price_shock(
            baseline_price=100.0,
            baseline_quantity=10.0,
            pct_price_change=0.1,
            elasticity=-1.0,
        )
        assert isinstance(scenario, PriceShockScenario)


@pytest.mark.slow
class TestFitHierarchical:
    """Smoke test — marked ``slow`` so unit runs exclude it by default.

    pymc + NUTS sampling is 1-3 minutes on a CPU; CI runs it, workstation
    runs can opt in with ``pytest -m slow``. The stack-health check is
    deferred inside the test so pytest collection doesn't pay the pymc
    import cost on a normal run.
    """

    def test_hierarchical_recovers_category_elasticities(self) -> None:
        """Two synthetic categories -> posterior means near the truth."""
        pytest.importorskip("pymc")
        pytest.importorskip("arviz")

        import arviz  # local, post-importorskip

        if not hasattr(arviz, "InferenceData"):  # pragma: no cover - env guard
            pytest.skip("arviz.InferenceData missing — pymc/arviz version mismatch")

        from smokefreelab.analytics.elasticity import fit_hierarchical

        price_a, qty_a = _synthetic_panel(true_elasticity=-0.6, n=120, seed=1)
        price_b, qty_b = _synthetic_panel(true_elasticity=-1.3, n=120, seed=2)
        price = np.concatenate([price_a, price_b])
        quantity = np.concatenate([qty_a, qty_b])
        category = np.array(["A"] * 120 + ["B"] * 120)

        result = fit_hierarchical(
            price, quantity, category, draws=300, tune=300, chains=2, random_seed=7
        )
        assert result.categories == ("A", "B")
        # Partial pooling shrinks — tolerance is generous.
        assert result.elasticities[0] == pytest.approx(-0.6, abs=0.25)
        assert result.elasticities[1] == pytest.approx(-1.3, abs=0.25)
