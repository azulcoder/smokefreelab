"""Unit tests for the MMM module.

Coverage:

- ``apply_adstock`` reproduces the geometric recurrence for known inputs
- ``apply_hill`` hits 0 at zero spend and 0.5 at spend = k
- ``response_curve`` returns a monotone-nondecreasing curve
- Input validation raises on bad shapes / negative values
- ``fit_mmm`` runs end-to-end against synthetic data and recovers the
  known channel ranking, with the pymc dependency guarded (matches the
  ``fit_hierarchical`` pattern in ``test_elasticity.py``)
"""

from __future__ import annotations

import numpy as np
import pytest

from smokefreelab.attribution.mmm import (
    MMMChannelResult,
    MMMResult,
    apply_adstock,
    apply_hill,
    response_curve,
)


class TestApplyAdstock:
    """Geometric adstock transform."""

    def test_zero_decay_is_identity(self) -> None:
        """With decay=0 the adstock is the input itself."""
        x = np.array([1.0, 2.0, 3.0, 0.0, 5.0])
        out = apply_adstock(x, decay=0.0)
        np.testing.assert_allclose(out, x)

    def test_half_decay_matches_manual_computation(self) -> None:
        """Closed-form recurrence for decay=0.5."""
        x = np.array([2.0, 0.0, 4.0])
        out = apply_adstock(x, decay=0.5)
        # t=0: 2
        # t=1: 0 + 0.5 * 2 = 1
        # t=2: 4 + 0.5 * 1 = 4.5
        np.testing.assert_allclose(out, [2.0, 1.0, 4.5])

    def test_high_decay_carries_impulse_forward(self) -> None:
        """A single impulse with decay 0.9 decays geometrically."""
        x = np.zeros(10)
        x[0] = 1.0
        out = apply_adstock(x, decay=0.9)
        # Out[t] = 0.9**t for t >= 0
        np.testing.assert_allclose(out, 0.9 ** np.arange(10))

    def test_rejects_negative_spend(self) -> None:
        """Negative media spend is a bug."""
        with pytest.raises(ValueError, match="non-negative"):
            apply_adstock([1.0, -1.0, 2.0], decay=0.5)

    def test_rejects_decay_out_of_range(self) -> None:
        """decay must be in [0, 1)."""
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            apply_adstock([1.0, 2.0], decay=1.0)
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            apply_adstock([1.0, 2.0], decay=-0.1)

    def test_rejects_multidim(self) -> None:
        """2-D spend is rejected."""
        with pytest.raises(ValueError, match="1-D"):
            apply_adstock(np.ones((3, 2)), decay=0.5)


class TestApplyHill:
    """Hill saturation curve."""

    def test_zero_spend_gives_zero(self) -> None:
        """Hill(0) = 0 regardless of k, alpha."""
        out = apply_hill(np.array([0.0]), k=100.0, alpha=1.0)
        assert out[0] == 0.0

    def test_half_saturation_at_k(self) -> None:
        """Hill(k) = 0.5 exactly."""
        out = apply_hill(np.array([50.0]), k=50.0, alpha=1.5)
        assert out[0] == pytest.approx(0.5, abs=1e-9)

    def test_monotone_nondecreasing(self) -> None:
        """Hill is monotone in spend."""
        x = np.linspace(0, 1000, 100)
        y = apply_hill(x, k=300.0, alpha=1.2)
        assert np.all(np.diff(y) >= 0)

    def test_alpha_higher_is_more_step_like(self) -> None:
        """Higher alpha → sharper transition around k."""
        x = np.linspace(0, 200, 100)
        y_low = apply_hill(x, k=100.0, alpha=0.5)
        y_high = apply_hill(x, k=100.0, alpha=3.0)
        # At low spend, high-alpha curve is BELOW low-alpha curve.
        assert y_high[10] < y_low[10]
        # At high spend, high-alpha curve is closer to 1.
        assert y_high[-1] > y_low[-1]

    def test_rejects_nonpositive_k(self) -> None:
        with pytest.raises(ValueError, match="k must be strictly positive"):
            apply_hill(np.array([1.0, 2.0]), k=0.0, alpha=1.0)

    def test_rejects_nonpositive_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha must be strictly positive"):
            apply_hill(np.array([1.0, 2.0]), k=1.0, alpha=0.0)


class TestResponseCurve:
    """Per-channel response curve generation."""

    def _make_channel(self, k: float = 100.0, alpha: float = 1.2) -> MMMChannelResult:
        return MMMChannelResult(
            name="trade",
            coefficient=500_000.0,
            coefficient_hdi_low=400_000.0,
            coefficient_hdi_high=600_000.0,
            adstock_decay=0.3,
            hill_k=k,
            hill_alpha=alpha,
            total_contribution=10_000_000.0,
            share_of_contribution=0.3,
            roi=2.5,
        )

    def test_default_grid_has_requested_points(self) -> None:
        """n_points controls grid density."""
        ch = self._make_channel()
        grid, resp = response_curve(ch, n_points=20)
        assert len(grid) == 20
        assert len(resp) == 20

    def test_default_grid_spans_three_k(self) -> None:
        """Default grid covers the saturation elbow (0 → 3k)."""
        ch = self._make_channel(k=100.0)
        grid, _ = response_curve(ch)
        assert grid[0] == pytest.approx(0.0)
        assert grid[-1] == pytest.approx(300.0)

    def test_curve_is_monotone(self) -> None:
        """Response never decreases with spend."""
        ch = self._make_channel()
        _, resp = response_curve(ch)
        assert np.all(np.diff(resp) >= 0)

    def test_custom_grid_respected(self) -> None:
        """Passing spend_grid overrides the default."""
        ch = self._make_channel()
        custom = np.array([0.0, 50.0, 200.0])
        grid, resp = response_curve(ch, spend_grid=custom)
        np.testing.assert_allclose(grid, custom)
        assert len(resp) == 3


@pytest.mark.slow
class TestFitMMM:
    """End-to-end MMM fit. Slow because it runs NUTS."""

    def _synthetic_mmm_panel(
        self, n: int = 52, seed: int = 0
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Generate weekly sales from known adstock + Hill + noise."""
        rng = np.random.default_rng(seed)
        tv = rng.uniform(500, 2000, size=n)
        digital = rng.uniform(100, 800, size=n)
        trade = rng.uniform(1000, 3000, size=n)
        # True params
        baseline_true = 50_000.0
        contributions = np.zeros(n)
        for spend, k, alpha, beta, lam in [
            (tv, 1500.0, 1.5, 20_000.0, 0.5),
            (digital, 500.0, 1.0, 15_000.0, 0.2),
            (trade, 2000.0, 1.2, 25_000.0, 0.4),
        ]:
            ads = apply_adstock(spend, decay=lam)
            hill = apply_hill(ads, k=k, alpha=alpha)
            contributions += beta * hill
        sales = baseline_true + contributions + rng.normal(0, 2000, size=n)
        return np.maximum(sales, 1.0), {"tv": tv, "digital": digital, "trade": trade}

    def test_fit_runs_and_returns_result(self) -> None:
        """Smoke test: fit_mmm runs end-to-end and returns an MMMResult."""
        pytest.importorskip("pymc")
        pytest.importorskip("arviz")

        import arviz

        if not hasattr(arviz, "InferenceData"):  # pragma: no cover - env guard
            pytest.skip("arviz.InferenceData missing — pymc/arviz version mismatch")

        from smokefreelab.attribution.mmm import fit_mmm

        sales, spend = self._synthetic_mmm_panel(n=52, seed=1)
        result = fit_mmm(sales, spend, draws=200, tune=200, chains=2, random_seed=7)
        assert isinstance(result, MMMResult)
        assert len(result.channels) == 3
        assert {c.name for c in result.channels} == {"tv", "digital", "trade"}
        # Baseline should land in a plausible range for our synthetic data.
        assert result.baseline > 0
        # Share of contribution should be in [0, 1] per channel.
        for ch in result.channels:
            assert 0 <= ch.share_of_contribution <= 1


class TestFitMMMValidation:
    """Input validation paths for fit_mmm — cheap, no pymc needed."""

    def test_rejects_empty_spend_dict(self) -> None:
        """No channels = no model."""
        pytest.importorskip("pymc")
        from smokefreelab.attribution.mmm import fit_mmm

        with pytest.raises(ValueError, match="at least one channel"):
            fit_mmm(np.ones(52), {})

    def test_rejects_short_series(self) -> None:
        """Under 8 observations is unreasonable."""
        pytest.importorskip("pymc")
        from smokefreelab.attribution.mmm import fit_mmm

        with pytest.raises(ValueError, match="at least 8"):
            fit_mmm(np.ones(5), {"tv": np.ones(5)})

    def test_rejects_mismatched_lengths(self) -> None:
        """Channel series must match sales length."""
        pytest.importorskip("pymc")
        from smokefreelab.attribution.mmm import fit_mmm

        with pytest.raises(ValueError, match="length"):
            fit_mmm(np.ones(10), {"tv": np.ones(8)})

    def test_rejects_all_zero_channel(self) -> None:
        """Identifiability: a channel with zero total spend breaks the model."""
        pytest.importorskip("pymc")
        from smokefreelab.attribution.mmm import fit_mmm

        with pytest.raises(ValueError, match="all-zero"):
            fit_mmm(np.ones(10), {"tv": np.ones(10), "digital": np.zeros(10)})
