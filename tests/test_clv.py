"""Unit tests for the CLV / RFM module.

Coverage:

- ``rfm_score`` correctly maps quantiles to R/F/M scores (with R inverted)
- Segmentation rules produce the canonical names
- Validation raises on bad input
- ``summarize_clv`` computes 80/20 shares and median correctly
- ``estimate_clv`` runs end-to-end against ``lifetimes`` on a synthetic
  repeat-purchase panel, with the lifetimes dependency guarded
"""

from __future__ import annotations

import numpy as np
import pytest

from smokefreelab.analytics.clv import (
    CLVEstimate,
    RFMScore,
    _classify_segment,
    _quantile_score,
    rfm_score,
    summarize_clv,
)


class TestQuantileScore:
    """Mechanics of the quantile-binning helper."""

    def test_uniform_values_produce_full_spread(self) -> None:
        """0..99 values split into 5 bins map to {1,2,3,4,5}."""
        values = np.arange(100, dtype=float)
        scores = _quantile_score(values, n_bins=5, invert=False)
        assert set(scores.tolist()) == {1, 2, 3, 4, 5}

    def test_inversion_flips_bins(self) -> None:
        """With invert=True, the lowest input values get the highest scores."""
        values = np.arange(100, dtype=float)
        scores = _quantile_score(values, n_bins=5, invert=True)
        # lowest input gets score 5
        assert scores[0] == 5
        assert scores[-1] == 1

    def test_constant_values_all_get_top_bin(self) -> None:
        """A degenerate quantile distribution collapses to a single bin."""
        values = np.full(50, 42.0)
        scores = _quantile_score(values, n_bins=5, invert=False)
        assert set(scores.tolist()) == {5}


class TestClassifySegment:
    """Rule-based RFM segmentation labels."""

    @pytest.mark.parametrize(
        ("r", "f", "expected"),
        [
            (5, 5, "Champions"),
            (4, 4, "Champions"),
            (5, 3, "Potential Loyalists"),
            (5, 1, "New Customers"),
            (2, 5, "Loyal Customers"),
            (1, 5, "Can't Lose Them"),
            (1, 2, "Hibernating"),
            (1, 1, "Lost"),
        ],
    )
    def test_canonical_mapping(self, r: int, f: int, expected: str) -> None:
        """Textbook (R, F) boxes land in the right segment."""
        assert _classify_segment(r, f) == expected


class TestRFMScore:
    """Top-level RFM scoring API."""

    def _panel(self, seed: int = 0) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        n = 500
        ids = [f"c{i:04d}" for i in range(n)]
        recency = rng.exponential(scale=30.0, size=n)
        frequency = rng.poisson(lam=3.0, size=n).astype(float)
        monetary = rng.gamma(shape=2.0, scale=20_000.0, size=n)
        return ids, recency, frequency, monetary

    def test_returns_one_score_per_customer(self) -> None:
        """Output length matches input length."""
        ids, r, f, m = self._panel()
        scores = rfm_score(ids, r, f, m)
        assert len(scores) == len(ids)
        assert isinstance(scores[0], RFMScore)

    def test_recency_is_inverted(self) -> None:
        """Most-recent customer (lowest recency_days) gets R=5."""
        ids = [f"c{i}" for i in range(100)]
        r = np.linspace(0, 1000, 100)
        f = np.full(100, 5)
        m = np.full(100, 10_000.0)
        scores = rfm_score(ids, r, f, m)
        # Customer 0 has the lowest recency → should get R=5
        assert scores[0].r_score == 5
        # Customer 99 has the highest recency → should get R=1
        assert scores[-1].r_score == 1

    def test_frequency_not_inverted(self) -> None:
        """Highest frequency gets F=5."""
        ids = [f"c{i}" for i in range(100)]
        r = np.full(100, 10.0)
        f = np.arange(100)
        m = np.full(100, 10_000.0)
        scores = rfm_score(ids, r, f, m)
        assert scores[-1].f_score == 5  # most frequent

    def test_rfm_code_is_three_digit_string(self) -> None:
        """rfm_code concatenates scores as a string."""
        ids, r, f, m = self._panel()
        scores = rfm_score(ids, r, f, m)
        for score in scores:
            assert len(score.rfm_code) == 3
            assert score.rfm_code == f"{score.r_score}{score.f_score}{score.m_score}"

    def test_segment_is_canonical_label(self) -> None:
        """Every customer gets a known segment name."""
        known = {
            "Champions",
            "Loyal Customers",
            "Potential Loyalists",
            "New Customers",
            "Promising",
            "Needs Attention",
            "About To Sleep",
            "At Risk",
            "Can't Lose Them",
            "Hibernating",
            "Lost",
        }
        ids, r, f, m = self._panel()
        scores = rfm_score(ids, r, f, m)
        assert {s.segment for s in scores} <= known

    def test_rejects_mismatched_lengths(self) -> None:
        """Length mismatch raises."""
        with pytest.raises(ValueError, match="same length"):
            rfm_score(["a", "b"], [1.0, 2.0, 3.0], [1, 2], [10.0, 20.0])

    def test_rejects_small_n_bins(self) -> None:
        """n_bins < 2 is nonsensical."""
        with pytest.raises(ValueError, match="n_bins"):
            rfm_score(["a", "b", "c"], [1.0, 2.0, 3.0], [1, 2, 3], [10.0, 20.0, 30.0], n_bins=1)

    def test_rejects_negative_values(self) -> None:
        """Negative recency/frequency/monetary raises."""
        with pytest.raises(ValueError, match="non-negative"):
            rfm_score(["a", "b", "c"], [1.0, -2.0, 3.0], [1, 2, 3], [10.0, 20.0, 30.0])


class TestSummarizeCLV:
    """Portfolio-level CLV aggregates."""

    def test_empty_raises(self) -> None:
        """An empty portfolio is a caller bug."""
        with pytest.raises(ValueError, match="non-empty"):
            summarize_clv([])

    def test_top_decile_share_recovers_pareto(self) -> None:
        """A CLV distribution concentrated in the head has share > 0.5."""
        # Heavy Pareto-shaped CLV distribution
        rng = np.random.default_rng(0)
        clv_values = rng.pareto(a=1.5, size=1000) * 1_000_000
        estimates = [
            CLVEstimate(
                customer_id=f"c{i}",
                predicted_purchases=1.0,
                predicted_avg_value=v,
                clv=float(v),
                probability_alive=0.9,
            )
            for i, v in enumerate(clv_values)
        ]
        summary = summarize_clv(estimates)
        assert summary.n_customers == 1000
        # Pareto(1.5) is heavy-tailed enough that top-10% > 50% of total.
        assert summary.top_decile_share > 0.5

    def test_uniform_clv_gives_share_near_ten_percent(self) -> None:
        """A flat distribution makes top-decile share = 10%."""
        estimates = [
            CLVEstimate(
                customer_id=f"c{i}",
                predicted_purchases=1.0,
                predicted_avg_value=100.0,
                clv=100.0,
                probability_alive=0.8,
            )
            for i in range(1000)
        ]
        summary = summarize_clv(estimates)
        assert summary.top_decile_share == pytest.approx(0.1, abs=1e-9)
        assert summary.median_clv == 100.0

    def test_zero_total_gives_zero_share(self) -> None:
        """Edge case: all CLV = 0 → share is defined as 0."""
        estimates = [
            CLVEstimate(
                customer_id=f"c{i}",
                predicted_purchases=0.0,
                predicted_avg_value=0.0,
                clv=0.0,
                probability_alive=0.5,
            )
            for i in range(20)
        ]
        summary = summarize_clv(estimates)
        assert summary.top_decile_share == 0.0


class TestEstimateCLV:
    """End-to-end BG/NBD + Gamma-Gamma CLV on a tiny synthetic panel."""

    def _synthetic_panel(self) -> dict[str, list[float] | list[int] | list[str]]:
        """Two-segment panel: heavy repeat buyers and one-time buyers."""
        rng = np.random.default_rng(42)
        # 40 repeat buyers with 3-10 transactions
        repeat_freq = rng.integers(3, 10, size=40).tolist()
        repeat_recency = rng.uniform(30, 180, size=40).tolist()
        repeat_obs = [365.0] * 40
        repeat_monetary = rng.uniform(15_000, 60_000, size=40).tolist()
        # 20 one-time buyers: freq=0 per BG/NBD convention
        once_freq = [0] * 20
        once_recency = [0.0] * 20
        once_obs = [365.0] * 20
        once_monetary = [0.0] * 20

        return {
            "customer_id": [f"c{i:03d}" for i in range(60)],
            "frequency": list(repeat_freq) + once_freq,
            "recency": list(repeat_recency) + once_recency,
            "observation_period": list(repeat_obs) + once_obs,
            "monetary_value": list(repeat_monetary) + once_monetary,
        }

    def test_clv_pipeline_runs_and_ranks_sensibly(self) -> None:
        """Repeat buyers should get non-zero CLV and one-timers near zero."""
        pytest.importorskip("lifetimes")
        from smokefreelab.analytics.clv import estimate_clv

        panel = self._synthetic_panel()
        estimates = estimate_clv(
            panel["customer_id"],  # type: ignore[arg-type]
            panel["frequency"],  # type: ignore[arg-type]
            panel["recency"],  # type: ignore[arg-type]
            panel["observation_period"],  # type: ignore[arg-type]
            panel["monetary_value"],  # type: ignore[arg-type]
            horizon_periods=180.0,
            discount_rate=0.0,
        )
        assert len(estimates) == 60
        # Repeat buyers (first 40) should have higher mean CLV than one-timers.
        repeat_mean = float(np.mean([e.clv for e in estimates[:40]]))
        once_mean = float(np.mean([e.clv for e in estimates[40:]]))
        assert repeat_mean > once_mean
        # All probabilities alive in [0, 1].
        assert all(0.0 <= e.probability_alive <= 1.0 for e in estimates)

    def test_rejects_mismatched_lengths(self) -> None:
        """Length mismatch raises."""
        pytest.importorskip("lifetimes")
        from smokefreelab.analytics.clv import estimate_clv

        with pytest.raises(ValueError, match="same length"):
            estimate_clv(["a", "b"], [1, 2, 3], [10.0, 20.0], [100.0, 100.0], [50.0, 50.0])

    def test_rejects_negative_values(self) -> None:
        """Negative inputs are caught before lifetimes is invoked."""
        pytest.importorskip("lifetimes")
        from smokefreelab.analytics.clv import estimate_clv

        with pytest.raises(ValueError, match="non-negative"):
            estimate_clv(["a"], [-1], [10.0], [100.0], [50.0])
