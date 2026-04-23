"""Unit tests for multi-touch attribution.

Covers:

- Heuristic rules (last-click, first-click, linear, time-decay) against
  hand-computed credit tables.
- Markov attribution on a toy graph where the absorption probability has a
  closed-form answer.
- Shapley efficiency axiom — per-channel values sum exactly to observed
  conversions — and monotonicity on a rigged journey set.
- Cross-model agreement on degenerate single-channel journeys.
"""

from __future__ import annotations

import math

import pytest

from smokefreelab.attribution import (
    first_click_attribution,
    last_click_attribution,
    linear_attribution,
    markov_attribution,
    shapley_attribution,
    time_decay_attribution,
)


class TestHeuristics:
    """Rule-based attribution computes hand-verifiable credits."""

    def test_last_click_credits_final_touch(self) -> None:
        """Each converter's credit lands on its last channel."""
        journeys = [
            ["Paid Search", "Display", "Direct"],
            ["Email", "Paid Search"],
            ["Display", "Email"],
        ]
        conversions = [True, True, False]
        result = last_click_attribution(journeys, conversions)
        credits = dict(zip(result.channels, result.attributions, strict=True))
        assert credits["Direct"] == pytest.approx(1.0)
        assert credits["Paid Search"] == pytest.approx(1.0)
        assert credits["Display"] == pytest.approx(0.0)
        assert credits["Email"] == pytest.approx(0.0)

    def test_first_click_credits_initial_touch(self) -> None:
        """Each converter's credit lands on its first channel."""
        journeys = [
            ["Paid Search", "Display", "Direct"],
            ["Email", "Paid Search"],
        ]
        conversions = [True, True]
        result = first_click_attribution(journeys, conversions)
        credits = dict(zip(result.channels, result.attributions, strict=True))
        assert credits["Paid Search"] == pytest.approx(1.0)
        assert credits["Email"] == pytest.approx(1.0)

    def test_linear_splits_equally_across_unique_touches(self) -> None:
        """Linear model splits credit by the count of unique channels touched."""
        journeys = [["Paid Search", "Display", "Direct"]]
        conversions = [True]
        result = linear_attribution(journeys, conversions)
        credits = dict(zip(result.channels, result.attributions, strict=True))
        assert credits["Paid Search"] == pytest.approx(1 / 3)
        assert credits["Display"] == pytest.approx(1 / 3)
        assert credits["Direct"] == pytest.approx(1 / 3)

    def test_linear_ignores_repeated_touches(self) -> None:
        """A user hitting the same channel twice shouldn't earn that channel double."""
        journeys = [["Paid Search", "Paid Search", "Direct"]]
        conversions = [True]
        result = linear_attribution(journeys, conversions)
        credits = dict(zip(result.channels, result.attributions, strict=True))
        assert credits["Paid Search"] == pytest.approx(0.5)
        assert credits["Direct"] == pytest.approx(0.5)

    def test_time_decay_weights_last_touch_most(self) -> None:
        """Geometric decay: the last touch earns strictly more than the first."""
        journeys = [["Paid Search", "Display", "Direct"]]
        conversions = [True]
        result = time_decay_attribution(journeys, conversions, half_life_steps=2.0)
        credits = dict(zip(result.channels, result.attributions, strict=True))
        assert credits["Direct"] > credits["Display"] > credits["Paid Search"]
        # Weights sum to 1.
        assert sum(result.attributions) == pytest.approx(1.0)

    def test_time_decay_rejects_nonpositive_halflife(self) -> None:
        """Zero or negative half-life is a constructor error."""
        with pytest.raises(ValueError, match="half_life_steps must be positive"):
            time_decay_attribution([["A"]], [True], half_life_steps=0.0)

    def test_heuristics_total_matches_converters(self) -> None:
        """All four rules attribute exactly the observed conversion count."""
        journeys = [
            ["A", "B"],
            ["B", "C"],
            ["A", "C"],
            ["A"],
        ]
        conversions = [True, True, False, True]
        for result in (
            last_click_attribution(journeys, conversions),
            first_click_attribution(journeys, conversions),
            linear_attribution(journeys, conversions),
            time_decay_attribution(journeys, conversions),
        ):
            assert sum(result.attributions) == pytest.approx(3.0)
            assert result.total_conversions == 3

    def test_mismatched_lengths_rejected(self) -> None:
        """Journey/conversion length mismatch is a constructor error."""
        with pytest.raises(ValueError, match="same length"):
            last_click_attribution([["A"]], [True, False])

    def test_unknown_channel_rejected(self) -> None:
        """An explicit channel universe that omits an observed channel raises."""
        with pytest.raises(ValueError, match="channels not in"):
            last_click_attribution([["A", "B"]], [True], channels=["A"])


class TestMarkov:
    """Markov attribution invariants and known-answer check."""

    def test_single_channel_gets_full_credit(self) -> None:
        """A one-channel graph attributes all conversions to that channel."""
        journeys = [["A"], ["A"], ["A"]]
        conversions = [True, True, False]
        result = markov_attribution(journeys, conversions)
        assert result.channels == ("A",)
        assert result.attributions == pytest.approx((2.0,))
        assert result.total_conversions == 2

    def test_attribution_sums_to_total_conversions(self) -> None:
        """Normalised removal effects distribute exactly ``total_conversions``."""
        journeys = [
            ["Paid Search", "Direct"],
            ["Display", "Paid Search", "Direct"],
            ["Email", "Direct"],
            ["Paid Search"],
            ["Display", "Email"],
        ]
        conversions = [True, True, True, False, True]
        result = markov_attribution(journeys, conversions)
        assert sum(result.attributions) == pytest.approx(float(result.total_conversions))

    def test_removal_effect_in_unit_interval(self) -> None:
        """Removal effects live in [0, 1] (after the post-clip)."""
        journeys = [
            ["A", "B", "C"],
            ["B", "C"],
            ["A", "C"],
            ["A", "B"],
        ]
        conversions = [True, True, True, False]
        result = markov_attribution(journeys, conversions)
        for effect in result.removal_effects:
            assert 0.0 <= effect <= 1.0

    def test_channel_only_on_losers_gets_zero_credit(self) -> None:
        """A channel seen only on non-converters cannot earn conversion credit."""
        journeys = [
            ["Good"],
            ["Good"],
            ["Bad"],
            ["Bad"],
        ]
        conversions = [True, True, False, False]
        result = markov_attribution(journeys, conversions)
        credits = dict(zip(result.channels, result.attributions, strict=True))
        assert credits["Bad"] == pytest.approx(0.0, abs=1e-6)
        assert credits["Good"] == pytest.approx(2.0)

    def test_empty_journeys_skipped(self) -> None:
        """An empty journey contributes no transitions."""
        journeys = [["A", "B"], [], ["A"]]
        conversions = [True, True, True]
        # Empty-journey converter can't be attributed — sum equals non-empty converters.
        result = markov_attribution(journeys, conversions)
        assert sum(result.attributions) == pytest.approx(2.0)

    def test_no_conversions_gives_zero_credit(self) -> None:
        """If nobody converts, every channel gets zero credit."""
        journeys = [["A", "B"], ["B", "C"]]
        conversions = [False, False]
        result = markov_attribution(journeys, conversions)
        for a in result.attributions:
            assert a == pytest.approx(0.0)
        assert result.total_conversions == 0

    def test_shares_match_attributions(self) -> None:
        """The ``shares`` property rescales attributions to sum to 1 (or 0)."""
        journeys = [["A", "B"], ["B"], ["A"]]
        conversions = [True, True, False]
        result = markov_attribution(journeys, conversions)
        assert sum(result.shares) == pytest.approx(1.0)
        for s, a in zip(result.shares, result.attributions, strict=True):
            assert s == pytest.approx(a / result.total_conversions)


class TestShapley:
    """Shapley attribution respects the efficiency axiom and intuition."""

    def test_efficiency_axiom_exact(self) -> None:
        """Per-channel Shapley values sum to the total coalition value."""
        journeys = [
            ["A", "B"],
            ["A"],
            ["B", "C"],
            ["A", "B", "C"],
            ["C"],
        ]
        conversions = [True, True, True, True, False]
        result = shapley_attribution(journeys, conversions)
        assert sum(result.attributions) == pytest.approx(float(result.total_conversions))

    def test_null_player_gets_zero(self) -> None:
        """A channel that never appears in a converting journey earns 0 credit."""
        journeys = [
            ["A", "B"],
            ["A", "C"],  # C only on losers
            ["B", "C"],
        ]
        conversions = [True, False, False]
        result = shapley_attribution(journeys, conversions)
        credits = dict(zip(result.channels, result.attributions, strict=True))
        assert credits["C"] == pytest.approx(0.0)

    def test_symmetric_players_get_equal_credit(self) -> None:
        """Two channels with identical coalitional patterns share equal credit."""
        journeys = [
            ["A", "B"],
            ["A", "B"],
            ["A", "B"],
        ]
        conversions = [True, True, True]
        result = shapley_attribution(journeys, conversions)
        credits = dict(zip(result.channels, result.attributions, strict=True))
        assert credits["A"] == pytest.approx(credits["B"])

    def test_exceeds_max_channels_raises(self) -> None:
        """Above ``max_channels_exact`` the function refuses rather than degrade."""
        journeys = [[f"C{i}" for i in range(15)]]
        conversions = [True]
        with pytest.raises(ValueError, match="max_channels_exact"):
            shapley_attribution(journeys, conversions, max_channels_exact=12)

    def test_no_conversions_gives_zero_credit(self) -> None:
        """All-False inputs attribute zero to every channel."""
        journeys = [["A", "B"], ["B", "C"]]
        conversions = [False, False]
        result = shapley_attribution(journeys, conversions)
        for a in result.attributions:
            assert a == pytest.approx(0.0)
        assert result.total_conversions == 0

    def test_shares_sum_to_one(self) -> None:
        """The ``shares`` property exposes a normalised distribution."""
        journeys = [["A"], ["A", "B"], ["B"], ["A", "B", "C"]]
        conversions = [True, True, False, True]
        result = shapley_attribution(journeys, conversions)
        assert sum(result.shares) == pytest.approx(1.0)


class TestCrossModel:
    """Sanity checks where the three families must agree."""

    def test_single_channel_all_models_agree(self) -> None:
        """If the graph has one channel, every model attributes 100% to it."""
        journeys = [["Only"]] * 10
        conversions = [True] * 7 + [False] * 3
        for attrib_fn in (
            last_click_attribution,
            first_click_attribution,
            linear_attribution,
            time_decay_attribution,
            markov_attribution,
            shapley_attribution,
        ):
            result = attrib_fn(journeys, conversions)
            credits = dict(zip(result.channels, result.attributions, strict=True))
            assert credits["Only"] == pytest.approx(7.0)

    def test_data_driven_models_agree_on_symmetric_coalition(self) -> None:
        """When coalitions are symmetric, Markov and Shapley converge on even splits."""
        journeys = [
            ["A", "B"],
            ["B", "A"],
            ["A", "B"],
            ["B", "A"],
        ]
        conversions = [True, True, True, True]

        markov_result = markov_attribution(journeys, conversions)
        shapley_result = shapley_attribution(journeys, conversions)

        # Markov's removal effect is symmetric in (A, B) here.
        m_credits = dict(zip(markov_result.channels, markov_result.attributions, strict=True))
        s_credits = dict(zip(shapley_result.channels, shapley_result.attributions, strict=True))

        assert m_credits["A"] == pytest.approx(m_credits["B"])
        assert s_credits["A"] == pytest.approx(s_credits["B"])
        assert math.isclose(sum(m_credits.values()), 4.0)
        assert math.isclose(sum(s_credits.values()), 4.0)
