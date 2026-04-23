"""Heuristic attribution rules (last-click, first-click, linear, time-decay).

These are the baseline attribution models that GA4, Adobe Analytics, and
every ad-tech dashboard ship by default. They are *not* causal — they
divide observed conversions by a rule — but they are fast, transparent,
and universally understood by stakeholders. A data-driven shop ships them
alongside Markov and Shapley so the gap between heuristic and causal
credit is itself a KPI.

Business context
----------------
The interesting comparison isn't "Markov is better than last-click" in the
abstract — it's the per-channel divergence in rupiah. A paid-search
channel credited IDR 8B under last-click but IDR 3B under Markov is the
classic example of a closer that's *riding* top-of-funnel demand; the
brand is over-investing in search and under-investing in awareness.

Design choices
--------------
- All four rules return a common ``HeuristicAttributionResult`` dataclass
  so the notebook can stack them in one DataFrame side-by-side with
  Markov and Shapley.
- ``linear`` and ``time_decay`` split credit across the *unique* channels
  in a journey, not across repeated touches of the same channel — a
  single long-path user should not drown out everyone else.
- ``time_decay`` uses a geometric decay indexed by position in the
  journey, not by wall-clock timestamp. This is the formulation GA4 uses
  and it's the one stakeholders expect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class HeuristicAttributionResult:
    """Channel-level credit from a rule-based attribution model.

    Attributes
    ----------
    method : str
        Human-readable name of the rule ("last_click", "first_click",
        "linear", "time_decay").
    channels : tuple[str, ...]
        Channel ordering.
    attributions : tuple[float, ...]
        Fractional conversion credit per channel. Sums to the number of
        converters in the input.
    total_conversions : int
        Observed conversions distributed across channels.
    """

    method: str
    channels: tuple[str, ...]
    attributions: tuple[float, ...]
    total_conversions: int

    @property
    def shares(self) -> tuple[float, ...]:
        """Fractional share per channel."""
        total = float(self.total_conversions)
        if total == 0:
            return tuple(0.0 for _ in self.channels)
        return tuple(a / total for a in self.attributions)


def _canonical_channels(
    journeys: Sequence[Sequence[str]],
    channels: Sequence[str] | None,
) -> list[str]:
    """Resolve the channel universe, validating against ``journeys``."""
    if channels is None:
        return sorted({c for j in journeys for c in j})
    unique = list(channels)
    observed = {c for j in journeys for c in j}
    missing = observed - set(unique)
    if missing:
        raise ValueError(f"journeys contain channels not in `channels`: {sorted(missing)}")
    return unique


def _validate_lengths(
    journeys: Sequence[Sequence[str]],
    conversions: Sequence[bool],
) -> None:
    if len(journeys) != len(conversions):
        raise ValueError(
            f"journeys and conversions must have the same length, "
            f"got {len(journeys)} vs {len(conversions)}"
        )


def last_click_attribution(
    journeys: Sequence[Sequence[str]],
    conversions: Sequence[bool],
    *,
    channels: Sequence[str] | None = None,
) -> HeuristicAttributionResult:
    """Credit the final touch of every converting journey."""
    _validate_lengths(journeys, conversions)
    unique = _canonical_channels(journeys, channels)
    idx = {c: i for i, c in enumerate(unique)}
    credit = [0.0] * len(unique)

    for journey, converted in zip(journeys, conversions, strict=True):
        if not converted or not journey:
            continue
        credit[idx[journey[-1]]] += 1.0

    return HeuristicAttributionResult(
        method="last_click",
        channels=tuple(unique),
        attributions=tuple(credit),
        total_conversions=int(sum(conversions)),
    )


def first_click_attribution(
    journeys: Sequence[Sequence[str]],
    conversions: Sequence[bool],
    *,
    channels: Sequence[str] | None = None,
) -> HeuristicAttributionResult:
    """Credit the first touch of every converting journey."""
    _validate_lengths(journeys, conversions)
    unique = _canonical_channels(journeys, channels)
    idx = {c: i for i, c in enumerate(unique)}
    credit = [0.0] * len(unique)

    for journey, converted in zip(journeys, conversions, strict=True):
        if not converted or not journey:
            continue
        credit[idx[journey[0]]] += 1.0

    return HeuristicAttributionResult(
        method="first_click",
        channels=tuple(unique),
        attributions=tuple(credit),
        total_conversions=int(sum(conversions)),
    )


def linear_attribution(
    journeys: Sequence[Sequence[str]],
    conversions: Sequence[bool],
    *,
    channels: Sequence[str] | None = None,
) -> HeuristicAttributionResult:
    """Split each conversion equally across the unique channels touched."""
    _validate_lengths(journeys, conversions)
    unique = _canonical_channels(journeys, channels)
    idx = {c: i for i, c in enumerate(unique)}
    credit = [0.0] * len(unique)

    for journey, converted in zip(journeys, conversions, strict=True):
        if not converted or not journey:
            continue
        touched = list(dict.fromkeys(journey))  # unique, order-preserving
        share = 1.0 / len(touched)
        for c in touched:
            credit[idx[c]] += share

    return HeuristicAttributionResult(
        method="linear",
        channels=tuple(unique),
        attributions=tuple(credit),
        total_conversions=int(sum(conversions)),
    )


def time_decay_attribution(
    journeys: Sequence[Sequence[str]],
    conversions: Sequence[bool],
    *,
    channels: Sequence[str] | None = None,
    half_life_steps: float = 2.0,
) -> HeuristicAttributionResult:
    """Weight touches by geometric decay from the end of the journey.

    The last touch gets weight 1, the second-to-last weight
    ``0.5 ** (1 / half_life_steps)``, and so on. Weights are renormalised
    so each conversion contributes exactly 1.0 of total credit.
    """
    if half_life_steps <= 0:
        raise ValueError(f"half_life_steps must be positive, got {half_life_steps}")
    _validate_lengths(journeys, conversions)
    unique = _canonical_channels(journeys, channels)
    idx = {c: i for i, c in enumerate(unique)}
    credit = [0.0] * len(unique)

    decay = 0.5 ** (1.0 / half_life_steps)

    for journey, converted in zip(journeys, conversions, strict=True):
        if not converted or not journey:
            continue
        # Collapse to unique, order-preserving so repeated pings on the same
        # channel do not overwhelm the denominator.
        touched = list(dict.fromkeys(journey))
        # Oldest touch earns decay**(k-1); newest earns decay**0 = 1.
        weights = [decay ** (len(touched) - 1 - i) for i in range(len(touched))]
        wsum = sum(weights)
        for c, w in zip(touched, weights, strict=True):
            credit[idx[c]] += w / wsum

    return HeuristicAttributionResult(
        method="time_decay",
        channels=tuple(unique),
        attributions=tuple(credit),
        total_conversions=int(sum(conversions)),
    )
