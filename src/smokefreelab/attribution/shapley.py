r"""Shapley-value multi-touch attribution.

Game-theoretic attribution: treat each channel as a player in a coalitional
game whose value ``v(S)`` is the conversion count (or rate) observed among
users whose touch-set is a subset of ``S``. The Shapley value of channel
``c`` averages the marginal contribution of ``c`` across every ordering of
the grand coalition:

    phi_c  =  sum_{S subset N \ {c}}  (|S|! * (n - |S| - 1)! / n!)  *  (v(S u {c}) - v(S))

Shapley's axioms (efficiency, symmetry, null-player, additivity) are the
cleanest game-theoretic justification for a single attribution scheme. The
industry paper that popularised this for marketing is Dalessandro,
Perlich, Stitelman & Provost (2012), "Causally motivated attribution for
online advertising."

Design choices
--------------
- **Exact enumeration**, bounded by ``max_channels_exact``. The classical
  Shapley formula runs in ``2^n`` subset evaluations. For ``n ≤ 12`` that
  is 4096 subsets — fast. Above that we refuse rather than silently
  swapping to Monte Carlo (which belongs in a follow-up).
- **Order-insensitive** by design. The value function is defined on the
  *set* of channels a user was exposed to, dropping sequence information.
  That is a deliberate modelling choice: for sequence, use Markov.
- **``v(S)`` = conversion count of users whose touch-set ⊆ S**, following
  Dalessandro et al. This gives a monotone, super-additive game and makes
  the interpretation "contribution to observed conversions" direct.
- **No pandas**. Same rationale as the Markov module.

Business context
----------------
The two data-driven attribution families answer different questions:

- **Shapley** — "if we were designing the media mix from scratch, how
  should we *budget* across channels?" It is a static, coalitional view.
- **Markov** — "given the sequential funnel, which channel's *removal*
  would hurt conversion most?" It is a dynamic, graph-walk view.

In a media-plan review you'd quote both. They usually agree on the top-3
ranking; when they diverge, the channel that Markov favours is typically a
mid-funnel *path builder* and the channel Shapley favours is a standalone
*closer*. That divergence itself is the insight worth showing.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain, combinations
from math import factorial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass(frozen=True)
class ShapleyAttributionResult:
    """Channel-level credit from an exact coalitional Shapley model.

    Attributes
    ----------
    channels : tuple[str, ...]
        Channel names, aligned with ``attributions`` and ``shapley_values``.
    attributions : tuple[float, ...]
        Fractional conversion credit per channel. Sums to
        ``total_conversions`` (up to floating-point error); by Shapley's
        efficiency axiom this is exact in the ``v(S) = conversion count``
        formulation.
    shapley_values : tuple[float, ...]
        Raw Shapley values on the "conversion count" game. For this choice
        of ``v`` the raw values equal ``attributions``; we keep both fields
        so the dataclass layout mirrors ``MarkovAttributionResult``.
    total_conversions : int
        Observed conversions being distributed across channels.
    """

    channels: tuple[str, ...]
    attributions: tuple[float, ...]
    shapley_values: tuple[float, ...]
    total_conversions: int

    @property
    def shares(self) -> tuple[float, ...]:
        """Fractional share per channel."""
        total = float(self.total_conversions)
        if total == 0:
            return tuple(0.0 for _ in self.channels)
        return tuple(a / total for a in self.attributions)


def _powerset(iterable: Iterable[str]) -> Iterable[tuple[str, ...]]:
    """All subsets of ``iterable``, including the empty set and the full set."""
    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(len(items) + 1))


def shapley_attribution(
    journeys: Sequence[Sequence[str]],
    conversions: Sequence[bool],
    *,
    channels: Sequence[str] | None = None,
    max_channels_exact: int = 12,
) -> ShapleyAttributionResult:
    """Attribute conversions to channels via the exact Shapley value.

    Parameters
    ----------
    journeys : Sequence[Sequence[str]]
        One touch sequence per user. Only the *set* of channels touched is
        used — sequence and frequency are ignored by construction.
    conversions : Sequence[bool]
        Same length as ``journeys``.
    channels : Sequence[str], optional
        Explicit channel ordering. If ``None``, uses sorted unique touches.
    max_channels_exact : int, default 12
        Hard cap on the number of distinct channels. Above this the exact
        enumeration (``2^n``) becomes slow enough that a Monte Carlo
        approximation would be the right choice; this module refuses
        rather than silently degrading.

    Returns
    -------
    ShapleyAttributionResult
        Frozen dataclass whose ``attributions`` sum to ``total_conversions``.

    Raises
    ------
    ValueError
        If the input lengths mismatch, the channel universe exceeds
        ``max_channels_exact``, or ``journeys`` contains a channel absent
        from ``channels``.

    Notes
    -----
    The value function is ``v(S) = | { i : set(journeys[i]) ⊆ S and
    conversions[i] } |`` — the observed conversion count from users whose
    full exposure set is contained in ``S``. This is monotone and gives a
    super-additive game; Shapley's efficiency axiom then guarantees that
    the per-channel values sum exactly to ``v(N)`` which is the total
    number of converters.
    """
    if len(journeys) != len(conversions):
        raise ValueError(
            f"journeys and conversions must have the same length, "
            f"got {len(journeys)} vs {len(conversions)}"
        )

    if channels is None:
        unique = sorted({c for j in journeys for c in j})
    else:
        unique = list(channels)
        observed = {c for j in journeys for c in j}
        missing = observed - set(unique)
        if missing:
            raise ValueError(f"journeys contain channels not in `channels`: {sorted(missing)}")

    n = len(unique)
    if n > max_channels_exact:
        raise ValueError(
            f"Exact Shapley requires 2^n subset evaluations. Got n={n} "
            f"channels which exceeds max_channels_exact={max_channels_exact}. "
            f"Aggregate channels or raise the cap explicitly."
        )

    total_conversions = int(sum(conversions))

    if n == 0 or total_conversions == 0:
        zero = tuple(0.0 for _ in unique)
        return ShapleyAttributionResult(
            channels=tuple(unique),
            attributions=zero,
            shapley_values=zero,
            total_conversions=total_conversions,
        )

    # Precompute each converter's touch-set as a frozenset for fast subset
    # checks. Non-converters contribute 0 to v, so they can be dropped.
    converter_sets: list[frozenset[str]] = [
        frozenset(j) for j, c in zip(journeys, conversions, strict=True) if c
    ]

    channel_universe = frozenset(unique)

    def v(coalition: frozenset[str]) -> int:
        """Conversion count from users whose touch-set is ⊆ coalition."""
        return sum(1 for s in converter_sets if s <= coalition)

    # Cache v(·) across all 2^n subsets of the channel universe. This makes
    # the Shapley loop below O(n * 2^n) value lookups without re-scanning
    # journeys for every evaluation.
    v_cache: dict[frozenset[str], int] = {}
    for subset in _powerset(unique):
        v_cache[frozenset(subset)] = v(frozenset(subset))

    shapley_vals = [0.0] * n
    n_fact = factorial(n)

    for i, channel in enumerate(unique):
        others = channel_universe - {channel}
        acc = 0.0
        for subset in _powerset(others):
            s = frozenset(subset)
            s_size = len(s)
            weight = factorial(s_size) * factorial(n - s_size - 1) / n_fact
            marginal = v_cache[s | {channel}] - v_cache[s]
            acc += weight * marginal
        shapley_vals[i] = acc

    # Efficiency axiom makes this exact in theory; we still normalise to
    # guard against subtle float drift on large channel sets.
    raw_sum = sum(shapley_vals)
    if raw_sum > 0:
        attributions = tuple(v * total_conversions / raw_sum for v in shapley_vals)
    else:
        attributions = tuple(0.0 for _ in unique)

    return ShapleyAttributionResult(
        channels=tuple(unique),
        attributions=attributions,
        shapley_values=tuple(shapley_vals),
        total_conversions=total_conversions,
    )
