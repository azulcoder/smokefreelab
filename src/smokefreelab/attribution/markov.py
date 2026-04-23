"""Markov-chain multi-touch attribution via removal effect.

This module implements the data-driven attribution method of Anderl, Becker,
von Wangenheim & Schumann (2016), "Mapping the customer journey: Lessons
learned from graph-based attribution modelling." The core idea is simple
once you see the matrix:

1. Treat every customer journey as a walk through a directed graph whose
   nodes are the marketing channels plus two absorbing states,
   ``Converted`` and ``Dropped``, and a starting node ``Start``.
2. Estimate transition probabilities empirically from the observed paths.
3. Compute the fraction of walks from ``Start`` that end at ``Converted`` —
   that's the base conversion rate explained by the full network.
4. For each channel ``c``, recompute step 3 after redirecting every walk
   that would have passed through ``c`` straight to ``Dropped``. The drop
   in conversion probability is the *removal effect* of ``c``.
5. Normalise removal effects across channels and multiply by total observed
   conversions. Each channel now owns a fractional credit that (by
   construction) sums to the actual conversion count.

Business context
----------------
Last-click attribution systematically over-credits the final touch; first-
click over-credits awareness; linear and time-decay just split uniformly or
geometrically. Markov attribution is the minimum viable upgrade that
attributes credit to the *sequential structure* of the funnel — which is
what an FMCG brand is paying for when it buys awareness on broadcast to
feed retail trial six weeks later.

Design choices
--------------
- **Exact linear algebra**, not simulation. For the sizes one sees in
  portfolio decks (k < 30 channels) the (I - Q)^-1 inversion is fast and
  deterministic. Simulation only helps when k is in the hundreds.
- **Removal via redirect to Dropped**, not node deletion, so the matrix
  dimensions stay fixed across variants and the code is branch-free.
- **Channel order is preserved**, so results line up with a user-supplied
  ``channels`` argument for joining to business metadata.
- **No pandas**. The core takes lists-of-lists so it is trivially testable
  and imports cleanly into notebooks, apps, and pipelines alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class MarkovAttributionResult:
    """Channel-level credit from a Markov removal-effect model.

    Attributes
    ----------
    channels : tuple[str, ...]
        Channel names in the order the rest of the fields are aligned to.
    attributions : tuple[float, ...]
        Fractional conversion credit per channel. Sums to
        ``total_conversions`` (up to floating-point error).
    removal_effects : tuple[float, ...]
        Relative drop in conversion probability when the channel is removed
        from the transition graph, in ``[0, 1]``. These are the un-
        normalised per-channel weights; ``attributions`` is their
        normalised scale-up to the observed conversion count.
    conversion_probability : float
        Baseline ``P(Converted | Start)`` from the full transition graph.
    total_conversions : int
        Observed conversions that are being distributed across channels.
    """

    channels: tuple[str, ...]
    attributions: tuple[float, ...]
    removal_effects: tuple[float, ...]
    conversion_probability: float
    total_conversions: int

    @property
    def shares(self) -> tuple[float, ...]:
        """Fractional share per channel (``attributions`` divided by total)."""
        total = float(self.total_conversions)
        if total == 0:
            return tuple(0.0 for _ in self.channels)
        return tuple(a / total for a in self.attributions)


def _build_transition_matrix(
    journeys: Sequence[Sequence[str]],
    conversions: Sequence[bool],
    channels: Sequence[str],
) -> np.ndarray:
    """Estimate a row-stochastic transition matrix from observed journeys.

    Row and column layout (size ``k + 3``):

    ``[Start, channel_0, channel_1, ..., channel_{k-1}, Converted, Dropped]``

    where indexing matches the order of ``channels``. ``Converted`` and
    ``Dropped`` are absorbing: their self-transition is 1.
    """
    k = len(channels)
    channel_idx = {c: i + 1 for i, c in enumerate(channels)}
    start_idx = 0
    converted_idx = k + 1
    dropped_idx = k + 2
    size = k + 3

    counts = np.zeros((size, size), dtype=np.float64)

    for journey, converted in zip(journeys, conversions, strict=True):
        if not journey:
            # A journey with no touches cannot carry credit.
            continue
        prev = start_idx
        for touch in journey:
            cur = channel_idx[touch]
            counts[prev, cur] += 1.0
            prev = cur
        counts[prev, converted_idx if converted else dropped_idx] += 1.0

    # Absorbing self-loops so the matrix is row-stochastic.
    counts[converted_idx, converted_idx] = 1.0
    counts[dropped_idx, dropped_idx] = 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    # Transient rows with no outgoing observations: send straight to Dropped
    # so the chain still absorbs and the inversion below is well-posed.
    zero_rows = row_sums.squeeze() == 0.0
    if np.any(zero_rows):
        counts[zero_rows, dropped_idx] = 1.0
        row_sums = counts.sum(axis=1, keepdims=True)

    transition: np.ndarray = counts / row_sums
    return transition


def _absorption_probability(transition: np.ndarray) -> float:
    """Return ``P(absorbed in Converted | started at Start)`` for the chain.

    The last two rows/cols are the absorbing ``Converted`` and ``Dropped``
    states; everything above them is transient. We solve
    ``N = (I - Q)^{-1}`` and read ``P(Converted | Start)`` off row 0 of
    ``B = N @ R``.
    """
    size = transition.shape[0]
    t = size - 2
    q = transition[:t, :t]
    r = transition[:t, t:]
    n = np.linalg.inv(np.eye(t) - q)
    absorption = n @ r
    return float(absorption[0, 0])


def _remove_channel(transition: np.ndarray, channel_index_in_matrix: int) -> np.ndarray:
    """Return a copy of ``transition`` with the channel redirected to Dropped.

    Physically: any walk that would step into the removed channel instead
    steps into ``Dropped`` and is done. We implement this by rewriting the
    channel's row (all self-mass goes to Dropped) and zeroing its inbound
    column, with the freed inbound mass also redirected to Dropped.
    """
    size = transition.shape[0]
    dropped_idx = size - 1
    t = transition.copy()

    # Inbound mass to the removed channel → Dropped.
    inbound = t[:, channel_index_in_matrix].copy()
    t[:, channel_index_in_matrix] = 0.0
    t[:, dropped_idx] += inbound

    # Outbound from the removed channel is academic (it's no longer
    # reachable), but make it absorbing-to-Dropped to keep the row sums at 1
    # and the inversion well-conditioned.
    t[channel_index_in_matrix, :] = 0.0
    t[channel_index_in_matrix, dropped_idx] = 1.0

    return t


def markov_attribution(
    journeys: Sequence[Sequence[str]],
    conversions: Sequence[bool],
    *,
    channels: Sequence[str] | None = None,
) -> MarkovAttributionResult:
    """Attribute conversions to channels via the removal-effect Markov model.

    Parameters
    ----------
    journeys : Sequence[Sequence[str]]
        One ordered touch sequence per user. ``journeys[i]`` is the list of
        channel names the user was exposed to, in order.
    conversions : Sequence[bool]
        Same length as ``journeys``. ``conversions[i]`` is ``True`` when
        the user converted.
    channels : Sequence[str], optional
        Explicit channel ordering. If ``None``, uses the set of channels
        observed in ``journeys``, sorted alphabetically for determinism.

    Returns
    -------
    MarkovAttributionResult
        A frozen dataclass with per-channel credit that sums to the observed
        conversion count.

    Raises
    ------
    ValueError
        If ``journeys`` and ``conversions`` have different lengths, or if
        ``journeys`` contains a channel absent from ``channels``.

    Notes
    -----
    The method assumes the transition matrix estimated from the observed
    journeys is a good approximation of the underlying customer process. In
    practice this means you need enough journeys per channel for the
    transition probabilities to stabilise — a rough heuristic is ≥500
    journeys per channel pair. Below that, credit estimates are noisy but
    still strictly better than heuristic rules (Anderl et al. 2016, §4).
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

    # Empty-journey converters can't be attributed to any channel: their
    # conversion was driven by an unobserved path (organic/brand/direct), so
    # they're excluded from the Markov total per Anderl et al. (2016).
    attributable_conversions = sum(1 for j, c in zip(journeys, conversions, strict=True) if c and j)

    if not unique:
        return MarkovAttributionResult(
            channels=(),
            attributions=(),
            removal_effects=(),
            conversion_probability=0.0,
            total_conversions=attributable_conversions,
        )

    full_matrix = _build_transition_matrix(journeys, conversions, unique)
    p_full = _absorption_probability(full_matrix)

    total_conversions = attributable_conversions

    if p_full <= 0.0 or total_conversions == 0:
        zero = tuple(0.0 for _ in unique)
        return MarkovAttributionResult(
            channels=tuple(unique),
            attributions=zero,
            removal_effects=zero,
            conversion_probability=p_full,
            total_conversions=total_conversions,
        )

    removal_effects: list[float] = []
    for i, _ in enumerate(unique):
        removed = _remove_channel(full_matrix, i + 1)
        p_removed = _absorption_probability(removed)
        effect = 1.0 - (p_removed / p_full) if p_full > 0 else 0.0
        # Clip tiny negative effects that can arise from numerical error.
        removal_effects.append(max(0.0, effect))

    total_effect = sum(removal_effects)
    if total_effect == 0.0:
        shares = [0.0 for _ in unique]
    else:
        shares = [e / total_effect for e in removal_effects]

    attributions = [s * total_conversions for s in shares]

    return MarkovAttributionResult(
        channels=tuple(unique),
        attributions=tuple(attributions),
        removal_effects=tuple(removal_effects),
        conversion_probability=p_full,
        total_conversions=total_conversions,
    )
