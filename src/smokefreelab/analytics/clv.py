"""Customer Lifetime Value — RFM + BG/NBD + Gamma-Gamma.

This module is the "who are our best customers worth investing in?" engine.
It pairs three standard models that together cover the full FMCG CLV
workflow:

1. **RFM segmentation** — deterministic, quantile-based Recency/Frequency/
   Monetary scoring that collapses a customer base into ~10 named segments
   (Champions, At Risk, Can't Lose Them, …). Cheap, transparent, and what
   every merchandiser already talks in.
2. **BG/NBD** (Fader, Hardie, Lee 2005) — probabilistic counting model for
   non-contractual repeat purchase. Given a customer's past ``(frequency,
   recency, T)`` tuple it returns their probability of being alive and the
   expected number of transactions in a future horizon.
3. **Gamma-Gamma** (Fader & Hardie 2013) — a companion model for the
   monetary side: expected value per future transaction, conditional on
   being alive. Independent of transaction count by assumption.

Multiplying 2 and 3 (then discounting) gives a per-customer CLV that you
can rank, threshold, and budget against.

Business context
----------------
Cigarettes are the archetypal recurring consumable — daily cadence,
low-but-steady ticket, heavy tail on top-decile loyalty. This is exactly
the regime the BG/NBD + Gamma-Gamma pair was designed for (the original
papers used CDNow music-purchase data with a similar shape). The
FMCG-relevant CLV question is less "who should we acquire" (regulatory
limits on tobacco advertising bite) and more "which registered members of
the age-verified e-commerce channel are worth direct-marketing offers to,
and at what cost cap". That is a ranking problem with a rupiah-denominated
budget constraint — exactly what this module outputs.

Design choices
--------------
- **Pure-python RFM** so the deterministic scoring is testable without
  ``lifetimes``. The BG/NBD and Gamma-Gamma paths are thin wrappers that
  import ``lifetimes`` at call time (same pattern as ``fit_hierarchical``
  in ``elasticity.py``).
- **Quantile-based RFM via ``numpy.quantile``**, not ``pandas.qcut``, so
  we avoid an unnecessary pandas round-trip for the pure-array users.
- **Frozen dataclasses** throughout — results serialize cleanly to JSON
  for Streamlit and notebook renderers.
- **Portfolio summary as a separate dataclass** so the 80/20 rule check
  (top-decile CLV share) is a first-class output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


FloatArray = Union["Sequence[float]", np.ndarray]
"""Array-like of floats — accepts Python sequences or 1-D numpy arrays."""

IntArray = Union["Sequence[int]", np.ndarray]
"""Array-like of ints — accepts Python sequences or 1-D numpy arrays."""

StrArray = Union["Sequence[str]", np.ndarray]
"""Array-like of strings — accepts Python sequences or 1-D numpy arrays."""


@dataclass(frozen=True)
class RFMScore:
    """Per-customer RFM score with canonical segment label.

    Attributes
    ----------
    customer_id : str
        External identifier, echoed back for joining.
    recency : float
        Days since last transaction at the snapshot date. Lower is better.
    frequency : int
        Observed transaction count over the observation window.
    monetary : float
        Average transaction value (not total). Per the Gamma-Gamma
        convention — repeat modelers use the mean, not sum.
    r_score, f_score, m_score : int
        Quantile bin 1..``n_bins``. Higher is better for all three (R is
        inverted internally so "recent" → high).
    rfm_code : str
        Concatenated scores, e.g. ``"555"`` for the top segment. Used as
        a grouping key in dashboards.
    segment : str
        Canonical segment label — one of ``"Champions"``, ``"Loyal
        Customers"``, ``"Potential Loyalists"``, ``"New Customers"``,
        ``"Promising"``, ``"Needs Attention"``, ``"About To Sleep"``,
        ``"At Risk"``, ``"Can't Lose Them"``, ``"Hibernating"``, ``"Lost"``.
    """

    customer_id: str
    recency: float
    frequency: int
    monetary: float
    r_score: int
    f_score: int
    m_score: int
    rfm_code: str
    segment: str


@dataclass(frozen=True)
class CLVEstimate:
    """Per-customer CLV from BG/NBD + Gamma-Gamma, discounted.

    Attributes
    ----------
    customer_id : str
    predicted_purchases : float
        Expected transaction count in the horizon (same units as the
        ``horizon`` argument passed to ``estimate_clv``).
    predicted_avg_value : float
        Expected spend per transaction, from the Gamma-Gamma posterior.
    clv : float
        Discounted value over the horizon, in the same currency unit as
        the input monetary values.
    probability_alive : float
        BG/NBD ``P(alive | frequency, recency, T)``. Useful as a retention-
        marketing targeting score even if the dollar CLV is not used.
    """

    customer_id: str
    predicted_purchases: float
    predicted_avg_value: float
    clv: float
    probability_alive: float


@dataclass(frozen=True)
class CLVSummary:
    """Portfolio-level CLV rollup.

    Attributes
    ----------
    n_customers : int
    total_clv : float
    top_decile_clv : float
        Sum of CLV for the top 10% of customers by CLV.
    top_decile_share : float
        ``top_decile_clv / total_clv``. 0.4-0.6 is the typical FMCG range;
        anything under 0.3 means either the top tail is unusually flat or
        the model has under-shrunk the high-frequency buyers.
    median_clv : float
    """

    n_customers: int
    total_clv: float
    top_decile_clv: float
    top_decile_share: float
    median_clv: float


_SEGMENT_RULES: tuple[tuple[str, int, int, int, int], ...] = (
    # (segment, r_min, r_max, f_min, f_max)
    ("Champions", 4, 5, 4, 5),
    ("Loyal Customers", 2, 5, 4, 5),
    ("Potential Loyalists", 3, 5, 1, 3),
    ("New Customers", 4, 5, 1, 1),
    ("Promising", 3, 4, 1, 1),
    ("Needs Attention", 2, 3, 2, 3),
    ("About To Sleep", 2, 3, 1, 2),
    ("At Risk", 1, 2, 2, 5),
    ("Can't Lose Them", 1, 1, 4, 5),
    ("Hibernating", 1, 2, 1, 2),
    ("Lost", 1, 1, 1, 1),
)
"""Canonical RFM→segment mapping.

The rules are evaluated top-down and the first match wins, so ordering
matters. Source: Blattberg, Kim & Neslin, Database Marketing (2008), with
minor renaming to match the industry-shared vocabulary used by common
CRM platforms.
"""


def _classify_segment(r: int, f: int) -> str:
    """Return the first segment whose (R, F) box contains (r, f)."""
    for name, r_min, r_max, f_min, f_max in _SEGMENT_RULES:
        if r_min <= r <= r_max and f_min <= f <= f_max:
            return name
    return "Unclassified"  # pragma: no cover - defensive, all (r,f) are covered above


def _quantile_score(values: np.ndarray, n_bins: int, *, invert: bool) -> np.ndarray:
    """Bin values into 1..n_bins by quantile, with optional inversion.

    ``invert=True`` flips the mapping so that LOW values get HIGH scores —
    the right convention for recency-days (recent = good).
    """
    # np.quantile with n_bins-1 interior cutpoints yields n_bins bins.
    probs = np.linspace(0, 1, n_bins + 1)[1:-1]
    cutpoints = np.quantile(values, probs)
    # np.searchsorted returns 0..n_bins-1; shift to 1..n_bins.
    bin_idx = np.searchsorted(cutpoints, values, side="right") + 1
    bin_idx = np.clip(bin_idx, 1, n_bins)
    if invert:
        bin_idx = n_bins + 1 - bin_idx
    return bin_idx


def rfm_score(
    customer_id: StrArray,
    recency_days: FloatArray,
    frequency: IntArray,
    monetary: FloatArray,
    *,
    n_bins: int = 5,
) -> tuple[RFMScore, ...]:
    """Quantile-based RFM scoring with canonical segment labels.

    Parameters
    ----------
    customer_id : Sequence[str]
        Customer identifiers. Length ``n``.
    recency_days : Sequence[float]
        Days since last transaction for each customer. Lower is better.
    frequency : Sequence[int]
        Count of transactions over the observation window.
    monetary : Sequence[float]
        Average spend per transaction. Use the *mean*, not the sum, to
        keep the subsequent Gamma-Gamma fit well-calibrated.
    n_bins : int, default 5
        Number of quantile bins per dimension. 5 is industry standard
        (the classic "RFM 555" vocabulary).

    Returns
    -------
    tuple[RFMScore, ...]
        One ``RFMScore`` per customer, in input order.

    Raises
    ------
    ValueError
        If input lengths mismatch, ``n_bins < 2``, or any customer has
        negative recency/monetary.

    Notes
    -----
    When all ``monetary`` values are identical (say, a fixed-price
    subscription), the M quantile degenerates and every customer gets the
    same M-score. This is the correct behavior and callers who want the
    R/F-only segmentation can simply ignore M.
    """
    r = np.asarray(recency_days, dtype=float)
    f = np.asarray(frequency, dtype=float)
    m = np.asarray(monetary, dtype=float)
    ids = list(customer_id)

    if not (len(r) == len(f) == len(m) == len(ids)):
        raise ValueError(
            "customer_id, recency_days, frequency, monetary must share length; "
            f"got {len(ids)}/{len(r)}/{len(f)}/{len(m)}"
        )
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if np.any(r < 0) or np.any(m < 0) or np.any(f < 0):
        raise ValueError("recency, frequency, monetary must all be non-negative")

    r_scores = _quantile_score(r, n_bins, invert=True)
    f_scores = _quantile_score(f, n_bins, invert=False)
    m_scores = _quantile_score(m, n_bins, invert=False)

    results: list[RFMScore] = []
    for i, cid in enumerate(ids):
        r_i = int(r_scores[i])
        f_i = int(f_scores[i])
        m_i = int(m_scores[i])
        results.append(
            RFMScore(
                customer_id=str(cid),
                recency=float(r[i]),
                frequency=int(f[i]),
                monetary=float(m[i]),
                r_score=r_i,
                f_score=f_i,
                m_score=m_i,
                rfm_code=f"{r_i}{f_i}{m_i}",
                segment=_classify_segment(r_i, f_i),
            )
        )
    return tuple(results)


def summarize_clv(estimates: Sequence[CLVEstimate]) -> CLVSummary:
    """Aggregate per-customer CLV into a portfolio summary.

    Parameters
    ----------
    estimates : Sequence[CLVEstimate]

    Returns
    -------
    CLVSummary

    Raises
    ------
    ValueError
        If ``estimates`` is empty.
    """
    if not estimates:
        raise ValueError("estimates must be non-empty")
    clvs = np.array([e.clv for e in estimates], dtype=float)
    n = len(clvs)
    total = float(clvs.sum())
    top_n = max(1, n // 10)
    top_clv = float(np.sort(clvs)[-top_n:].sum())
    share = top_clv / total if total > 0 else 0.0
    return CLVSummary(
        n_customers=n,
        total_clv=total,
        top_decile_clv=top_clv,
        top_decile_share=share,
        median_clv=float(np.median(clvs)),
    )


def estimate_clv(
    customer_id: StrArray,
    frequency: IntArray,
    recency: FloatArray,
    observation_period: FloatArray,
    monetary_value: FloatArray,
    *,
    horizon_periods: float = 12.0,
    discount_rate: float = 0.01,
    bg_penalizer: float = 0.0,
    gg_penalizer: float = 0.0,
) -> tuple[CLVEstimate, ...]:
    """Per-customer CLV via BG/NBD + Gamma-Gamma, discounted.

    Parameters
    ----------
    customer_id : Sequence[str]
        Identifiers, length ``n``.
    frequency : Sequence[int]
        **Repeat** transaction count (i.e. total transactions minus 1).
        The BG/NBD convention — a customer whose only purchase is the
        acquisition has ``frequency = 0``.
    recency : Sequence[float]
        Time between first and last purchase, in the same units as
        ``observation_period`` and ``horizon_periods``.
    observation_period : Sequence[float]
        Time from first purchase to the observation snapshot, per customer.
    monetary_value : Sequence[float]
        Average spend per *repeat* transaction for customers with
        ``frequency >= 1``. For ``frequency = 0`` customers, the Gamma-
        Gamma model is undefined — pass any non-negative placeholder; it
        won't be used (predicted_avg_value falls back to the portfolio
        mean of repeat monetary values).
    horizon_periods : float, default 12.0
        Forecast horizon in the same time units as recency/T. A month-
        indexed fit with ``horizon_periods=12`` gives 12-month CLV.
    discount_rate : float, default 0.01
        Per-period discount rate. ``0.01`` monthly ≈ ~12.7% annual.
    bg_penalizer, gg_penalizer : float
        L2 penalty on the lifetimes MLE; useful for stabilising small
        panels. Default 0 (no penalty) matches standard BG/NBD.

    Returns
    -------
    tuple[CLVEstimate, ...]
        One ``CLVEstimate`` per customer, in input order.

    Raises
    ------
    ImportError
        If ``lifetimes`` is not installed.
    ValueError
        If inputs mismatch in length or have negative values.
    """
    try:
        from lifetimes import BetaGeoFitter, GammaGammaFitter
    except ImportError as exc:  # pragma: no cover - env guard
        raise ImportError("estimate_clv requires `lifetimes`. Install with `uv sync`.") from exc

    ids = list(customer_id)
    freq = np.asarray(frequency, dtype=float)
    rec = np.asarray(recency, dtype=float)
    obs = np.asarray(observation_period, dtype=float)
    mon = np.asarray(monetary_value, dtype=float)

    if not (len(ids) == len(freq) == len(rec) == len(obs) == len(mon)):
        raise ValueError("all input sequences must have the same length")
    if np.any(freq < 0) or np.any(rec < 0) or np.any(obs < 0) or np.any(mon < 0):
        raise ValueError("frequency, recency, observation_period, monetary must be non-negative")

    bgf = BetaGeoFitter(penalizer_coef=bg_penalizer)
    bgf.fit(freq, rec, obs)

    predicted_purchases = np.asarray(
        bgf.conditional_expected_number_of_purchases_up_to_time(horizon_periods, freq, rec, obs),
        dtype=float,
    )
    prob_alive = np.asarray(bgf.conditional_probability_alive(freq, rec, obs), dtype=float)

    repeat_mask = freq >= 1
    if repeat_mask.sum() >= 2:
        ggf = GammaGammaFitter(penalizer_coef=gg_penalizer)
        ggf.fit(freq[repeat_mask], mon[repeat_mask])
        predicted_avg = np.asarray(ggf.conditional_expected_average_profit(freq, mon), dtype=float)
    else:
        # Not enough repeat data to fit Gamma-Gamma — fall back to simple
        # portfolio mean of the non-zero monetary values.
        fallback_mean = float(mon[repeat_mask].mean()) if repeat_mask.any() else 0.0
        predicted_avg = np.full_like(predicted_purchases, fallback_mean)

    # Simple deterministic discount: sum_{t=1..H} v / (1 + r)^t approximated
    # by the closed-form geometric series, then scaled by the fraction of
    # predicted purchases that fall in [1, horizon].
    if discount_rate > 0:
        discount_factor = (1 - (1 + discount_rate) ** -horizon_periods) / discount_rate
        discount_factor /= horizon_periods  # average discount over horizon
    else:
        discount_factor = 1.0
    clv = predicted_purchases * predicted_avg * discount_factor

    return tuple(
        CLVEstimate(
            customer_id=str(ids[i]),
            predicted_purchases=float(predicted_purchases[i]),
            predicted_avg_value=float(predicted_avg[i]),
            clv=float(clv[i]),
            probability_alive=float(prob_alive[i]),
        )
        for i in range(len(ids))
    )
