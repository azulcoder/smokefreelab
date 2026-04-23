"""Multi-touch attribution — heuristic, Markov, Shapley, and MMM."""

from smokefreelab.attribution.heuristics import (
    HeuristicAttributionResult,
    first_click_attribution,
    last_click_attribution,
    linear_attribution,
    time_decay_attribution,
)
from smokefreelab.attribution.markov import (
    MarkovAttributionResult,
    markov_attribution,
)
from smokefreelab.attribution.mmm import (
    MMMChannelResult,
    MMMResult,
    apply_adstock,
    apply_hill,
    fit_mmm,
    response_curve,
)
from smokefreelab.attribution.shapley import (
    ShapleyAttributionResult,
    shapley_attribution,
)

__all__ = [
    "HeuristicAttributionResult",
    "MMMChannelResult",
    "MMMResult",
    "MarkovAttributionResult",
    "ShapleyAttributionResult",
    "apply_adstock",
    "apply_hill",
    "first_click_attribution",
    "fit_mmm",
    "last_click_attribution",
    "linear_attribution",
    "markov_attribution",
    "response_curve",
    "shapley_attribution",
    "time_decay_attribution",
]
