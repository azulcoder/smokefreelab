"""Experiment design and analysis — frequentist + Bayesian A/B engine."""

from smokefreelab.experiment.ab_test import (
    ArmStats,
    BayesianResult,
    FrequentistResult,
    PowerResult,
    SRMResult,
    bayesian_test,
    check_srm,
    experiment_duration_days,
    frequentist_test,
    sample_size_per_arm,
    simulate_peeking_inflation,
)

__all__ = [
    "ArmStats",
    "BayesianResult",
    "FrequentistResult",
    "PowerResult",
    "SRMResult",
    "bayesian_test",
    "check_srm",
    "experiment_duration_days",
    "frequentist_test",
    "sample_size_per_arm",
    "simulate_peeking_inflation",
]
