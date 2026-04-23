"""Unit tests for the XGBoost propensity pipeline.

Covers:

- Happy-path training on a synthetic signal-plus-noise dataset with an
  AUC lower bound that is strict enough to catch regressions but loose
  enough to tolerate RNG jitter.
- Shape, length, and label-domain validation.
- Reproducibility under a fixed seed.
- SHAP output shape and ``ranked_features`` ordering.
- Calibration curve length and value range.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

# XGBoost on macOS needs libomp (``brew install libomp``). Skip the module
# if the native library can't load — CI installs libomp so the suite still
# runs end-to-end there. ``import xgboost`` raises ``XGBoostError`` rather
# than ``ImportError`` on a missing shared library, so we catch broadly.
try:
    import xgboost as _xgb  # noqa: F401
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(
        f"xgboost unavailable in this environment: {exc}",
        allow_module_level=True,
    )

from smokefreelab.features import (
    PropensityModelResult,
    train_propensity_model,
)


def _signal_dataset(
    n_samples: int = 600,
    n_features: int = 6,
    n_informative: int = 3,
    random_state: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic binary classification with a fixed seed."""
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        class_sep=1.5,
        flip_y=0.05,
        random_state=random_state,
    )
    return x, y


class TestTrainPropensityModel:
    """End-to-end training returns a well-formed result."""

    def test_returns_propensity_result(self) -> None:
        """``train_propensity_model`` returns the expected dataclass."""
        x, y = _signal_dataset()
        result = train_propensity_model(x, y)
        assert isinstance(result, PropensityModelResult)

    def test_auc_beats_baseline_on_signal(self) -> None:
        """A 3-informative-feature dataset should clear AUC ≥ 0.80."""
        x, y = _signal_dataset()
        result = train_propensity_model(x, y)
        assert result.test_auc >= 0.80

    def test_brier_is_finite_and_in_range(self) -> None:
        """Brier score lives in ``[0, 1]`` and is finite."""
        x, y = _signal_dataset()
        result = train_propensity_model(x, y)
        assert 0.0 <= result.test_brier <= 1.0
        assert np.isfinite(result.test_brier)
        assert np.isfinite(result.test_log_loss)

    def test_feature_importances_align_with_names(self) -> None:
        """Gain and SHAP importance tuples match feature_names length."""
        x, y = _signal_dataset()
        names = [f"feat_{i}" for i in range(x.shape[1])]
        result = train_propensity_model(x, y, feature_names=names)
        assert result.feature_names == tuple(names)
        assert len(result.feature_importances) == len(names)
        assert len(result.shap_mean_abs) == len(names)

    def test_default_feature_names_match_column_count(self) -> None:
        """When ``feature_names`` is omitted, default to ``f0, f1, ...``."""
        x, y = _signal_dataset(n_features=4)
        result = train_propensity_model(x, y)
        assert result.feature_names == ("f0", "f1", "f2", "f3")

    def test_ranked_features_are_sorted_descending(self) -> None:
        """``ranked_features`` is mean-abs-SHAP sorted, largest first."""
        x, y = _signal_dataset()
        result = train_propensity_model(x, y)
        importances = [imp for _, imp in result.ranked_features]
        assert importances == sorted(importances, reverse=True)

    def test_calibration_curve_shape(self) -> None:
        """Calibration outputs share the same length and live in [0, 1]."""
        x, y = _signal_dataset()
        result = train_propensity_model(x, y, calibration_bins=10)
        assert len(result.calibration_mean_predicted) == len(result.calibration_mean_observed)
        for v in result.calibration_mean_predicted:
            assert 0.0 <= v <= 1.0
        for v in result.calibration_mean_observed:
            assert 0.0 <= v <= 1.0

    def test_split_counts_sum_to_input(self) -> None:
        """Train + test row counts equal the input row count."""
        x, y = _signal_dataset(n_samples=400)
        result = train_propensity_model(x, y, test_size=0.25)
        assert result.n_train + result.n_test == 400
        assert result.n_test == 100

    def test_seed_reproducibility(self) -> None:
        """Two runs with identical ``random_state`` yield identical AUC."""
        x, y = _signal_dataset()
        a = train_propensity_model(x, y, random_state=123)
        b = train_propensity_model(x, y, random_state=123)
        assert a.test_auc == pytest.approx(b.test_auc)
        assert a.test_brier == pytest.approx(b.test_brier)

    def test_model_can_predict_new_rows(self) -> None:
        """The fitted booster inside the result supports ``predict_proba``."""
        x, y = _signal_dataset()
        result = train_propensity_model(x, y)
        preds = result.model.predict_proba(x[:5])
        assert preds.shape == (5, 2)
        # Each row is a distribution.
        assert np.allclose(preds.sum(axis=1), 1.0)


class TestValidation:
    """Constructor-level input validation."""

    def test_1d_features_rejected(self) -> None:
        """A flat 1-d feature array is a shape error."""
        with pytest.raises(ValueError, match="features must be 2-d"):
            train_propensity_model(np.array([1.0, 2.0, 3.0]), [0, 1, 0])

    def test_length_mismatch_rejected(self) -> None:
        """Row-count mismatch is a shape error."""
        x = np.zeros((10, 3))
        y = np.zeros(9, dtype=int)
        with pytest.raises(ValueError, match="same length"):
            train_propensity_model(x, y)

    def test_non_binary_labels_rejected(self) -> None:
        """Multiclass labels must be rejected — this is a binary model."""
        x, _ = _signal_dataset(n_samples=30)
        y = np.array([0, 1, 2] * 10)
        with pytest.raises(ValueError, match="binary 0/1"):
            train_propensity_model(x, y)

    def test_single_class_labels_rejected(self) -> None:
        """All-zero labels have no learnable signal."""
        x, _ = _signal_dataset(n_samples=30)
        y = np.zeros(30, dtype=int)
        with pytest.raises(ValueError, match="both classes"):
            train_propensity_model(x, y)

    def test_feature_names_wrong_length_rejected(self) -> None:
        """``feature_names`` must match the column count."""
        x, y = _signal_dataset(n_features=4)
        with pytest.raises(ValueError, match="feature_names length"):
            train_propensity_model(x, y, feature_names=["only", "three", "names"])
