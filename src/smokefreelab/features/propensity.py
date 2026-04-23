"""XGBoost purchase-propensity modelling with SHAP explanations.

This module operationalises the classic "who is most likely to buy next?"
question for FMCG commercial teams. In a typical deployment the feature
vector would be RFM + demographic + touchpoint aggregates; here we expose
a pipeline where any feature matrix produces a fitted model, AUC/Brier/
log-loss diagnostics, a reliability curve, and SHAP-based feature
attributions in a single call.

Business context
----------------
A well-calibrated propensity model is the lever behind every targeted
trade-promotion campaign: rather than blanket the full audience we focus
the top decile where marginal CAC is 3-5x lower. Two properties matter
in production:

- **Ranking quality (AUC)** — the top-decile lift is roughly
  ``2 * (AUC - 0.5) / (top-decile base rate)``. For FMCG scoring a
  well-targeted top decile typically lifts 2-4x over random.
- **Calibration** — not just ranking. If the model says ``P = 0.7`` we
  want to observe ~70% of those users actually convert, because promo
  budgets are allocated in *absolute* rupiah from calibrated probabilities.
  An uncalibrated scorer can rank correctly yet misallocate budget.

Design choices
--------------
- **XGBoost histogram trees** as the default learner. CatBoost would also
  work; we pick XGBoost because the SHAP integration via ``TreeExplainer``
  is exact and polynomial in tree depth, not sampling-based.
- **SHAP mean |value| per feature** as the importance summary. Gain-based
  importance has a well-known bias toward high-cardinality features;
  SHAP mean-abs is the corrected metric (Lundberg & Lee 2017; Lundberg
  et al. 2020). We report both so the analyst can compare.
- **Calibration measured, not forced**. We compute the reliability curve
  on the test set so the analyst can see miscalibration, but we do not
  silently wrap the model in isotonic regression — that choice belongs
  with the caller who knows whether probabilities or ranks are the lever.
- **Frozen result dataclass**, mirroring ``ab_test`` and ``attribution``
  modules so notebooks can stack outputs into a single comparison frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import shap
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class PropensityModelResult:
    """Fitted propensity model plus diagnostics and SHAP explanations.

    Attributes
    ----------
    feature_names : tuple[str, ...]
        Names of input features in model column order.
    test_auc : float
        ROC-AUC on the held-out test set. > 0.75 is typically the bar for
        a production propensity scorer in FMCG.
    test_brier : float
        Brier score on the test set (lower is better; 0 is perfect, 0.25
        is the baseline for a 50/50 class balance).
    test_log_loss : float
        Binary cross-entropy on the test set.
    feature_importances : tuple[float, ...]
        XGBoost gain-based importance. Kept for comparison with SHAP;
        biased toward high-cardinality features.
    shap_mean_abs : tuple[float, ...]
        Mean absolute SHAP value per feature on the test set. The
        interpretable alternative to gain-based importance.
    calibration_mean_predicted : tuple[float, ...]
        Bin-wise mean of predicted probability — x-axis of a reliability
        diagram.
    calibration_mean_observed : tuple[float, ...]
        Bin-wise observed conversion rate — y-axis. Perfect calibration
        traces the diagonal.
    n_train, n_test : int
        Row counts per split.
    n_positive_train, n_positive_test : int
        Positive-class counts per split.
    random_state : int
        Seed used for split + booster init.
    model : xgb.XGBClassifier
        Fitted booster. Held in the dataclass so downstream code can
        call ``result.model.predict_proba(new_X)`` without re-training.
    """

    feature_names: tuple[str, ...]
    test_auc: float
    test_brier: float
    test_log_loss: float
    feature_importances: tuple[float, ...]
    shap_mean_abs: tuple[float, ...]
    calibration_mean_predicted: tuple[float, ...]
    calibration_mean_observed: tuple[float, ...]
    n_train: int
    n_test: int
    n_positive_train: int
    n_positive_test: int
    random_state: int
    model: xgb.XGBClassifier

    @property
    def ranked_features(self) -> tuple[tuple[str, float], ...]:
        """Features ranked by mean ``|SHAP|`` value, largest first."""
        pairs = list(zip(self.feature_names, self.shap_mean_abs, strict=True))
        pairs.sort(key=lambda p: p[1], reverse=True)
        return tuple(pairs)


def train_propensity_model(
    features: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    test_size: float = 0.25,
    random_state: int = 42,
    max_depth: int = 4,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    calibration_bins: int = 10,
) -> PropensityModelResult:
    """Train an XGBoost propensity model and return diagnostics + SHAP.

    Parameters
    ----------
    features : np.ndarray of shape (n_samples, n_features)
        Feature matrix. Categorical variables must be encoded upstream;
        XGBoost's histogram tree handles continuous input directly.
    labels : Sequence[int] or np.ndarray of shape (n_samples,)
        Binary labels in ``{0, 1}``.
    feature_names : Sequence[str], optional
        Column names aligned with ``features``. If ``None``, defaults to
        ``f0, f1, ...``.
    test_size : float, default 0.25
        Fraction of rows held out for evaluation.
    random_state : int, default 42
        Seeded split and booster init for reproducibility.
    max_depth : int, default 4
        Max tree depth. Shallow trees help SHAP stability and
        generalisation on small portfolio datasets.
    n_estimators : int, default 200
        Number of boosting rounds. With ``learning_rate=0.1`` and
        ``max_depth=4`` this is a deliberately conservative setting.
    learning_rate : float, default 0.1
    calibration_bins : int, default 10
        Bins for the reliability-curve computation (quantile strategy).

    Returns
    -------
    PropensityModelResult
        Fitted booster, per-split counts, and all diagnostics.

    Raises
    ------
    ValueError
        If ``features`` is not 2-d, row counts differ, labels are not
        binary or contain only one class, or if ``feature_names`` length
        does not match the feature count.

    Notes
    -----
    We deliberately do not wrap the model in
    ``CalibratedClassifierCV``. XGBoost with logistic objective tends to
    be near-calibrated on balanced classes but can be over-confident on
    imbalanced ones. Reading the reliability curve first and deciding
    whether isotonic / sigmoid calibration is needed is a cleaner
    interview narrative than silently calibrating.
    """
    x_arr = np.asarray(features, dtype=np.float64)
    y_arr = np.asarray(labels).astype(np.int64)

    if x_arr.ndim != 2:
        raise ValueError(f"features must be 2-d, got shape {x_arr.shape}")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"features and labels must have same length, "
            f"got {x_arr.shape[0]} vs {y_arr.shape[0]}"
        )

    unique_labels = set(np.unique(y_arr).tolist())
    if not unique_labels <= {0, 1}:
        raise ValueError(f"labels must be binary 0/1, got values {sorted(unique_labels)}")
    if len(unique_labels) < 2:
        raise ValueError("labels must contain both classes (0 and 1)")

    if feature_names is None:
        feature_names_list = [f"f{i}" for i in range(x_arr.shape[1])]
    else:
        feature_names_list = list(feature_names)
        if len(feature_names_list) != x_arr.shape[1]:
            raise ValueError(
                f"feature_names length {len(feature_names_list)} does not match "
                f"feature count {x_arr.shape[1]}"
            )

    x_train, x_test, y_train, y_test = train_test_split(
        x_arr,
        y_arr,
        test_size=test_size,
        random_state=random_state,
        stratify=y_arr,
    )

    model = xgb.XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    y_pred_test = model.predict_proba(x_test)[:, 1]
    test_auc = float(roc_auc_score(y_test, y_pred_test))
    test_brier = float(brier_score_loss(y_test, y_pred_test))
    test_log_loss = float(log_loss(y_test, y_pred_test))

    gain_importance = tuple(float(v) for v in model.feature_importances_)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    # SHAP ≥ 0.44 on XGBClassifier returns a 2-d array; older versions
    # returned a length-2 list for binary classification. Handle both.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_mean_abs = tuple(float(v) for v in np.mean(np.abs(shap_values), axis=0))

    # `quantile` binning gives roughly equal-count bins which is robust
    # under imbalance. `uniform` would empty the top bins on rare events.
    mean_observed, mean_predicted = calibration_curve(
        y_test,
        y_pred_test,
        n_bins=calibration_bins,
        strategy="quantile",
    )

    return PropensityModelResult(
        feature_names=tuple(feature_names_list),
        test_auc=test_auc,
        test_brier=test_brier,
        test_log_loss=test_log_loss,
        feature_importances=gain_importance,
        shap_mean_abs=shap_mean_abs,
        calibration_mean_predicted=tuple(float(v) for v in mean_predicted),
        calibration_mean_observed=tuple(float(v) for v in mean_observed),
        n_train=int(x_train.shape[0]),
        n_test=int(x_test.shape[0]),
        n_positive_train=int(y_train.sum()),
        n_positive_test=int(y_test.sum()),
        random_state=random_state,
        model=model,
    )
