"""
The core abstraction of the whole project.

Everything else you build (data loading, feature engineering, visualization,
validation) should be able to work with *any* class that implements
`TireDegradationModel`, whether it's today's physics/statistical baseline or
next semester's Random Forest / SVM / neural network.

Design notes
------------
- `fit` / `predict` mirrors the scikit-learn estimator API on purpose. When
  you take your ML course, wrapping an `sklearn.ensemble.RandomForestRegressor`
  in a class that implements this interface will be a ~20 line adapter, not a
  redesign.
- `feature_columns` is declared explicitly (not inferred from whatever
  columns happen to be in the DataFrame) so a model can validate its inputs
  and so different models can require different feature subsets without
  the pipeline needing to know the difference.
- `evaluate` lives on the base class (not overridden) because the scoring
  logic should be identical across physics and ML models — that's what
  makes them comparable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Standardized metrics so physics and ML models can be compared apples-to-apples."""

    mae: float
    rmse: float
    r2: float
    n_samples: int

    def __str__(self) -> str:
        return (
            f"MAE={self.mae:.4f}  RMSE={self.rmse:.4f}  "
            f"R2={self.r2:.4f}  (n={self.n_samples})"
        )


class TireDegradationModel(ABC):
    """Interface every tire degradation model must implement.

    Subclasses today: a physics/statistical baseline (e.g. linear or
    exponential lap-time-vs-tire-age fit per compound).
    Subclasses later: RandomForestTireModel, SVMTireModel, NeuralNetTireModel, etc.
    """

    #: Column names in the feature DataFrame this model expects as input.
    #: Subclasses must define this so the pipeline can validate feature
    #: availability before attempting to fit/predict.
    feature_columns: tuple[str, ...] = ()

    #: Column name in the feature DataFrame this model predicts.
    #: e.g. "lap_time_delta_s" or "remaining_tire_life_laps"
    target_column: str = ""

    @abstractmethod
    def fit(self, features_df: pd.DataFrame) -> "TireDegradationModel":
        """Fit the model on a features DataFrame. Must return self (sklearn convention)."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict the target for each row of features_df."""
        raise NotImplementedError

    def evaluate(self, features_df: pd.DataFrame) -> EvaluationResult:
        """Compute standard regression metrics against the true target column.

        Shared across all model types so baseline vs. ML results are directly
        comparable without reimplementing scoring per model.
        """
        if self.target_column not in features_df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in features_df; "
                f"available columns: {list(features_df.columns)}"
            )

        y_true = features_df[self.target_column].to_numpy(dtype=float)
        y_pred = self.predict(features_df)

        # Real datasets have rows with a missing target (e.g. a stint with no
        # valid degradation lap to set a baseline for lap_time_delta_s) or a
        # missing prediction (e.g. a lap with no tire_age_laps). These must
        # be dropped before scoring, not silently propagated -- one NaN in a
        # residual array poisons every metric (np.mean of an array
        # containing NaN is NaN), which is exactly what happened here.
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        n_dropped = len(y_true) - int(valid_mask.sum())
        if n_dropped > 0:
            logger.warning(
                "evaluate(): dropping %d/%d rows with a missing target or "
                "prediction before scoring.",
                n_dropped, len(y_true),
            )

        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        if len(y_true) == 0:
            raise ValueError(
                "No rows with both a valid target and a valid prediction to evaluate."
            )

        residuals = y_true - y_pred
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals**2)))

        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        return EvaluationResult(mae=mae, rmse=rmse, r2=r2, n_samples=len(y_true))

    def _validate_features(self, features_df: pd.DataFrame) -> None:
        """Helper subclasses can call at the top of fit()/predict()."""
        missing = set(self.feature_columns) - set(features_df.columns)
        if missing:
            raise ValueError(f"features_df is missing required columns: {sorted(missing)}")
