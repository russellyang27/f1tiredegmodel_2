"""
Tests for the TireDegradationModel interface itself.

This test doesn't test any specific model yet (we haven't written the
physics baseline in code form in this step) — it tests that the *contract*
behaves correctly: a minimal dummy implementation must be usable through
`evaluate()`. This matters because `evaluate()` is what both the physics
model and every future ML model will rely on for comparable metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_tire_model.models.base import TireDegradationModel


class _ConstantModel(TireDegradationModel):
    """Minimal concrete model for testing: always predicts a constant value.

    Standing in for a real model here keeps this test independent of
    whatever the first physics baseline implementation looks like.
    """

    feature_columns = ("tire_age_laps",)
    target_column = "lap_time_delta_s"

    def __init__(self, constant: float = 0.0):
        self.constant = constant

    def fit(self, features_df: pd.DataFrame) -> "_ConstantModel":
        self._validate_features(features_df)
        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        self._validate_features(features_df)
        return np.full(len(features_df), self.constant)


def test_fit_returns_self():
    model = _ConstantModel(constant=0.5)
    df = pd.DataFrame({"tire_age_laps": [1, 2, 3], "lap_time_delta_s": [0.4, 0.5, 0.6]})
    assert model.fit(df) is model


def test_evaluate_computes_expected_metrics():
    model = _ConstantModel(constant=0.5)
    df = pd.DataFrame({"tire_age_laps": [1, 2, 3], "lap_time_delta_s": [0.5, 0.5, 0.5]})

    result = model.evaluate(df)

    assert result.mae == pytest.approx(0.0)
    assert result.rmse == pytest.approx(0.0)
    assert result.n_samples == 3


def test_evaluate_raises_on_missing_target_column():
    model = _ConstantModel(constant=0.5)
    df = pd.DataFrame({"tire_age_laps": [1, 2, 3]})

    with pytest.raises(ValueError, match="Target column"):
        model.evaluate(df)


def test_validate_features_raises_on_missing_feature_column():
    model = _ConstantModel(constant=0.5)
    df = pd.DataFrame({"lap_time_delta_s": [0.5, 0.5, 0.5]})

    with pytest.raises(ValueError, match="missing required columns"):
        model.predict(df)
