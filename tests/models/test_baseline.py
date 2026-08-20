from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_tire_model.models.baseline import (
    LinearTireDegradationModel,
    QuadraticTireDegradationModel,
)


def _linear_dataset(seed: int = 0) -> pd.DataFrame:
    """Two compounds with known, different linear degradation rates + small noise.

    SOFT: 0.15 s/lap degradation, HARD: 0.05 s/lap — soft degrades faster,
    matching real tire physics, so a per-compound model should recover
    noticeably different slopes for each.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for compound, rate in [("SOFT", 0.15), ("HARD", 0.05)]:
        for age in range(1, 15):
            true_delta = rate * age
            noisy_delta = true_delta + rng.normal(scale=0.01)
            rows.append(
                {
                    "tire_compound": compound,
                    "tire_age_laps": age,
                    "lap_time_delta_s": noisy_delta,
                    "is_valid_lap": True,
                }
            )
    return pd.DataFrame(rows)


def _quadratic_dataset(seed: int = 0) -> pd.DataFrame:
    """One compound with a genuine 'cliff': stays flat, then degrades sharply."""
    rng = np.random.default_rng(seed)
    rows = []
    for age in range(1, 20):
        true_delta = 0.01 * age**2  # accelerating degradation
        noisy_delta = true_delta + rng.normal(scale=0.02)
        rows.append(
            {
                "tire_compound": "MEDIUM",
                "tire_age_laps": age,
                "lap_time_delta_s": noisy_delta,
                "is_valid_lap": True,
            }
        )
    return pd.DataFrame(rows)


def test_linear_model_recovers_approximate_per_compound_rates():
    df = _linear_dataset()
    model = LinearTireDegradationModel().fit(df)

    summary = model.degradation_rate_summary().set_index("compound")

    assert summary.loc["SOFT", "degradation_rate_s_per_lap"] == pytest.approx(0.15, abs=0.02)
    assert summary.loc["HARD", "degradation_rate_s_per_lap"] == pytest.approx(0.05, abs=0.02)
    # Physically expected: soft degrades faster than hard.
    assert summary.loc["SOFT", "degradation_rate_s_per_lap"] > summary.loc["HARD", "degradation_rate_s_per_lap"]


def test_linear_model_predict_matches_fit_reasonably_well():
    df = _linear_dataset()
    model = LinearTireDegradationModel().fit(df)

    predictions = model.predict(df)
    residuals = df["lap_time_delta_s"].to_numpy() - predictions

    assert np.mean(np.abs(residuals)) < 0.05  # small noise, should fit tightly


def test_linear_model_falls_back_to_global_fit_for_unseen_compound():
    df = _linear_dataset()
    model = LinearTireDegradationModel().fit(df)

    unseen = pd.DataFrame({"tire_compound": ["INTERMEDIATE"], "tire_age_laps": [5]})
    prediction = model.predict(unseen)

    assert not np.isnan(prediction[0])  # global fallback should still produce a number


def test_linear_model_raises_on_insufficient_data():
    df = pd.DataFrame({"tire_compound": ["SOFT"], "tire_age_laps": [1], "lap_time_delta_s": [0.1]})

    with pytest.raises(ValueError, match="Not enough data"):
        LinearTireDegradationModel().fit(df)


def test_linear_model_excludes_invalid_laps_from_fit():
    df = _linear_dataset()
    contaminated = df.copy()
    # Inject a wildly wrong "invalid" lap (e.g. a Safety Car lap with huge delta)
    # that should be excluded from fitting via is_valid_lap=False.
    contaminated = pd.concat(
        [
            contaminated,
            pd.DataFrame(
                {
                    "tire_compound": ["SOFT"],
                    "tire_age_laps": [5],
                    "lap_time_delta_s": [50.0],  # nonsense value if included
                    "is_valid_lap": [False],
                }
            ),
        ],
        ignore_index=True,
    )

    model = LinearTireDegradationModel().fit(contaminated)
    summary = model.degradation_rate_summary().set_index("compound")

    # Rate should still be close to the true 0.15 — the outlier must have been excluded.
    assert summary.loc["SOFT", "degradation_rate_s_per_lap"] == pytest.approx(0.15, abs=0.03)


def test_quadratic_model_fits_better_than_linear_on_curved_data():
    df = _quadratic_dataset()

    linear_model = LinearTireDegradationModel().fit(df)
    quadratic_model = QuadraticTireDegradationModel().fit(df)

    linear_result = linear_model.evaluate(df)
    quadratic_result = quadratic_model.evaluate(df)

    assert quadratic_result.rmse < linear_result.rmse
    assert quadratic_result.r2 > linear_result.r2


def test_evaluate_result_str_is_readable():
    df = _linear_dataset()
    model = LinearTireDegradationModel().fit(df)

    result = model.evaluate(df)

    assert "MAE=" in str(result)
    assert "R2=" in str(result)
