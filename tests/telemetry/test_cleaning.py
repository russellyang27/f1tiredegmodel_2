from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_tire_model.telemetry.cleaning import clean_telemetry, resample_by_distance
from f1_tire_model.telemetry.exceptions import TelemetryNotAvailableError


def _synthetic_telemetry(n=20) -> pd.DataFrame:
    """Build a small synthetic telemetry DataFrame shaped like FastF1 output."""
    time = pd.to_timedelta(np.arange(n), unit="s")
    return pd.DataFrame(
        {
            "Time": time,
            "Speed": np.linspace(100, 300, n),
            "Throttle": np.linspace(50, 100, n),
            "Brake": np.zeros(n),
            "nGear": np.full(n, 6),
            "Distance": np.linspace(0, 2000, n),
        }
    )


def test_clean_telemetry_raises_on_empty_input():
    with pytest.raises(TelemetryNotAvailableError):
        clean_telemetry(pd.DataFrame())


def test_clean_telemetry_raises_on_missing_channels():
    df = pd.DataFrame({"Time": pd.to_timedelta([0, 1], unit="s")})
    with pytest.raises(TelemetryNotAvailableError, match="missing required channels"):
        clean_telemetry(df)


def test_clean_telemetry_drops_duplicate_timestamps():
    df = _synthetic_telemetry()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # inject a duplicate

    cleaned = clean_telemetry(df)

    assert cleaned["Time"].duplicated().sum() == 0


def test_clean_telemetry_clips_implausible_speed():
    df = _synthetic_telemetry()
    df.loc[0, "Speed"] = 999.0  # sensor glitch

    cleaned = clean_telemetry(df)

    assert cleaned["Speed"].max() <= 380.0


def test_clean_telemetry_enforces_monotonic_distance():
    df = _synthetic_telemetry()
    df.loc[5, "Distance"] = 10.0  # inject a backwards glitch

    cleaned = clean_telemetry(df)

    assert cleaned["Distance"].is_monotonic_increasing


def test_resample_by_distance_produces_uniform_grid():
    df = clean_telemetry(_synthetic_telemetry())

    resampled = resample_by_distance(df, step_meters=100.0)

    diffs = resampled["Distance"].diff().dropna()
    assert np.allclose(diffs, 100.0)
    assert resampled["Speed"].min() >= df["Speed"].min() - 1e-6
    assert resampled["Speed"].max() <= df["Speed"].max() + 1e-6
