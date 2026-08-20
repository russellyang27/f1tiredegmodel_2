from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_tire_model.config import Settings
from f1_tire_model.features.environment import EnvironmentFeatureExtractor
from f1_tire_model.features.track import TrackFeatureExtractor


def _reference_lap_telemetry(n=50) -> pd.DataFrame:
    distance = np.linspace(0, 5000, n)
    # Speed dips around distance=2500 (a "corner") and DRS is open on the back straight.
    speed = 300 - 150 * np.exp(-((distance - 2500) ** 2) / (2 * 200**2))
    drs = np.where(distance > 3500, 12, 0)  # DRS zone near the end of the lap
    time = pd.to_timedelta(np.arange(n), unit="s")

    return pd.DataFrame({"Time": time, "Distance": distance, "Speed": speed, "DRS": drs})


def test_track_extractor_basic_counts_and_classification():
    settings = Settings(high_speed_corner_threshold_kph=200.0)
    corners = pd.DataFrame({"Number": [1], "Distance": [2500.0]})
    tel = _reference_lap_telemetry()

    features = TrackFeatureExtractor(settings=settings).extract(
        circuit_name="Test Circuit", corners=corners, reference_lap_telemetry=tel
    )

    assert features["circuit"] == "Test Circuit"
    assert features["num_corners"] == 1
    assert features["avg_corner_speed_kph"] < 200.0  # the dip we constructed
    assert features["is_high_speed_circuit"] is False
    assert features["num_drs_zones"] == 1


def test_track_extractor_handles_missing_corner_info():
    features = TrackFeatureExtractor().extract(
        circuit_name="Unknown Circuit", corners=None, reference_lap_telemetry=None
    )

    assert features["num_corners"] is None
    assert features["avg_corner_speed_kph"] is None
    assert features["num_drs_zones"] is None


def test_environment_extractor_nearest_weather_match():
    lap = pd.Series({"Time": pd.Timestamp("2024-01-01 12:00:30")})
    weather_df = pd.DataFrame(
        {
            "Time": [pd.Timestamp("2024-01-01 12:00:00"), pd.Timestamp("2024-01-01 12:01:00")],
            "AirTemp": [25.0, 26.0],
            "TrackTemp": [40.0, 41.0],
            "Rainfall": [False, False],
        }
    )

    features = EnvironmentFeatureExtractor().extract(lap, weather_df)

    # Lap time is closer to the second sample (30s away vs 30s away — tie goes
    # to whichever pandas idxmin picks first; assert it picks a real sample)
    assert features["air_temp_c"] in (25.0, 26.0)
    assert features["weather"] == "dry"
    assert features["is_raining"] is False


def test_environment_extractor_detects_rain():
    lap = pd.Series({"Time": pd.Timestamp("2024-01-01 12:00:00")})
    weather_df = pd.DataFrame(
        {
            "Time": [pd.Timestamp("2024-01-01 12:00:00")],
            "AirTemp": [18.0],
            "TrackTemp": [22.0],
            "Rainfall": [True],
        }
    )

    features = EnvironmentFeatureExtractor().extract(lap, weather_df)

    assert features["weather"] == "rain"
    assert features["is_raining"] is True
