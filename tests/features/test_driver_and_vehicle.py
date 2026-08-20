from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_tire_model.config import Settings
from f1_tire_model.features.driver import DriverFeatureExtractor
from f1_tire_model.features.vehicle import VehicleFeatureExtractor


def _accelerate_then_brake_telemetry(n=40) -> pd.DataFrame:
    """Synthetic lap: accelerate to 300 kph over first half, brake to 100 over second half."""
    time = pd.to_timedelta(np.arange(n), unit="s")
    half = n // 2
    speed = np.concatenate([np.linspace(100, 300, half), np.linspace(300, 100, n - half)])
    brake = np.concatenate([np.zeros(half), np.full(n - half, 100.0)])
    throttle = np.concatenate([np.full(half, 100.0), np.zeros(n - half)])
    distance = np.cumsum(speed) * (1000 / 3600)  # rough distance accumulation

    return pd.DataFrame(
        {
            "Time": time,
            "Speed": speed,
            "Throttle": throttle,
            "Brake": brake,
            "nGear": np.full(n, 6),
            "Distance": distance,
            "DRS": np.zeros(n),
        }
    )


def test_driver_extractor_detects_braking_and_throttle():
    tel = _accelerate_then_brake_telemetry()

    features = DriverFeatureExtractor().extract(tel)

    assert features["avg_braking_decel_ms2"] > 0
    assert features["peak_braking_decel_ms2"] >= features["avg_braking_decel_ms2"]
    assert 0 <= features["avg_throttle_pct"] <= 100
    assert features["max_acceleration_ms2"] > 0  # acceleration phase present


def test_driver_extractor_consistency_none_for_single_lap():
    result = DriverFeatureExtractor().compute_stint_consistency([90.0])
    assert result["driver_consistency_std_s"] is None


def test_driver_extractor_consistency_computed_for_multiple_laps():
    result = DriverFeatureExtractor().compute_stint_consistency([90.0, 90.5, 91.0])
    assert result["driver_consistency_std_s"] == pytest.approx(np.std([90.0, 90.5, 91.0], ddof=1))


def test_vehicle_extractor_basic_speed_features():
    tel = _accelerate_then_brake_telemetry()

    features = VehicleFeatureExtractor().extract(tel, corners=None)

    assert features["max_speed_kph"] == pytest.approx(300.0, rel=0.05)
    assert features["avg_speed_kph"] < features["max_speed_kph"]
    assert features["brake_energy_estimate_kj"] > 0
    assert features["corner_entry_speed_kph"] is None  # no corners provided


def test_vehicle_extractor_brake_energy_scales_with_mass():
    tel = _accelerate_then_brake_telemetry()

    light = VehicleFeatureExtractor(settings=Settings(car_mass_kg=500.0)).extract(tel, corners=None)
    heavy = VehicleFeatureExtractor(settings=Settings(car_mass_kg=1000.0)).extract(tel, corners=None)

    assert heavy["brake_energy_estimate_kj"] == pytest.approx(
        light["brake_energy_estimate_kj"] * 2.0, rel=1e-6
    )


def test_vehicle_extractor_corner_speeds_with_synthetic_corner():
    tel = _accelerate_then_brake_telemetry()
    max_distance = tel["Distance"].max()

    # Place one corner near the middle of the lap.
    corners = pd.DataFrame({"Number": [1], "Distance": [max_distance / 2]})

    features = VehicleFeatureExtractor().extract(tel, corners=corners)

    assert features["apex_speed_kph"] is not None
    assert features["corner_entry_speed_kph"] is not None
    assert features["corner_exit_speed_kph"] is not None
