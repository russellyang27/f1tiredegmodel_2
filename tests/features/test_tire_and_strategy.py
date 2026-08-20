from __future__ import annotations

import pandas as pd

from f1_tire_model.config import Settings
from f1_tire_model.features.strategy import StrategyFeatureExtractor
from f1_tire_model.features.tire import TireFeatureExtractor
from f1_tire_model.preprocessing.segmentation import StintInfo


def test_tire_extractor_masks_stint_length_when_incomplete():
    lap = pd.Series({"TyreLife": 3})
    ongoing_stint = StintInfo(
        driver_code="VER", stint_number=2, compound="HARD", lap_numbers=[6, 7, 8], is_complete=False
    )

    features = TireFeatureExtractor().extract(lap, ongoing_stint)

    assert features["tire_compound"] == "HARD"
    assert features["tire_age_laps"] == 3
    assert features["stint_number"] == 2
    assert features["stint_length_laps"] is None  # leakage guard


def test_tire_extractor_exposes_stint_length_when_complete():
    lap = pd.Series({"TyreLife": 5})
    completed_stint = StintInfo(
        driver_code="VER", stint_number=1, compound="MEDIUM", lap_numbers=[1, 2, 3, 4, 5], is_complete=True
    )

    features = TireFeatureExtractor().extract(lap, completed_stint)

    assert features["stint_length_laps"] == 5


def test_strategy_extractor_fuel_estimate_decreases_with_lap_number():
    settings = Settings(initial_fuel_kg=100.0, fuel_burn_rate_kg_per_lap=2.0)
    extractor = StrategyFeatureExtractor(settings=settings)

    lap1 = pd.Series({"LapNumber": 1, "PitInTime": pd.NaT, "PitOutTime": pd.NaT, "TrackStatus": "1"})
    lap10 = pd.Series({"LapNumber": 10, "PitInTime": pd.NaT, "PitOutTime": pd.NaT, "TrackStatus": "1"})

    f1 = extractor.extract(lap1)
    f10 = extractor.extract(lap10)

    assert f1["fuel_load_estimate_kg"] == 100.0
    assert f10["fuel_load_estimate_kg"] == 100.0 - 9 * 2.0
    assert f10["fuel_load_estimate_kg"] < f1["fuel_load_estimate_kg"]


def test_strategy_extractor_flags_pit_and_caution_laps():
    extractor = StrategyFeatureExtractor()
    pit_lap = pd.Series(
        {
            "LapNumber": 12,
            "PitInTime": pd.Timestamp("2024-01-01"),
            "PitOutTime": pd.NaT,
            "TrackStatus": "1",
        }
    )
    sc_lap = pd.Series(
        {"LapNumber": 20, "PitInTime": pd.NaT, "PitOutTime": pd.NaT, "TrackStatus": "4"}
    )

    assert extractor.extract(pit_lap)["is_pit_lap"] is True
    assert extractor.extract(sc_lap)["is_safety_car_lap"] is True
