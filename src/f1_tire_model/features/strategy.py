"""
Strategy features: pit lap flag, fuel load estimate, Safety Car lap flag.

Reuses `preprocessing.lap_filters` for the pit/caution checks rather than
re-implementing them, so "is this a pit lap" has exactly one definition
across the whole codebase (used both for filtering out laps in
preprocessing and for labeling laps here).
"""

from __future__ import annotations

import pandas as pd

from f1_tire_model.config import DEFAULT_SETTINGS, Settings
from f1_tire_model.features.base import FeatureExtractor
from f1_tire_model.preprocessing.lap_filters import is_pit_lap, is_under_caution


class StrategyFeatureExtractor(FeatureExtractor):
    feature_group = "strategy"

    def __init__(self, settings: Settings = DEFAULT_SETTINGS):
        self.settings = settings

    def extract(self, lap: pd.Series) -> dict[str, float]:
        """
        Parameters
        ----------
        lap : one row of session.laps (needs LapNumber, PitInTime,
            PitOutTime, TrackStatus).
        """
        lap_number = lap.get("LapNumber")

        return {
            "is_pit_lap": is_pit_lap(lap),
            "fuel_load_estimate_kg": self._fuel_load_estimate_kg(lap_number),
            "is_safety_car_lap": is_under_caution(lap),
        }

    def _fuel_load_estimate_kg(self, lap_number: float | None) -> float | None:
        """Linear fuel burn model: starting load minus (laps completed * burn rate).

        See config.Settings docstring for the "linear burn" simplification
        this relies on — a clearly labeled V1 approximation, not a measured
        quantity (FastF1 does not expose actual fuel load).
        """
        if pd.isna(lap_number):
            return None

        laps_completed = max(int(lap_number) - 1, 0)
        remaining = self.settings.initial_fuel_kg - laps_completed * self.settings.fuel_burn_rate_kg_per_lap
        return float(max(remaining, 0.0))
