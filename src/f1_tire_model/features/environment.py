"""
Environment features: air/track temperature, weather classification, rain.

FastF1 exposes weather as its own time series (`session.weather_data`),
sampled roughly every ~1 minute — NOT aligned to lap boundaries. This
extractor's job is specifically the nearest-time join between "when did
this lap happen" and "what was the weather at that time", which is a
distinct enough piece of logic (a merge-asof style match) to warrant
living in its own function rather than being inlined wherever environment
features are needed.
"""

from __future__ import annotations

import pandas as pd

from f1_tire_model.features.base import FeatureExtractor


class EnvironmentFeatureExtractor(FeatureExtractor):
    feature_group = "environment"

    def extract(self, lap: pd.Series, weather_df: pd.DataFrame) -> dict[str, float]:
        """
        Parameters
        ----------
        lap : one row of session.laps (needs a `Time` column — lap end time).
        weather_df : `session.weather_data`, with Time, AirTemp, TrackTemp,
            Rainfall columns.
        """
        sample = self._nearest_weather_sample(lap, weather_df)
        if sample is None:
            return {
                "air_temp_c": None,
                "track_temp_c": None,
                "weather": None,
                "is_raining": None,
            }

        is_raining = bool(sample.get("Rainfall", False))

        return {
            "air_temp_c": float(sample["AirTemp"]) if pd.notna(sample.get("AirTemp")) else None,
            "track_temp_c": float(sample["TrackTemp"]) if pd.notna(sample.get("TrackTemp")) else None,
            "weather": "rain" if is_raining else "dry",
            "is_raining": is_raining,
        }

    @staticmethod
    def _nearest_weather_sample(lap: pd.Series, weather_df: pd.DataFrame) -> pd.Series | None:
        lap_time = lap.get("Time")
        if pd.isna(lap_time) or weather_df.empty:
            return None

        time_diffs = (weather_df["Time"] - lap_time).abs()
        nearest_idx = time_diffs.idxmin()
        return weather_df.loc[nearest_idx]
