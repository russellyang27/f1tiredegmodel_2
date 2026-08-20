"""
Track features: circuit identity, corner count, average corner speed,
high/low-speed classification, and DRS zone count.

Unlike the other extractors, these are SESSION-level, not lap-level — a
circuit's corner count doesn't change lap to lap. This extractor is meant
to be called once per session and the result broadcast onto every lap row,
same pattern as DriverFeatureExtractor.compute_stint_consistency.

DRS zone counting assumption (flagged explicitly): FastF1 does not expose
a "number of DRS zones" field directly. This extractor approximates it by
counting contiguous distance ranges where a REFERENCE lap's DRS channel
shows an open/active state. Caveats:
- This only counts zones where DRS was actually activated on that lap,
  which requires eligibility (e.g. not the first laps, not too far behind
  another car in some sessions) — a practice/qualifying lap with DRS
  enabled throughout is a much more reliable reference than a race lap.
- Prefer passing a Qualifying lap as the reference when available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1_tire_model.config import DEFAULT_SETTINGS, Settings
from f1_tire_model.features.base import FeatureExtractor
from f1_tire_model.telemetry import schema

# FastF1 DRS channel values: 10, 12, 14 indicate DRS open/active;
# 0, 1, 8 indicate closed/unavailable. See FastF1 documentation for the
# full mapping — this project only needs the open/closed distinction.
_DRS_OPEN_VALUES = {10, 12, 14}


class TrackFeatureExtractor(FeatureExtractor):
    feature_group = "track"

    def __init__(self, settings: Settings = DEFAULT_SETTINGS):
        self.settings = settings

    def extract(
        self,
        circuit_name: str,
        corners: pd.DataFrame | None,
        reference_lap_telemetry: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        Parameters
        ----------
        circuit_name : e.g. session.event['Location'] or similar FastF1 field.
        corners : `session.get_circuit_info().corners`, or None if unavailable.
        reference_lap_telemetry : cleaned telemetry for a representative lap
            (ideally a Qualifying lap), used only for corner-speed and DRS-zone
            estimation. If None, those features return None rather than raising.
        """
        num_corners = len(corners) if corners is not None else None

        avg_corner_speed = None
        is_high_speed = None
        if corners is not None and reference_lap_telemetry is not None and num_corners:
            avg_corner_speed = self._avg_corner_speed(corners, reference_lap_telemetry)
            if avg_corner_speed is not None:
                is_high_speed = avg_corner_speed >= self.settings.high_speed_corner_threshold_kph

        num_drs_zones = None
        if reference_lap_telemetry is not None:
            num_drs_zones = self._count_drs_zones(reference_lap_telemetry)

        return {
            "circuit": circuit_name,
            "num_corners": num_corners,
            "avg_corner_speed_kph": avg_corner_speed,
            "is_high_speed_circuit": is_high_speed,
            "num_drs_zones": num_drs_zones,
        }

    @staticmethod
    def _avg_corner_speed(corners: pd.DataFrame, tel: pd.DataFrame) -> float | None:
        distance = tel[schema.DISTANCE].to_numpy()
        speed = tel[schema.SPEED].to_numpy()

        speeds = []
        for _, corner in corners.iterrows():
            d = corner["Distance"]
            if distance.min() <= d <= distance.max():
                speeds.append(float(np.interp(d, distance, speed)))

        return float(np.mean(speeds)) if speeds else None

    @staticmethod
    def _count_drs_zones(tel: pd.DataFrame) -> int | None:
        if schema.DRS not in tel.columns:
            return None

        drs_open = tel[schema.DRS].isin(_DRS_OPEN_VALUES).to_numpy()
        if not drs_open.any():
            return 0

        # Count contiguous True runs: a zone starts wherever `open` is True
        # and the previous sample was False (or it's the first sample).
        starts = drs_open & ~np.concatenate(([False], drs_open[:-1]))
        return int(starts.sum())
