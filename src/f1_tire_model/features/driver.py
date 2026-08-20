"""
Driver features: braking behavior, throttle application, peak acceleration,
and cross-lap consistency.

Two different granularities live in this one class, which is worth reading
carefully before reusing this pattern elsewhere:

- `extract()` — genuinely per-lap: computed from one lap's telemetry alone.
- `compute_stint_consistency()` — a stint-level aggregate (needs multiple
  laps' times), NOT part of the `FeatureExtractor.extract()` contract. Its
  result should be broadcast identically onto every lap-row in that stint by
  the caller (the future dataset builder), not treated as if it varied
  lap-to-lap. This is called out explicitly because it's an easy thing to
  forget once there are many extractors and a builder gluing them together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1_tire_model.features._telemetry_math import is_braking, longitudinal_acceleration_ms2
from f1_tire_model.features.base import FeatureExtractor


class DriverFeatureExtractor(FeatureExtractor):
    feature_group = "driver"

    def extract(self, tel: pd.DataFrame) -> dict[str, float]:
        """
        Parameters
        ----------
        tel : cleaned, time-indexed telemetry for one lap
            (output of telemetry.cleaning.clean_telemetry).
        """
        accel = longitudinal_acceleration_ms2(tel)
        braking_mask = is_braking(tel)[1:]  # align with accel, which is len(tel)-1

        braking_decel = -accel[braking_mask]  # deceleration is positive-valued here
        braking_decel = braking_decel[~np.isnan(braking_decel)]

        avg_throttle = tel["Throttle"].mean()

        return {
            "avg_braking_decel_ms2": float(np.mean(braking_decel)) if len(braking_decel) else None,
            "peak_braking_decel_ms2": float(np.max(braking_decel)) if len(braking_decel) else None,
            "avg_throttle_pct": float(avg_throttle) if pd.notna(avg_throttle) else None,
            "max_acceleration_ms2": float(np.nanmax(accel)) if len(accel) else None,
        }

    def compute_stint_consistency(self, lap_times_s: list[float]) -> dict[str, float]:
        """Standard deviation of lap times (seconds) across a stint's valid
        degradation laps.

        Parameters
        ----------
        lap_times_s : lap times, in seconds, for the *valid degradation laps*
            of one stint (see preprocessing.segmentation.valid_degradation_laps)
            — pit/caution laps should already be excluded by the caller so
            this reflects genuine pace consistency, not strategy artifacts.

        Notes
        -----
        Requires at least 2 laps to be meaningful; returns None for a
        single-lap stint rather than a misleading 0.0.
        """
        if len(lap_times_s) < 2:
            return {"driver_consistency_std_s": None}
        return {"driver_consistency_std_s": float(np.std(lap_times_s, ddof=1))}
