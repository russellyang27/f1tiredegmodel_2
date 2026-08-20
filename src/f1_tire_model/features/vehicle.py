"""
Vehicle features: speed, brake energy, longitudinal acceleration, and
corner entry/apex/exit speeds.

Corner speeds require locating corners along the lap, which comes from
FastF1's `session.get_circuit_info().corners` — a DataFrame with one row
per corner and a `Distance` column marking (approximately) where that
corner sits along the lap. This is a single reference point per corner,
not exact apex/entry/exit geometry, so what's computed here is a proxy:

- entry speed  = speed at (corner_distance - corner_entry_offset_m)
- apex speed   = minimum speed within +/- corner_apex_window_m of corner_distance
- exit speed   = speed at (corner_distance + corner_exit_offset_m)

Because the schema stores one scalar per lap (not one value per corner),
this extractor reports the MEAN across all corners on track as the lap's
representative entry/apex/exit speed. That's a deliberate simplification
for a V1 baseline — a natural future extension (once you're doing feature
importance analysis in the ML phase) is exploding this into per-corner
columns or per-corner-type (high/low speed corner) aggregates instead of
collapsing to one number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1_tire_model.config import DEFAULT_SETTINGS, Settings
from f1_tire_model.features._telemetry_math import is_braking, longitudinal_acceleration_ms2
from f1_tire_model.features.base import FeatureExtractor
from f1_tire_model.telemetry import schema
from f1_tire_model.telemetry.cleaning import resample_by_distance

KPH_TO_MS = 1000.0 / 3600.0


class VehicleFeatureExtractor(FeatureExtractor):
    feature_group = "vehicle"

    def __init__(self, settings: Settings = DEFAULT_SETTINGS):
        self.settings = settings

    def extract(
        self,
        tel: pd.DataFrame,
        corners: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        Parameters
        ----------
        tel : cleaned, time-indexed telemetry for one lap.
        corners : `session.get_circuit_info().corners`, or None if
            unavailable (corner-speed features will be None in that case
            rather than raising — corner info isn't available for every
            session/track in FastF1).
        """
        speed = tel[schema.SPEED]
        accel = longitudinal_acceleration_ms2(tel)
        accel = accel[~np.isnan(accel)]

        features: dict[str, float] = {
            "avg_speed_kph": float(speed.mean()),
            "max_speed_kph": float(speed.max()),
            "brake_energy_estimate_kj": self._brake_energy_estimate_kj(tel),
            # Mean magnitude, distinct from Driver's peak-based max_acceleration_ms2 —
            # see the module docstring and the note in the step-by-step writeup
            # about resolving this overlap in the original spec.
            "longitudinal_accel_ms2": float(np.mean(np.abs(accel))) if len(accel) else None,
        }

        corner_speeds = self._corner_speeds(tel, corners)
        features.update(corner_speeds)

        return features

    def _brake_energy_estimate_kj(self, tel: pd.DataFrame) -> float | None:
        """Estimate kinetic energy dissipated under braking (kJ) for this lap.

        Physics: KE = 0.5 * m * v^2. For each pair of consecutive samples
        where the driver is braking and speed is decreasing, the energy
        dissipated is the drop in kinetic energy between the two samples.
        Summed over the lap, this gives a rough brake-energy figure that
        should scale sensibly with the project spec's "brake energy
        estimate" feature.

        This ignores drivetrain/engine braking contribution and treats mass
        as a fixed estimate (see `Settings.car_mass_kg`) — a simplification
        appropriate for a V1 physics baseline, not a substitute for real
        brake-force telemetry.
        """
        braking_mask = is_braking(tel).astype(bool)
        speed_ms = tel[schema.SPEED].to_numpy() * KPH_TO_MS

        if len(speed_ms) < 2:
            return None

        # Pair i, i+1: energy dissipated only counted while braking AND speed
        # is actually decreasing (avoids counting trail-braking-into-acceleration
        # transition samples as energy dissipation).
        v_from = speed_ms[:-1]
        v_to = speed_ms[1:]
        braking_pair_mask = braking_mask[:-1] & (v_to < v_from)

        delta_ke = 0.5 * self.settings.car_mass_kg * (v_from**2 - v_to**2)
        total_ke_j = float(np.sum(delta_ke[braking_pair_mask]))

        return total_ke_j / 1000.0  # J -> kJ

    def _corner_speeds(
        self, tel: pd.DataFrame, corners: pd.DataFrame | None
    ) -> dict[str, float | None]:
        empty = {
            "corner_entry_speed_kph": None,
            "apex_speed_kph": None,
            "corner_exit_speed_kph": None,
        }
        if corners is None or corners.empty or schema.DISTANCE not in tel.columns:
            return empty

        distance_tel = resample_by_distance(tel, step_meters=2.0)
        distance = distance_tel[schema.DISTANCE].to_numpy()
        speed = distance_tel[schema.SPEED].to_numpy()

        entry_speeds, apex_speeds, exit_speeds = [], [], []
        for _, corner in corners.iterrows():
            corner_distance = corner["Distance"]

            entry_speeds.append(
                self._speed_at(distance, speed, corner_distance - self.settings.corner_entry_offset_m)
            )
            apex_speeds.append(
                self._min_speed_in_window(
                    distance, speed, corner_distance, self.settings.corner_apex_window_m
                )
            )
            exit_speeds.append(
                self._speed_at(distance, speed, corner_distance + self.settings.corner_exit_offset_m)
            )

        def _mean_or_none(values: list[float | None]) -> float | None:
            valid = [v for v in values if v is not None]
            return float(np.mean(valid)) if valid else None

        return {
            "corner_entry_speed_kph": _mean_or_none(entry_speeds),
            "apex_speed_kph": _mean_or_none(apex_speeds),
            "corner_exit_speed_kph": _mean_or_none(exit_speeds),
        }

    @staticmethod
    def _speed_at(distance: np.ndarray, speed: np.ndarray, target_distance: float) -> float | None:
        if target_distance < distance.min() or target_distance > distance.max():
            return None  # corner too close to lap start/finish for this offset
        return float(np.interp(target_distance, distance, speed))

    @staticmethod
    def _min_speed_in_window(
        distance: np.ndarray, speed: np.ndarray, center: float, window: float
    ) -> float | None:
        mask = (distance >= center - window) & (distance <= center + window)
        if not mask.any():
            return None
        return float(speed[mask].min())
