"""
Assemble a list of `LapFeatureRecord`s for one session.

This module is the integration point for everything built so far:
preprocessing (stint segmentation, lap validity), all six feature
extractors, and target computation. It intentionally does NOT import
`fastf1` — it consumes a `SessionInputs` (see data.session_adapter), so it
can be fully tested against plain synthetic DataFrames.

Resilience: a full-season build means thousands of per-lap telemetry
fetches, and a single corrupted lap or a FastF1 quirk on one stint
shouldn't abort a multi-hour run. Failures are caught at the lap and stint
level, logged, and the build continues — you end up with a dataset that
has a few gaps and a log of what was skipped, not a crashed script.
"""

from __future__ import annotations

import logging

import pandas as pd

from f1_tire_model.config import DEFAULT_SETTINGS, Settings
from f1_tire_model.data.session_adapter import SessionInputs
from f1_tire_model.datasets.schema import LapFeatureRecord
from f1_tire_model.features.driver import DriverFeatureExtractor
from f1_tire_model.features.environment import EnvironmentFeatureExtractor
from f1_tire_model.features.strategy import StrategyFeatureExtractor
from f1_tire_model.features.tire import TireFeatureExtractor
from f1_tire_model.features.track import TrackFeatureExtractor
from f1_tire_model.features.vehicle import VehicleFeatureExtractor
from f1_tire_model.preprocessing.lap_filters import is_valid_degradation_lap
from f1_tire_model.preprocessing.segmentation import (
    StintInfo,
    build_stints,
    stint_median_lap_time_s,
    valid_degradation_laps,
)
from f1_tire_model.telemetry.exceptions import TelemetryNotAvailableError

logger = logging.getLogger(__name__)

# Dicts with the same keys the corresponding extractor produces, used when
# telemetry for a lap is unavailable — keeps every record the same shape
# (all columns present, values None) instead of raising or dropping columns.
_NONE_DRIVER_FEATURES = {
    "avg_braking_decel_ms2": None,
    "peak_braking_decel_ms2": None,
    "avg_throttle_pct": None,
    "max_acceleration_ms2": None,
}
_NONE_VEHICLE_FEATURES = {
    "avg_speed_kph": None,
    "max_speed_kph": None,
    "brake_energy_estimate_kj": None,
    "longitudinal_accel_ms2": None,
    "corner_entry_speed_kph": None,
    "apex_speed_kph": None,
    "corner_exit_speed_kph": None,
}


class SessionDatasetBuilder:
    """Builds LapFeatureRecords for one session by running every extractor."""

    def __init__(self, settings: Settings = DEFAULT_SETTINGS):
        self.settings = settings
        self.tire_extractor = TireFeatureExtractor()
        self.driver_extractor = DriverFeatureExtractor()
        self.vehicle_extractor = VehicleFeatureExtractor(settings)
        self.track_extractor = TrackFeatureExtractor(settings)
        self.environment_extractor = EnvironmentFeatureExtractor()
        self.strategy_extractor = StrategyFeatureExtractor(settings)

    def build_session_records(
        self, inputs: SessionInputs, *, season: int, round_number: int, session_name: str
    ) -> list[LapFeatureRecord]:
        track_features = self._compute_track_features(inputs)

        records: list[LapFeatureRecord] = []
        drivers = sorted(inputs.laps_df["Driver"].dropna().unique())

        for driver_code in drivers:
            try:
                stints = build_stints(inputs.laps_df, driver_code)
            except Exception:
                logger.exception("Failed to build stints for driver %s — skipping driver.", driver_code)
                continue

            for stint in stints:
                records.extend(
                    self._build_stint_records(
                        inputs, stint, track_features, season, round_number, session_name
                    )
                )

        return records

    def _compute_track_features(self, inputs: SessionInputs) -> dict:
        reference_tel = None
        if inputs.reference_driver and inputs.reference_lap_number:
            try:
                reference_tel = inputs.telemetry_fetcher(
                    inputs.reference_driver, inputs.reference_lap_number
                )
            except TelemetryNotAvailableError as exc:
                logger.warning("Reference lap telemetry unavailable for track features: %s", exc)

        return self.track_extractor.extract(
            circuit_name=inputs.circuit_name,
            corners=inputs.corners,
            reference_lap_telemetry=reference_tel,
        )

    def _build_stint_records(
        self,
        inputs: SessionInputs,
        stint: StintInfo,
        track_features: dict,
        season: int,
        round_number: int,
        session_name: str,
    ) -> list[LapFeatureRecord]:
        try:
            valid_lap_numbers = set(
                valid_degradation_laps(inputs.laps_df, stint, self.settings.outlier_lap_time_factor)
            )
        except ValueError as exc:
            logger.warning(
                "Could not compute valid laps for %s stint %s: %s — treating all laps as invalid.",
                stint.driver_code,
                stint.stint_number,
                exc,
            )
            valid_lap_numbers = set()

        baseline_lap_time_s = self._baseline_lap_time(inputs.laps_df, stint, valid_lap_numbers)
        consistency_features = self._compute_consistency(inputs.laps_df, stint, valid_lap_numbers)

        stint_records = []
        for lap_number in stint.lap_numbers:
            try:
                record = self._build_lap_record(
                    inputs,
                    stint,
                    lap_number,
                    lap_number in valid_lap_numbers,
                    baseline_lap_time_s,
                    consistency_features,
                    track_features,
                    season,
                    round_number,
                    session_name,
                )
                stint_records.append(record)
            except Exception:
                logger.exception(
                    "Failed to build record for %s lap %s — skipping lap.",
                    stint.driver_code,
                    lap_number,
                )
                continue

        return stint_records

    @staticmethod
    def _baseline_lap_time(
        laps_df: pd.DataFrame, stint: StintInfo, valid_lap_numbers: set[int]
    ) -> float | None:
        """First valid degradation lap's time — the baseline `lap_time_delta_s` is measured against."""
        if not valid_lap_numbers:
            return None
        first_valid_lap = min(valid_lap_numbers)
        return _lap_time_seconds(laps_df, stint.driver_code, first_valid_lap)

    def _compute_consistency(
        self, laps_df: pd.DataFrame, stint: StintInfo, valid_lap_numbers: set[int]
    ) -> dict:
        lap_times = [
            t
            for lap_num in sorted(valid_lap_numbers)
            if (t := _lap_time_seconds(laps_df, stint.driver_code, lap_num)) is not None
        ]
        return self.driver_extractor.compute_stint_consistency(lap_times)

    def _build_lap_record(
        self,
        inputs: SessionInputs,
        stint: StintInfo,
        lap_number: int,
        is_valid: bool,
        baseline_lap_time_s: float | None,
        consistency_features: dict,
        track_features: dict,
        season: int,
        round_number: int,
        session_name: str,
    ) -> LapFeatureRecord:
        lap_rows = inputs.laps_df[
            (inputs.laps_df["Driver"] == stint.driver_code)
            & (inputs.laps_df["LapNumber"] == lap_number)
        ]
        lap = lap_rows.iloc[0]

        tire_features = self.tire_extractor.extract(lap, stint)

        try:
            tel = inputs.telemetry_fetcher(stint.driver_code, lap_number)
            driver_features = self.driver_extractor.extract(tel)
            vehicle_features = self.vehicle_extractor.extract(tel, inputs.corners)
        except TelemetryNotAvailableError as exc:
            logger.info(
                "Telemetry unavailable for %s lap %s: %s — driver/vehicle features set to None.",
                stint.driver_code,
                lap_number,
                exc,
            )
            driver_features = dict(_NONE_DRIVER_FEATURES)
            vehicle_features = dict(_NONE_VEHICLE_FEATURES)

        driver_features = {**driver_features, **consistency_features}

        environment_features = self.environment_extractor.extract(lap, inputs.weather_df)
        strategy_features = self.strategy_extractor.extract(lap)

        lap_time_s = _lap_time_seconds(inputs.laps_df, stint.driver_code, lap_number)
        lap_time_delta_s = (
            lap_time_s - baseline_lap_time_s
            if lap_time_s is not None and baseline_lap_time_s is not None
            else None
        )
        remaining_tire_life_laps = (
            stint.stint_length_laps - tire_features["tire_age_laps"]
            if stint.is_complete and tire_features["tire_age_laps"] is not None
            else None
        )

        return LapFeatureRecord(
            season=season,
            round_number=round_number,
            session_name=session_name,
            driver_code=stint.driver_code,
            lap_number=lap_number,
            is_valid_lap=is_valid,
            **tire_features,
            **driver_features,
            **vehicle_features,
            **track_features,
            **environment_features,
            **strategy_features,
            lap_time_s=lap_time_s,
            lap_time_delta_s=lap_time_delta_s,
            remaining_tire_life_laps=remaining_tire_life_laps,
        )


def _lap_time_seconds(laps_df: pd.DataFrame, driver_code: str, lap_number: int) -> float | None:
    rows = laps_df[(laps_df["Driver"] == driver_code) & (laps_df["LapNumber"] == lap_number)]
    if rows.empty:
        return None
    lap_time = rows.iloc[0].get("LapTime")
    if pd.isna(lap_time):
        return None
    return lap_time.total_seconds() if hasattr(lap_time, "total_seconds") else float(lap_time)
