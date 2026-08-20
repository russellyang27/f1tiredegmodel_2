from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_tire_model.config import Settings
from f1_tire_model.data.session_adapter import SessionInputs
from f1_tire_model.datasets.builder import SessionDatasetBuilder
from f1_tire_model.telemetry.exceptions import TelemetryNotAvailableError


def _synthetic_telemetry(n=30) -> pd.DataFrame:
    time = pd.to_timedelta(np.arange(n), unit="s")
    half = n // 2
    speed = np.concatenate([np.linspace(100, 280, half), np.linspace(280, 100, n - half)])
    brake = np.concatenate([np.zeros(half), np.full(n - half, 100.0)])
    throttle = np.concatenate([np.full(half, 100.0), np.zeros(n - half)])
    distance = np.cumsum(speed) * (1000 / 3600)

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


def _synthetic_laps_df() -> pd.DataFrame:
    """One driver, one complete 4-lap stint (SOFT), no pit/caution laps."""
    rows = []
    for lap_num in range(1, 5):
        rows.append(
            {
                "Driver": "VER",
                "LapNumber": lap_num,
                "Stint": 1,
                "Compound": "SOFT",
                "TyreLife": lap_num,
                "LapTime": pd.Timedelta(seconds=90.0 + lap_num * 0.3),
                "PitInTime": pd.NaT,
                "PitOutTime": pd.NaT,
                "TrackStatus": "1",
                "Deleted": False,
                "Time": pd.Timestamp("2024-01-01 12:00:00") + pd.Timedelta(minutes=lap_num),
            }
        )
    return pd.DataFrame(rows)


def _synthetic_weather_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": [pd.Timestamp("2024-01-01 12:00:00")],
            "AirTemp": [24.0],
            "TrackTemp": [38.0],
            "Rainfall": [False],
        }
    )


def _session_inputs(telemetry_fetcher=None) -> SessionInputs:
    if telemetry_fetcher is None:
        def telemetry_fetcher(driver_code, lap_number):
            return _synthetic_telemetry()

    return SessionInputs(
        laps_df=_synthetic_laps_df(),
        weather_df=_synthetic_weather_df(),
        corners=None,
        circuit_name="Test Circuit",
        reference_driver="VER",
        reference_lap_number=1,
        telemetry_fetcher=telemetry_fetcher,
    )


def test_build_session_records_produces_one_record_per_lap():
    builder = SessionDatasetBuilder()
    inputs = _session_inputs()

    records = builder.build_session_records(inputs, season=2024, round_number=1, session_name="R")

    assert len(records) == 4
    assert {r.lap_number for r in records} == {1, 2, 3, 4}
    assert all(r.driver_code == "VER" for r in records)
    assert all(r.tire_compound == "SOFT" for r in records)


def test_build_session_records_sets_valid_lap_flag():
    builder = SessionDatasetBuilder()
    inputs = _session_inputs()

    records = {r.lap_number: r for r in builder.build_session_records(
        inputs, season=2024, round_number=1, session_name="R"
    )}

    # Lap 1 is the race-start lap -- correctly invalid now.
    assert records[1].is_valid_lap is False
    assert all(records[n].is_valid_lap for n in [2, 3, 4])


def test_build_session_records_computes_lap_time_delta_relative_to_first_lap():
    builder = SessionDatasetBuilder()
    inputs = _session_inputs()

    records = {r.lap_number: r for r in builder.build_session_records(
        inputs, season=2024, round_number=1, session_name="R"
    )}

    # Baseline is now lap 2 (the first VALID lap -- lap 1 is the race start).
    assert records[2].lap_time_delta_s == pytest.approx(0.0)
    assert records[4].lap_time_delta_s == pytest.approx(0.6, abs=1e-6)  # 2 laps * 0.3s from the new baseline


def test_build_session_records_masks_remaining_tire_life_when_stint_incomplete():
    laps_df = _synthetic_laps_df()
    # Turn this into an ongoing (incomplete) stint by not giving it a "next stint"
    # marker — build_stints treats the max stint number as incomplete, which is
    # already the case here (only Stint=1 exists), so this should be masked.
    inputs = _session_inputs()
    inputs.laps_df = laps_df

    builder = SessionDatasetBuilder()
    records = builder.build_session_records(inputs, season=2024, round_number=1, session_name="R")

    assert all(r.remaining_tire_life_laps is None for r in records)


def test_build_session_records_handles_missing_telemetry_gracefully():
    def failing_telemetry_fetcher(driver_code, lap_number):
        raise TelemetryNotAvailableError("simulated failure")

    inputs = _session_inputs(telemetry_fetcher=failing_telemetry_fetcher)
    builder = SessionDatasetBuilder()

    records = builder.build_session_records(inputs, season=2024, round_number=1, session_name="R")

    # Records should still be produced (tire/strategy/environment features intact)
    assert len(records) == 4
    assert all(r.avg_speed_kph is None for r in records)
    assert all(r.avg_braking_decel_ms2 is None for r in records)
    assert all(r.tire_compound == "SOFT" for r in records)  # unaffected by telemetry failure


def test_build_session_records_excludes_pit_lap_from_valid_flag():
    laps_df = _synthetic_laps_df()
    laps_df.loc[laps_df["LapNumber"] == 2, "PitInTime"] = pd.Timestamp("2024-01-01")
    inputs = _session_inputs()
    inputs.laps_df = laps_df

    builder = SessionDatasetBuilder()
    records = {r.lap_number: r for r in builder.build_session_records(
        inputs, season=2024, round_number=1, session_name="R"
    )}

    assert records[2].is_valid_lap is False  # the injected pit lap
    assert records[3].is_valid_lap is True   # unaffected control -- proves the pit flag didn't leak elsewhere