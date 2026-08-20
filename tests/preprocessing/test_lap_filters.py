from __future__ import annotations

from f1_tire_model.preprocessing.segmentation import build_stints, valid_degradation_laps
import pandas as pd
import pytest

from f1_tire_model.preprocessing.lap_filters import (
    is_deleted,
    is_pit_lap,
    is_under_caution,
    is_lap_time_outlier,
    is_valid_degradation_lap,
)


def _lap(**overrides) -> pd.Series:
    base = {
        "Deleted": False,
        "PitInTime": pd.NaT,
        "PitOutTime": pd.NaT,
        "TrackStatus": "1",
        "LapTime": pd.Timedelta(seconds=90.0),
    }
    base.update(overrides)
    return pd.Series(base)


def test_is_deleted():
    assert is_deleted(_lap(Deleted=True)) is True
    assert is_deleted(_lap(Deleted=False)) is False


def test_is_pit_lap_true_on_pit_in_or_out():
    assert is_pit_lap(_lap(PitInTime=pd.Timestamp("2024-01-01"))) is True
    assert is_pit_lap(_lap(PitOutTime=pd.Timestamp("2024-01-01"))) is True
    assert is_pit_lap(_lap()) is False


@pytest.mark.parametrize("status,expected", [("1", False), ("12", False), ("14", True), ("16", True)])
def test_is_under_caution(status, expected):
    assert is_under_caution(_lap(TrackStatus=status)) is expected


def test_is_lap_time_outlier_relative_to_reference():
    reference = 90.0
    assert is_lap_time_outlier(_lap(LapTime=pd.Timedelta(seconds=95.0)), reference, factor=1.15) is False
    assert is_lap_time_outlier(_lap(LapTime=pd.Timedelta(seconds=110.0)), reference, factor=1.15) is True


def test_is_valid_degradation_lap_combines_all_checks():
    clean_lap = _lap()
    assert is_valid_degradation_lap(clean_lap, reference_lap_time_s=90.0, outlier_factor=1.15) is True

    pit_lap = _lap(PitInTime=pd.Timestamp("2024-01-01"))
    assert is_valid_degradation_lap(pit_lap, reference_lap_time_s=90.0, outlier_factor=1.15) is False

    sc_lap = _lap(TrackStatus="4")
    assert is_valid_degradation_lap(sc_lap, reference_lap_time_s=90.0, outlier_factor=1.15) is False

def test_lap_1_of_race_excluded_from_valid_degradation_laps():
    rows = []
    lap_times = {1: 95.0, 2: 85.2, 3: 85.0, 4: 85.4, 5: 85.1}
    for lap_num, lap_time in lap_times.items():
        rows.append({
            "Driver": "VER", "LapNumber": lap_num, "Stint": 1, "Compound": "MEDIUM",
            "TyreLife": lap_num, "LapTime": pd.Timedelta(seconds=lap_time),
            "PitInTime": pd.NaT, "PitOutTime": pd.NaT, "TrackStatus": "1", "Deleted": False,
        })
    laps_df = pd.DataFrame(rows)
    stint = build_stints(laps_df, "VER")[0]

    valid_laps = valid_degradation_laps(laps_df, stint, outlier_factor=1.15)

    assert 1 not in valid_laps
    assert set(valid_laps) == {2, 3, 4, 5}


def test_lap_1_of_a_later_stint_is_not_penalized():
    rows = []
    for lap_num, lap_time in [(20, 84.0), (21, 84.5), (22, 84.8)]:
        rows.append({
            "Driver": "VER", "LapNumber": lap_num, "Stint": 2, "Compound": "HARD",
            "TyreLife": lap_num - 19, "LapTime": pd.Timedelta(seconds=lap_time),
            "PitInTime": pd.NaT, "PitOutTime": pd.NaT, "TrackStatus": "1", "Deleted": False,
        })
    laps_df = pd.DataFrame(rows)
    stint = build_stints(laps_df, "VER")[0]

    valid_laps = valid_degradation_laps(laps_df, stint, outlier_factor=1.15)

    assert 20 in valid_laps